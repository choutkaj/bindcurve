from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bindcurve as bc
from bindcurve.fitting.calculator import _FitCalculator
from bindcurve.modeling import IC50Model, ParameterSpec


def _ic50_response(
    concentration: np.ndarray,
    *,
    IC50: float = 1.7,
    hill_slope: float = 1.2,
) -> np.ndarray:
    return 100.0 / (1.0 + (concentration / IC50) ** hill_slope)


def make_uncertain_data(*, uncertainty: str = "sigma") -> bc.DoseResponseData:
    concentration = np.logspace(-2, 2, 10)
    sigma = np.linspace(0.4, 1.3, concentration.size)
    noise = np.asarray([0.2, -0.4, 0.1, 0.5, -0.3, 0.6, -0.2, 0.3, -0.1, 0.4])
    table = pd.DataFrame(
        {
            "compound_id": "cmpd_a",
            "experiment_id": "exp1",
            "concentration": concentration,
            "replicate_id": [f"rep{index}" for index in range(concentration.size)],
            "response": _ic50_response(concentration) + noise,
        }
    )
    if uncertainty == "sigma":
        table["sigma"] = sigma
    elif uncertainty == "weight":
        table["weight"] = 1.0 / sigma
    else:  # pragma: no cover - test helper guard
        raise ValueError(uncertainty)
    return bc.DoseResponseData.from_dataframe(table)


def fit_one_parameter(data: bc.DoseResponseData) -> bc.FitResults:
    return bc.fit(
        data,
        fixed={"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.2},
    )


def fitted_curve_count(ax: plt.Axes) -> int:
    return sum(str(line.get_linestyle()).lower() != "none" for line in ax.lines)


def test_sigma_fit_metrics_match_independent_residual_calculation():
    data = make_uncertain_data(uncertainty="sigma")
    fit = fit_one_parameter(data).successful()[0]
    observations = data.select_compound("cmpd_a").fit_observations()
    parameters = {
        name: estimate.value for name, estimate in fit.parameters.items()
    }
    predicted = fit.model.evaluate(
        observations["concentration"].to_numpy(dtype=float),
        **parameters,
    )
    residual = observations["response"].to_numpy(dtype=float) - predicted
    sigma = observations["sigma"].to_numpy(dtype=float)

    assert fit.metrics is not None
    assert fit.metrics.rss == pytest.approx(float(np.sum(residual**2)))
    assert fit.metrics.chi_square == pytest.approx(
        float(np.sum((residual / sigma) ** 2))
    )
    assert fit.metrics.reduced_chi_square == pytest.approx(
        fit.metrics.chi_square
        / (fit.metrics.n_data - fit.metrics.n_varying_parameters)
    )
    negative_twice_log_likelihood = fit.metrics.chi_square + float(
        np.sum(np.log(2.0 * np.pi) + 2.0 * np.log(sigma))
    )
    assert fit.metrics.aic == pytest.approx(
        negative_twice_log_likelihood
        + 2.0 * fit.metrics.n_varying_parameters
    )
    assert fit.metrics.bic == pytest.approx(
        negative_twice_log_likelihood
        + fit.metrics.n_varying_parameters * np.log(fit.metrics.n_data)
    )


def test_sigma_and_reciprocal_weight_are_equivalent():
    sigma_fit = fit_one_parameter(
        make_uncertain_data(uncertainty="sigma")
    ).successful()[0]
    weight_fit = fit_one_parameter(
        make_uncertain_data(uncertainty="weight")
    ).successful()[0]

    assert weight_fit.parameters["IC50"].value == pytest.approx(
        sigma_fit.parameters["IC50"].value,
        rel=1e-10,
    )
    assert weight_fit.metrics is not None
    assert sigma_fit.metrics is not None
    assert weight_fit.metrics.chi_square == pytest.approx(
        sigma_fit.metrics.chi_square,
        rel=1e-10,
    )


def test_all_fixed_parameters_are_evaluated_without_false_optimizer_failure():
    data = make_uncertain_data()

    fit = bc.fit(
        data,
        fixed={
            "ymin": 0.0,
            "ymax": 100.0,
            "IC50": 1.7,
            "hill_slope": 1.2,
        },
    ).fit_results[0]

    assert fit.success
    assert fit.variable_names == ()
    assert fit.covariance is None
    assert fit.optimizer_message == "No optimization: all parameters were fixed."
    assert fit.metrics is not None
    assert fit.metrics.n_varying_parameters == 0


def test_replicate_sigma_is_propagated_for_the_arithmetic_mean():
    data = bc.DoseResponseData.from_dataframe(
        pd.DataFrame(
            {
                "compound_id": ["cmpd_a", "cmpd_a"],
                "experiment_id": ["exp1", "exp1"],
                "concentration": [1.0, 1.0],
                "replicate_id": ["rep1", "rep2"],
                "response": [10.0, 14.0],
                "sigma": [2.0, 3.0],
            }
        )
    )

    observations = data.select_compound("cmpd_a").fit_observations()

    assert observations.loc[0, "response"] == pytest.approx(12.0)
    assert observations.loc[0, "sigma"] == pytest.approx(np.sqrt(13.0) / 2.0)
    assert observations.loc[0, "weight"] == pytest.approx(
        2.0 / np.sqrt(13.0)
    )


def test_standardized_residual_plot_uses_the_same_sigma_definition():
    data = make_uncertain_data(uncertainty="sigma")
    fit = fit_one_parameter(data).successful()[0]
    observations = data.select_compound("cmpd_a").fit_observations()
    parameters = {
        name: estimate.value for name, estimate in fit.parameters.items()
    }
    predicted = fit.model.evaluate(
        observations["concentration"].to_numpy(dtype=float),
        **parameters,
    )
    expected = (
        observations["response"].to_numpy(dtype=float) - predicted
    ) / observations["sigma"].to_numpy(dtype=float)
    fig, ax = plt.subplots()

    bc.plot_residuals(data, bc.FitResults(fit.model, (fit,)), ax=ax, standardized=True)

    observed = np.asarray(ax.collections[0].get_offsets()[:, 1], dtype=float)
    np.testing.assert_allclose(observed, expected)
    plt.close(fig)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("concentration", np.nan),
        ("concentration", np.inf),
        ("response", np.nan),
        ("response", -np.inf),
        ("sigma", 0.0),
        ("weight", np.inf),
    ],
)
def test_numeric_input_validation_rejects_nonfinite_or_nonpositive_values(
    column: str,
    value: float,
):
    row = {
        "compound_id": "cmpd_a",
        "experiment_id": "exp1",
        "concentration": 1.0,
        "replicate_id": "rep1",
        "response": 10.0,
    }
    row[column] = value

    with pytest.raises(ValueError):
        bc.DoseResponseData.from_dataframe(pd.DataFrame([row]))


def test_data_rejects_both_sigma_and_weight():
    table = make_uncertain_data().table
    table["weight"] = 1.0 / table["sigma"]

    with pytest.raises(ValueError, match="either sigma or weight"):
        bc.DoseResponseData.from_dataframe(table)


def test_data_rejects_duplicate_observation_identity():
    table = make_uncertain_data().table
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate observations"):
        bc.DoseResponseData.from_dataframe(table)


def test_blank_identifiers_are_rejected_before_string_coercion():
    table = make_uncertain_data().table
    table.loc[0, "experiment_id"] = "  "

    with pytest.raises(ValueError, match="experiment_id"):
        bc.DoseResponseData.from_dataframe(table)


def test_nested_metadata_and_exported_tables_cannot_mutate_canonical_data():
    data = bc.DoseResponseData.from_dataframe(
        make_uncertain_data().table,
        metadata={"assay": {"temperature": 25}},
    )
    metadata = data.metadata
    metadata["assay"]["temperature"] = 99
    table = data.table
    table.loc[0, "response"] = -999.0

    assert data.metadata == {"assay": {"temperature": 25}}
    assert data.table.loc[0, "response"] != -999.0


def test_wide_output_rejects_information_it_cannot_represent():
    table = make_uncertain_data().table.drop(columns="sigma")
    table["replicate_id"] = [f"technical-{index}" for index in range(len(table))]
    data = bc.DoseResponseData.from_dataframe(table)

    with pytest.raises(ValueError, match="positional replicate identifiers"):
        data.to_dataframe(format="wide")

    table["plate_well"] = "A01"
    data_with_extra_column = bc.DoseResponseData.from_dataframe(table)
    with pytest.raises(ValueError, match="cannot represent"):
        data_with_extra_column.to_dataframe(format="wide")


def test_wide_input_rejects_silently_dropped_columns():
    wide = pd.DataFrame(
        {
            "compound_id": ["cmpd_a"],
            "concentration": [1.0],
            "response_1": [10.0],
            "plate_well": ["A01"],
        }
    )

    with pytest.raises(ValueError, match="unsupported non-replicate columns"):
        bc.DoseResponseData.from_dataframe(wide, format="wide")


def test_fit_selectors_are_stably_deduplicated_and_empty_is_explicit():
    first = make_uncertain_data().table
    second = first.copy()
    second["compound_id"] = "cmpd_b"
    data = bc.DoseResponseData.from_dataframe(
        pd.concat([first, second], ignore_index=True)
    )

    results = bc.fit(
        data,
        compounds=["cmpd_b", "cmpd_b", "cmpd_a"],
        fixed={"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.2},
    )
    empty = bc.fit(
        data,
        compounds=[],
        fixed={"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.2},
    )

    assert [fit.compound_id for fit in results.fit_results] == ["cmpd_b", "cmpd_a"]
    assert empty.fit_results == ()
    assert empty.summary().empty


def test_custom_model_instance_flows_through_fit_results_and_plotting():
    class CustomIC50Model(IC50Model):
        name = "custom_ic50"

    model = CustomIC50Model()
    data = make_uncertain_data()
    results = bc.fit(
        data,
        model=model,
        fixed={"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.2},
    )
    fig, ax = plt.subplots()

    bc.plot_fits(data, results, ax=ax)

    assert results.model is model
    assert all(fit.model is model for fit in results.fit_results)
    assert fitted_curve_count(ax) == 1
    plt.close(fig)


def test_plot_compounds_never_refits(monkeypatch: pytest.MonkeyPatch):
    data = make_uncertain_data()
    results = fit_one_parameter(data)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("plotting attempted to fit")

    monkeypatch.setattr(_FitCalculator, "fit", fail_if_called)
    fig, ax = plt.subplots()

    bc.plot_compounds(data, results, ax=ax)

    assert fitted_curve_count(ax) == 1
    plt.close(fig)


def test_result_records_are_immutable_and_covariance_is_read_only():
    fit = fit_one_parameter(make_uncertain_data()).successful()[0]

    with pytest.raises(FrozenInstanceError):
        fit.compound_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        fit.parameters["IC50"] = fit.parameters["IC50"]  # type: ignore[index]
    assert fit.covariance is not None
    with pytest.raises(ValueError):
        fit.covariance[0, 0] = 0.0


def test_fit_results_reject_schema_drift_and_non_global_fixed_values():
    fit = fit_one_parameter(make_uncertain_data()).successful()[0]
    missing_parameter = dict(fit.parameters)
    missing_parameter.pop("IC50")
    incomplete = replace(
        fit,
        experiment_id="exp2",
        parameters=missing_parameter,
        covariance=None,
        variable_names=(),
    )
    with pytest.raises(ValueError, match="parameter schema"):
        bc.FitResults(fit.model, (incomplete,))

    changed_parameters = dict(fit.parameters)
    changed_parameters["ymax"] = replace(
        changed_parameters["ymax"],
        value=101.0,
    )
    changed_fixed = replace(
        fit,
        experiment_id="exp2",
        parameters=changed_parameters,
    )
    with pytest.raises(ValueError, match="global value"):
        bc.FitResults(fit.model, (fit, changed_fixed))


def test_model_contract_rejects_duplicate_specs_and_invalid_outputs():
    class DuplicateSpecModel(IC50Model):
        name = "duplicate_spec"
        parameter_specs = IC50Model.parameter_specs + (ParameterSpec("IC50"),)

    with pytest.raises(ValueError, match="duplicate parameter names"):
        DuplicateSpecModel()

    class WrongShapeModel(IC50Model):
        name = "wrong_shape"

        def _component_arrays(self, concentration, **params):
            return {"fraction_response": np.asarray([1.0])}

    parameters = {
        "ymin": 0.0,
        "ymax": 100.0,
        "IC50": 1.0,
        "hill_slope": 1.0,
    }
    with pytest.raises(ValueError, match="shape"):
        WrongShapeModel().evaluate(np.asarray([1.0, 2.0]), **parameters)

    class NonfiniteResponseModel(IC50Model):
        name = "nonfinite_response"

        def response_from_components(self, components, **params):
            return np.full_like(components["fraction_response"], np.nan)

    with pytest.raises(ValueError, match="finite"):
        NonfiniteResponseModel().evaluate(np.asarray([1.0]), **parameters)
