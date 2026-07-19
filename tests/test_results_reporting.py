from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc
from bindcurve.modeling import IC50Model, ParameterSpec


class AmbiguousConcentrationModel(IC50Model):
    name = "ambiguous_concentrations"
    parameter_specs = IC50Model.parameter_specs + (
        ParameterSpec(
            "Kd",
            min=np.finfo(float).tiny,
            kind="concentration",
            scale="log10",
            log_name="logKd",
        ),
    )


def make_ambiguous_results() -> bc.FitResults:
    model = AmbiguousConcentrationModel()
    fits = tuple(
        bc.FitResult(
            model=model,
            compound_id="cmpd_a",
            experiment_id=f"exp{index}",
            parameters={
                "ymin": bc.ParameterEstimate("ymin", 0.0, vary=False),
                "amplitude": bc.ParameterEstimate(
                    "amplitude", 100.0, vary=False
                ),
                "IC50": bc.ParameterEstimate("IC50", 1.0 + index, stderr=0.1),
                "hill_slope": bc.ParameterEstimate(
                    "hill_slope", 1.0, vary=False
                ),
                "Kd": bc.ParameterEstimate("Kd", 2.0 + index, stderr=0.1),
            },
        )
        for index in range(1, 4)
    )
    return bc.FitResults(model=model, fit_results=fits)


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.8, hill_slope=1.0):
    return ymin + (ymax - ymin) / (1.0 + (x / ic50) ** hill_slope)


def make_multi_experiment_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    for experiment_id, offset in {"exp1": -0.03, "exp2": 0.0, "exp3": 0.04}.items():
        effective_ic50 = 1.8 * (10**offset)
        for concentration in concentrations:
            base_response = ic50_curve(concentration, ic50=effective_ic50)
            for replicate_id, noise in enumerate([-0.3, 0.0, 0.3], start=1):
                rows.append(
                    {
                        "compound_id": "cmpd_a",
                        "experiment_id": experiment_id,
                        "concentration": concentration,
                        "replicate_id": f"rep{replicate_id}",
                        "response": base_response + noise,
                    }
                )
    return bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))


def make_single_experiment_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    for concentration in concentrations:
        base_response = ic50_curve(concentration, ic50=1.8)
        for replicate_id, noise in enumerate([-0.3, 0.0, 0.3], start=1):
            rows.append(
                {
                    "compound_id": "cmpd_a",
                    "experiment_id": "exp1",
                    "concentration": concentration,
                    "replicate_id": f"rep{replicate_id}",
                    "response": base_response + noise,
                }
            )
    return bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))


def test_report_representation_both_labels_linear_and_log_faces():
    data = make_multi_experiment_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "amplitude": 100.0})

    report = results.report(
        representation="both",
        uncertainty="sd",
        rounding="decimals",
        places_mean=2,
        places_uncertainty=2,
        unit="uM",
    )

    assert len(report) == 1
    text = report.loc[0, "report"]
    assert "IC50:" in text
    assert "logIC50:" in text
    assert "uM" in text
    assert "±" in text
    assert "[" in text


def test_report_omits_missing_uncertainty_for_single_experiment():
    data = make_single_experiment_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "amplitude": 100.0})

    summary = results.summary()
    report = results.report(rounding="decimals", places_mean=2, places_uncertainty=2)

    assert np.isnan(summary.loc[0, "IC50_SD_lower"])
    assert np.isnan(summary.loc[0, "IC50_CI95_lower"])
    assert "[" not in report.loc[0, "report"]
    assert "±" not in report.loc[0, "report"]


def test_fit_summary_exposes_explicit_optimizer_and_failure_diagnostics():
    data = make_single_experiment_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "amplitude": 100.0})

    columns = set(results.fit_summary().columns)
    assert {
        "optimizer_message",
        "failure_stage",
        "error_type",
        "error_message",
    } <= columns


def test_report_auto_raises_for_multiple_reportable_quantities():
    results = make_ambiguous_results()

    with pytest.raises(
        ValueError,
        match="Multiple reportable concentration quantities",
    ):
        results.report()
