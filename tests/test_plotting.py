from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.5, hill_slope=-1.1):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    for experiment_id, multiplier in {"exp1": 0.95, "exp2": 1.05}.items():
        for concentration in concentrations:
            response = ic50_curve(concentration, ic50=1.5 * multiplier)
            for replicate_id, noise in enumerate([-0.25, 0.25], start=1):
                rows.append(
                    {
                        "compound_id": "cmpd_a",
                        "experiment_id": experiment_id,
                        "concentration": concentration,
                        "replicate_id": f"rep{replicate_id}",
                        "response": response + noise,
                    }
                )
    return bc.DoseResponseData.from_dataframe(
        pd.DataFrame(rows),
        concentration_unit="uM",
        response_unit="percent",
    )


def make_results(data: bc.DoseResponseData) -> bc.FitResults:
    return bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})


def test_plot_observations_draws_on_existing_axes():
    data = make_data()
    fig, ax = plt.subplots()

    returned_ax = bc.plot_observations(data, ax=ax)

    assert returned_ax is ax
    assert len(ax.collections) > 0
    assert ax.get_xlabel() == "concentration (uM)"
    assert ax.get_ylabel() == "response (percent)"
    plt.close(fig)


def test_plot_fits_draws_fit_lines_on_existing_axes():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_fits(data, results, ax=ax, n_points=50)

    assert returned_ax is ax
    assert len(ax.lines) == 2
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_curves_draws_observations_and_fits():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_curves(
        data,
        results,
        ax=ax,
        n_points=50,
        observation_kwargs={"markersize": 4},
        fit_kwargs={"linewidth": 1.5},
    )

    assert returned_ax is ax
    assert len(ax.collections) > 0
    assert len(ax.lines) >= 2
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_curves_requires_compound_id_for_multi_compound_data():
    data = make_data()
    extra = data.table.copy()
    extra["compound_id"] = "cmpd_b"
    multi = bc.DoseResponseData.from_dataframe(pd.concat([data.table, extra]))
    results = bc.fit(multi, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="compound_id"):
        bc.plot_curves(multi, results, ax=ax)
    plt.close(fig)


def test_plot_curves_can_select_experiment_subset():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_curves(data, results, ax=ax, experiments=["exp1"], n_points=50)

    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_asymptotes_draws_horizontal_ymin_ymax_lines():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_asymptotes(data, results, ax=ax, experiments=["exp1"])

    assert returned_ax is ax
    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {"exp1 ymin", "exp1 ymax"}
    assert {line.get_linestyle() for line in ax.lines} == {"--"}
    plt.close(fig)


def test_plot_asymptotes_can_plot_single_parameter_without_labels():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_asymptotes(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        parameters=("ymax",),
        label=False,
    )

    assert len(ax.lines) == 1
    assert ax.lines[0].get_label().startswith("_child")
    plt.close(fig)


def test_plot_curve_points_draws_points_and_annotations():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_curve_points(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        points=[(1.5, "IC50")],
    )

    assert returned_ax is ax
    assert len(ax.collections) == 1
    assert [text.get_text() for text in ax.texts] == ["IC50"]
    plt.close(fig)


def test_plot_curve_points_accepts_dict_specs_and_appends_experiment_names():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_curve_points(
        data,
        results,
        ax=ax,
        points=[{"x": 1.5, "label": "point"}],
    )

    labels = {text.get_text() for text in ax.texts}
    assert labels == {"point (exp1)", "point (exp2)"}
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_curves_can_add_asymptotes_and_curve_points():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_curves(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        show_asymptotes=True,
        curve_points=[(1.5, "IC50")],
        n_points=50,
    )

    assert len(ax.lines) >= 3
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "IC50"
    plt.close(fig)


def test_plot_curve_points_rejects_invalid_dict_spec():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="x"):
        bc.plot_curve_points(data, results, ax=ax, points=[{"label": "bad"}])
    plt.close(fig)


def test_plot_confidence_bands_draws_fill_between_collection():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_confidence_bands(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        n_points=50,
        confidence_level=0.90,
    )

    assert returned_ax is ax
    assert len(ax.collections) == 1
    assert ax.collections[0].get_label() == "exp1 90% confidence band"
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_curves_can_add_confidence_band():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_curves(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        confidence_band=True,
        n_points=50,
        confidence_band_kwargs={"label": "band"},
    )

    assert returned_ax is ax
    assert len(ax.collections) >= 2
    assert any(collection.get_label() == "band" for collection in ax.collections)
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_confidence_bands_rejects_invalid_confidence_level():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="confidence_level"):
        bc.plot_confidence_bands(
            data,
            results,
            ax=ax,
            experiments=["exp1"],
            confidence_level=1.5,
        )
    plt.close(fig)


def test_plot_confidence_bands_requires_covariance_matrix():
    data = make_data()
    results = make_results(data)
    results.successful()[0].lmfit_result.covar = None
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="covariance"):
        bc.plot_confidence_bands(
            data,
            results,
            ax=ax,
            experiments=["exp1"],
        )
    plt.close(fig)


def test_plot_residuals_draws_aggregated_residuals_and_zero_line():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_residuals(data, results, ax=ax, experiments=["exp1"])

    assert returned_ax is ax
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    assert ax.lines[0].get_ydata()[0] == 0.0
    assert ax.get_xlabel() == "concentration (uM)"
    assert ax.get_ylabel() == "residual (percent)"
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_residuals_can_plot_raw_replicate_residuals():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_residuals(data, results, ax=ax, experiments=["exp1"], aggregate=False)

    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 24
    plt.close(fig)


def test_plot_residuals_can_disable_zero_line_and_use_linear_xscale():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_residuals(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        zero_line=False,
        xscale="linear",
    )

    assert len(ax.lines) == 0
    assert ax.get_xscale() == "linear"
    plt.close(fig)


def test_plot_residuals_requires_matching_successful_fits():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="No successful fits"):
        bc.plot_residuals(data, results, ax=ax, experiments=["missing"])
    plt.close(fig)
