from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

import bindcurve as bc
from bindcurve.plotting.confidence import _fit_confidence_band, _get_covariance


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.5, hill_slope=1.1):
    return ymin + (ymax - ymin) / (1.0 + (x / ic50) ** hill_slope)


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
    )


def make_results(data: bc.DoseResponseData) -> bc.FitResults:
    return bc.fit(data, model="ic50", fixed={"ymin": 0.0, "amplitude": 100.0})


def errorbar_half_heights(ax: plt.Axes, *, container_index: int = 0) -> np.ndarray:
    container = ax.containers[container_index]
    barlinecols = container.lines[2]
    segments = barlinecols[0].get_segments()
    return np.asarray(
        [abs(segment[1, 1] - segment[0, 1]) / 2.0 for segment in segments],
        dtype=float,
    )


def legend_labels(ax: plt.Axes) -> list[str]:
    return [
        label
        for label in ax.get_legend_handles_labels()[1]
        if not label.startswith("_")
    ]


def observation_lines(ax: plt.Axes) -> list[Line2D]:
    return [
        container.lines[0]
        for container in ax.containers
        if container.lines[0] is not None
    ]


def curve_lines(ax: plt.Axes) -> list[Line2D]:
    return [line for line in ax.lines if str(line.get_linestyle()).lower() != "none"]


def test_plot_fits_couples_series_labels_and_colors():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()
    ax.set_xlabel("dose")
    ax.set_ylabel("signal")

    returned_ax = bc.plot_fits(data, results, ax=ax, n_points=50)

    assert returned_ax is ax
    assert ax.get_xlabel() == "dose"
    assert ax.get_ylabel() == "signal"
    assert legend_labels(ax) == ["exp1", "exp2"]
    assert len(observation_lines(ax)) == 2
    assert len(curve_lines(ax)) == 2
    for observation, curve in zip(observation_lines(ax), curve_lines(ax), strict=True):
        assert to_rgba(observation.get_color()) == to_rgba(curve.get_color())
        assert curve.get_marker() == "o"
        assert curve.get_markevery() == []
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_fits_supports_explicit_styling_args():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_fits(
        data,
        results,
        ax=ax,
        n_points=50,
        marker_kind="s",
        marker_size=4,
        curve_width=1.5,
        curve_style="--",
        show_errorbars=False,
    )

    assert returned_ax is ax
    assert len(ax.collections) == 0
    assert len(curve_lines(ax)) == 2
    assert all(line.get_marker() == "s" for line in curve_lines(ax))
    assert all(line.get_markevery() == [] for line in curve_lines(ax))
    assert all(line.get_linewidth() == 1.5 for line in curve_lines(ax))
    assert all(line.get_linestyle() == "--" for line in curve_lines(ax))
    plt.close(fig)


def test_plot_compounds_averages_experiment_level_predictions_without_refitting():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_compounds(data, results, ax=ax, n_points=50)

    assert legend_labels(ax) == ["cmpd_a"]
    assert len(observation_lines(ax)) == 1
    assert len(curve_lines(ax)) == 1
    curve = curve_lines(ax)[0]
    grid = np.asarray(curve.get_xdata(), dtype=float)
    predictions = np.stack(
        [
            fit.model.evaluate(
                grid,
                **{
                    name: estimate.value
                    for name, estimate in fit.parameters.items()
                },
            )
            for fit in results.successful()
            if fit.compound_id == "cmpd_a"
        ]
    )
    expected = np.mean(predictions, axis=0)
    np.testing.assert_allclose(
        curve.get_ydata(),
        expected,
    )
    plt.close(fig)


def test_plot_compounds_supports_sd_or_sem_error_bars():
    data = make_data()
    results = make_results(data)
    sem_fig, sem_ax = plt.subplots()
    sd_fig, sd_ax = plt.subplots()

    bc.plot_compounds(data, results, ax=sem_ax, errorbar_kind="sem")
    bc.plot_compounds(data, results, ax=sd_ax, errorbar_kind="sd")

    np.testing.assert_allclose(
        errorbar_half_heights(sd_ax),
        errorbar_half_heights(sem_ax) * np.sqrt(2.0),
    )
    plt.close(sem_fig)
    plt.close(sd_fig)


def test_plot_compounds_can_show_experiment_level_means_with_one_label():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_compounds(
        data,
        results,
        ax=ax,
        dose_representation="experiments",
        n_points=50,
    )

    assert legend_labels(ax) == ["cmpd_a"]
    assert len(ax.containers) == 2
    assert len(observation_lines(ax)) == 2
    assert len(curve_lines(ax)) == 1
    assert all(
        to_rgba(line.get_color()) == to_rgba(curve_lines(ax)[0].get_color())
        for line in observation_lines(ax)
    )
    plt.close(fig)


def test_plot_compounds_supports_uniform_color():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_compounds(data, results, ax=ax, colors="black")

    assert to_rgba(observation_lines(ax)[0].get_color()) == to_rgba("black")
    assert to_rgba(curve_lines(ax)[0].get_color()) == to_rgba("black")
    plt.close(fig)


def test_plot_fits_supports_color_lists_for_plotted_series():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_fits(data, results, ax=ax, colors=["red", "blue"])

    assert [to_rgba(line.get_color()) for line in curve_lines(ax)] == [
        to_rgba("red"),
        to_rgba("blue"),
    ]
    assert [to_rgba(line.get_color()) for line in observation_lines(ax)] == [
        to_rgba("red"),
        to_rgba("blue"),
    ]
    plt.close(fig)


def test_plot_fits_rejects_wrong_color_count():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="colors must contain exactly 2 entries"):
        bc.plot_fits(data, results, ax=ax, colors=["red"])
    plt.close(fig)


def test_plot_fits_supports_multi_compound_data_by_default():
    data = make_data()
    extra = data.table.copy()
    extra["compound_id"] = "cmpd_b"
    multi = bc.DoseResponseData.from_dataframe(pd.concat([data.table, extra]))
    results = bc.fit(
        multi,
        model="ic50",
        fixed={"ymin": 0.0, "amplitude": 100.0},
    )

    fig, ax = plt.subplots()
    bc.plot_fits(multi, results, ax=ax, n_points=50)

    assert legend_labels(ax) == [
        "cmpd_a exp1",
        "cmpd_a exp2",
        "cmpd_b exp1",
        "cmpd_b exp2",
    ]
    assert len(observation_lines(ax)) == 4
    assert len(curve_lines(ax)) == 4
    plt.close(fig)


def test_plot_fits_uses_each_fit_observation_range_for_its_curve_grid():
    rows = []
    for compound_id, concentrations, ic50 in [
        ("cmpd_a", [1.0, 10.0, 100.0], 10.0),
        ("cmpd_b", [0.001, 0.01, 0.1], 0.01),
    ]:
        for index, concentration in enumerate(concentrations):
            rows.append(
                {
                    "compound_id": compound_id,
                    "experiment_id": "exp1",
                    "concentration": concentration,
                    "replicate_id": f"rep{index}",
                    "response": ic50_curve(
                        concentration,
                        ic50=ic50,
                        hill_slope=1.0,
                    ),
                }
            )
    data = bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))
    results = bc.fit(
        data,
        fixed={"ymin": 0.0, "amplitude": 100.0, "hill_slope": 1.0},
    )
    fig, ax = plt.subplots()

    bc.plot_fits(data, results, ax=ax, n_points=20, show_errorbars=False)

    ranges = {
        line.get_label(): (line.get_xdata()[0], line.get_xdata()[-1])
        for line in curve_lines(ax)
    }
    assert ranges["cmpd_a exp1"] == pytest.approx((1.0, 100.0))
    assert ranges["cmpd_b exp1"] == pytest.approx((0.001, 0.1))
    plt.close(fig)


def test_plot_fits_can_select_experiment_subset():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_fits(data, results, ax=ax, experiments=["exp1"], n_points=50)

    assert legend_labels(ax) == ["exp1"]
    assert len(curve_lines(ax)) == 1
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


def test_plot_fits_does_not_accept_removed_wrapper_arguments():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(TypeError, match="show_asymptotes"):
        bc.plot_fits(data, results, ax=ax, show_asymptotes=True)
    with pytest.raises(TypeError, match="curve_points"):
        bc.plot_fits(data, results, ax=ax, curve_points=[(1.5, "IC50")])
    plt.close(fig)


def test_plot_curve_points_rejects_invalid_dict_spec():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="x"):
        bc.plot_curve_points(data, results, ax=ax, points=[{"label": "bad"}])
    plt.close(fig)


def test_plot_fits_can_add_confidence_band():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_fits(
        data,
        results,
        ax=ax,
        experiments=["exp1"],
        confidence_band=True,
        n_points=50,
    )

    assert returned_ax is ax
    assert len(ax.collections) >= 2
    assert legend_labels(ax) == ["exp1"]
    assert all(collection.get_label() != "band" for collection in ax.collections)
    bands = [
        collection
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    ]
    assert len(bands) == 1
    assert bands[0].get_alpha() == pytest.approx(0.25)
    assert np.max(bands[0].get_linewidths()) == pytest.approx(0.8)
    assert len(curve_lines(ax)) == 1
    plt.close(fig)


def test_fit_confidence_band_uses_student_t_multiplier():
    data = make_data()
    results = make_results(data)
    fit = results.successful()[0]
    grid = np.logspace(-2, 2, 7)
    finite_difference_step = 1.0e-2
    model = fit.model

    y, lower, upper = _fit_confidence_band(
        fit,
        grid,
        confidence_level=0.95,
        finite_difference_step=finite_difference_step,
    )

    variable_names, covariance = _get_covariance(fit)
    parameters = {name: estimate.value for name, estimate in fit.parameters.items()}
    jacobian = np.empty((grid.size, len(variable_names)), dtype=float)

    for index, name in enumerate(variable_names):
        value = parameters[name]
        stderr = fit.parameters[name].stderr
        assert stderr is not None
        step = finite_difference_step * stderr
        plus_parameters = dict(parameters)
        minus_parameters = dict(parameters)
        plus_parameters[name] = value + step
        minus_parameters[name] = value - step
        plus = model.evaluate(grid, **plus_parameters)
        minus = model.evaluate(grid, **minus_parameters)
        jacobian[:, index] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)

    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    assert fit.metrics is not None
    degrees_of_freedom = (
        fit.metrics.n_data - fit.metrics.n_varying_parameters
    )
    multiplier = float(student_t.ppf(0.975, df=degrees_of_freedom))
    expected_half_width = multiplier * np.sqrt(np.maximum(variance, 0.0))

    np.testing.assert_allclose(upper - y, expected_half_width, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(y - lower, expected_half_width, rtol=1.0e-6, atol=1.0e-9)


def test_fit_confidence_band_uses_one_sided_difference_at_parameter_bound():
    fit = make_results(make_data()).successful()[0]
    parameters = dict(fit.parameters)
    estimate = parameters["IC50"]
    parameters["IC50"] = replace(estimate, min=estimate.value)
    bounded_fit = replace(fit, parameters=parameters)

    y, lower, upper = _fit_confidence_band(
        bounded_fit,
        np.logspace(-2, 2, 9),
        confidence_level=0.95,
        finite_difference_step=1e-2,
    )

    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(lower <= y)
    assert np.all(y <= upper)


def test_plot_fits_rejects_invalid_confidence_level():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="confidence_level"):
        bc.plot_fits(
            data,
            results,
            ax=ax,
            experiments=["exp1"],
            confidence_band=True,
            confidence_level=1.5,
        )
    plt.close(fig)


def test_plot_fits_confidence_band_requires_covariance_matrix():
    data = make_data()
    results = make_results(data)
    fits = list(results.fit_results)
    fits[0] = replace(fits[0], covariance=None)
    results = bc.FitResults(model=results.model, fit_results=tuple(fits))
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="covariance"):
        bc.plot_fits(
            data,
            results,
            ax=ax,
            experiments=["exp1"],
            confidence_band=True,
        )
    plt.close(fig)


def test_plot_compounds_does_not_accept_removed_wrapper_arguments():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    with pytest.raises(TypeError, match="confidence_band"):
        bc.plot_compounds(data, results, ax=ax, confidence_band=True)
    with pytest.raises(TypeError, match="experiments"):
        bc.plot_compounds(data, results, ax=ax, experiments=["exp1"])
    with pytest.raises(TypeError, match="aggregate"):
        bc.plot_compounds(data, results, ax=ax, aggregate=False)
    plt.close(fig)


def test_plot_residuals_draws_aggregated_residuals_and_zero_line():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()
    ax.set_xlabel("dose")
    ax.set_ylabel("delta")

    returned_ax = bc.plot_residuals(data, results, ax=ax, experiments=["exp1"])

    assert returned_ax is ax
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    assert ax.lines[0].get_ydata()[0] == 0.0
    assert ax.get_xlabel() == "dose"
    assert ax.get_ylabel() == "delta"
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


def test_plot_residuals_returns_ax_if_no_matching_fits():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_residuals(data, results, ax=ax, experiments=["missing"])
    assert returned_ax is ax
    assert len(ax.collections) == 0
    plt.close(fig)
