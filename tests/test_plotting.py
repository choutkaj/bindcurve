import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from scipy.stats import norm
from scipy.stats import t as student_t

import bindcurve as bc
from bindcurve.plotting import _confidence_band


def ic50_curve(x, IC50=1.5, hill_slope=1.1):
    return 100.0 / (1.0 + (x / IC50) ** hill_slope)


def make_results(sigma=None, compounds=("a",)):
    x = np.logspace(-2, 2, 12)
    rows = []
    for compound_id in compounds:
        for experiment_id, factor, n_replicates in (("e1", 0.8, 2), ("e2", 1.25, 4)):
            for replicate in range(n_replicates):
                rows.append(
                    pd.DataFrame(
                        {
                            "compound_id": compound_id,
                            "experiment_id": experiment_id,
                            "concentration": x,
                            "response": ic50_curve(x, 1.5 * factor)
                            + np.cos(3 * x + replicate),
                        }
                    )
                )
    table = pd.concat(rows)
    if sigma is not None:
        table["sigma"] = sigma
    return bc.fit(
        bc.DoseResponseData(table), "ic50", fixed={"ymin": 0.0, "ymax": 100.0}
    )


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def curves(ax):
    return [line for line in ax.lines if not line.get_label().startswith("_")]


def test_plot_fits_draws_means_and_one_curve_per_fit(ax):
    results = make_results(compounds=("a", "b"))
    bc.plot_fits(results, compounds="b", colors=["red", "blue"], ax=ax)
    assert [line.get_label() for line in curves(ax)] == ["e1", "e2"]
    fit = next(
        f for f in results.fits if f.compound_id == "b" and f.experiment_id == "e1"
    )
    x, y = curves(ax)[0].get_data()
    np.testing.assert_allclose(y, fit.predict(x))
    assert to_rgba(curves(ax)[1].get_color()) == to_rgba("blue")
    assert ax.get_xscale() == "log"
    # Markers are replicate means of experiment e1 with sample-SD error bars.
    observed = results.data.table.query("compound_id == 'b' and experiment_id == 'e1'")
    means = observed.groupby("concentration")["response"]
    marker = ax.containers[0]
    np.testing.assert_allclose(marker.lines[0].get_ydata(), means.mean())
    bar_heights = [
        abs(s[1, 1] - s[0, 1]) / 2 for s in marker.lines[2][0].get_segments()
    ]
    np.testing.assert_allclose(bar_heights, means.std())


def test_plot_fits_labels_include_compounds_when_several_are_shown(ax):
    bc.plot_fits(
        make_results(compounds=("a", "b")), experiments="e2", errorbars=None, ax=ax
    )
    labels = [
        line.get_label() for line in ax.lines if not line.get_label().startswith("_")
    ]
    assert labels == ["a e2", "b e2"]


@pytest.mark.parametrize("sigma", [None, 1.0])
def test_confidence_band_is_the_delta_method_band(ax, sigma):
    results = make_results(sigma=sigma)
    fit = results.fits[0]
    x = np.logspace(-2, 2, 30)
    low, high = _confidence_band(fit, x, 0.9)

    # Independent delta method in the coordinates (log10 IC50, hill_slope).
    def curve(log_ic50, hill):
        return ic50_curve(x, 10**log_ic50, hill)

    log_ic50, hill, h = np.log10(fit.values["IC50"]), fit.values["hill_slope"], 1e-6
    J = np.column_stack(
        [
            (curve(log_ic50 + h, hill) - curve(log_ic50 - h, hill)) / (2 * h),
            (curve(log_ic50, hill + h) - curve(log_ic50, hill - h)) / (2 * h),
        ]
    )
    to_log = np.diag([1 / (np.log(10) * fit.values["IC50"]), 1.0])
    se = np.sqrt(np.einsum("ij,jk,ik->i", J, to_log @ fit.covariance @ to_log, J))
    # Known sigma: absolute covariance and a normal quantile; otherwise Student t.
    q = norm.ppf(0.95) if sigma else student_t.ppf(0.95, fit.n_data - 2)
    np.testing.assert_allclose(high - fit.predict(x), q * se, rtol=1e-6)
    np.testing.assert_allclose(fit.predict(x) - low, q * se, rtol=1e-6)

    bc.plot_fits(results, experiments="e1", errorbars=None, band=True, ax=ax)
    assert len(ax.collections) == 1


def test_plot_compounds_draws_grand_means_and_the_summary_curve(ax):
    results = make_results()
    bc.plot_compounds(results, errorbars="sem", ax=ax)
    x, y = ax.lines[-1].get_data()
    np.testing.assert_allclose(y, results.model.evaluate(x, **results.parameters("a")))
    assert ax.lines[-1].get_label() == "a"
    # Grand means weight experiments equally despite 2 vs 4 replicates.
    table = results.data.table
    experiment_means = table.groupby(["experiment_id", "concentration"])[
        "response"
    ].mean()
    grand = experiment_means.groupby("concentration")
    np.testing.assert_allclose(ax.containers[0].lines[0].get_ydata(), grand.mean())
    heights = [
        abs(s[1, 1] - s[0, 1]) / 2 for s in ax.containers[0].lines[2][0].get_segments()
    ]
    np.testing.assert_allclose(heights, grand.std() / np.sqrt(2))


def test_plot_residuals(ax):
    results = make_results(sigma=0.5)
    bc.plot_residuals(results, experiments="e1", standardized=True, ax=ax)
    fit = results.fits[0]
    observed = results.data.table.query("experiment_id == 'e1'")
    means = observed.groupby("concentration")["response"].mean()
    sigma_of_mean = 0.5 / np.sqrt(2)
    expected = (means.to_numpy() - fit.predict(means.index.to_numpy())) / sigma_of_mean
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], expected)
    with pytest.raises(ValueError, match="sigma"):
        bc.plot_residuals(make_results(), standardized=True, ax=ax)


def test_invalid_plot_options_are_rejected(ax):
    results = make_results()
    with pytest.raises(ValueError, match="errorbars"):
        bc.plot_fits(results, errorbars="range", ax=ax)
    with pytest.raises(ValueError, match="Expected 2 colors"):
        bc.plot_fits(results, colors=["red"], ax=ax)
    with pytest.raises(KeyError, match="'z'"):
        bc.plot_compounds(results, compounds="z", ax=ax)
