import numpy as np
import pandas as pd
import pytest
from references import COMPETITIVE, observations
from scipy.optimize import minimize_scalar

import bindcurve as bc

FIXED = {"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.2}


def ic50_curve(x, IC50=1.7, hill_slope=1.2):
    return 100.0 / (1.0 + (x / IC50) ** hill_slope)


def frame(x, y, **columns):
    return pd.DataFrame(
        {"compound_id": "a", "concentration": x, "response": y, **columns}
    )


def least_squares_ic50(x, y, sigma=1.0):
    """Independent one-parameter least squares on log10 IC50."""
    optimum = minimize_scalar(
        lambda log_ic50: np.sum(((y - ic50_curve(x, 10**log_ic50)) / sigma) ** 2),
        bounds=(-1.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return 10**optimum.x


def log_jacobian(x, IC50, sigma=1.0, step=1e-6):
    """d(weighted residual)/d(log10 IC50) by central differences."""
    up, down = ic50_curve(x, IC50 * 10**step), ic50_curve(x, IC50 * 10**-step)
    return -(up - down) / (2 * step) / sigma


def test_replicate_means_are_weighted_by_replicate_count():
    counts = np.array([1, 1, 1, 1, 4, 4, 4, 4])
    x = np.repeat(np.logspace(-2, 2, counts.size), counts)
    y = ic50_curve(x) + np.random.default_rng(3).normal(0.0, 3.0, x.size)
    fit = bc.fit(bc.DoseResponseData(frame(x, y)), "ic50", fixed=FIXED).fits[0]
    # A mean of n replicates carries n observations' worth of information.
    assert fit.values["IC50"] == pytest.approx(least_squares_ic50(x, y), rel=1e-6)
    means = frame(x, y).groupby("concentration")["response"].mean()
    unweighted = least_squares_ic50(means.index.to_numpy(), means.to_numpy())
    assert fit.values["IC50"] != pytest.approx(unweighted, rel=1e-4)


def test_known_sigma_gives_absolute_standard_errors_and_chi_square():
    x = np.logspace(-2, 2, 10)
    sigma = np.linspace(0.4, 1.3, x.size)
    noise = np.array([0.2, -0.4, 0.1, 0.5, -0.3, 0.6, -0.2, 0.3, -0.1, 0.4])
    y = ic50_curve(x) + noise
    fit = bc.fit(
        bc.DoseResponseData(frame(x, y, sigma=sigma)), "ic50", fixed=FIXED
    ).fits[0]

    IC50 = least_squares_ic50(x, y, sigma)
    assert fit.values["IC50"] == pytest.approx(IC50, rel=1e-6)
    residual = y - ic50_curve(x, IC50)
    assert fit.rss == pytest.approx(np.sum(residual**2), rel=1e-6)
    assert fit.chi_square == pytest.approx(np.sum((residual / sigma) ** 2), rel=1e-6)
    J = log_jacobian(x, IC50, sigma)
    assert fit.stderr["IC50"] == pytest.approx(
        np.log(10) * IC50 / np.sqrt(np.sum(J**2)), rel=1e-5
    )
    assert fit.warnings == ()


def test_unknown_sigma_scales_standard_errors_by_residual_scatter():
    x = np.logspace(-2, 2, 10)
    y = ic50_curve(x) + np.random.default_rng(5).normal(0.0, 2.0, x.size)
    fit = bc.fit(bc.DoseResponseData(frame(x, y)), "ic50", fixed=FIXED).fits[0]
    IC50 = least_squares_ic50(x, y)
    variance = np.sum((y - ic50_curve(x, IC50)) ** 2) / (x.size - 1)
    J = log_jacobian(x, IC50)
    assert fit.chi_square is None
    assert fit.stderr["IC50"] == pytest.approx(
        np.log(10) * IC50 * np.sqrt(variance / np.sum(J**2)), rel=1e-5
    )


def test_estimate_outside_the_tested_range_is_flagged():
    x = np.logspace(-3, -1, 8)
    data = bc.DoseResponseData(frame(x, ic50_curve(x, IC50=50.0)))
    with pytest.warns(UserWarning, match="1 of 1 fits have quality warnings"):
        results = bc.fit(data, "ic50", fixed=FIXED)
    fit = results.fits[0]
    assert fit.success
    assert fit.values["IC50"] == pytest.approx(50.0, rel=1e-4)
    assert "IC50 lies outside the tested concentration range." in fit.warnings
    assert results.summary().loc[0, "N_flagged"] == 1


def test_runaway_and_unidentifiable_fits_are_flagged_not_raised():
    x = np.logspace(-2, 2, 10)
    with pytest.warns(UserWarning, match="quality warnings"):
        # Every response above the fixed top plateau: IC50 runs upward.
        inactive = bc.fit(
            bc.DoseResponseData(frame(x, 101.0)), "ic50", fixed=FIXED
        ).fits[0]
    assert inactive.success and inactive.warnings

    x = np.logspace(-3, -1, 8)
    y = ic50_curve(x, IC50=50.0, hill_slope=1.0)
    y += np.random.default_rng(1).normal(0.0, 2.0, x.size)
    with pytest.warns(UserWarning, match="quality warnings"):
        # No inhibition is visible, so free plateaus and IC50 are not identifiable.
        unidentifiable = bc.fit(bc.DoseResponseData(frame(x, y)), "ic50").fits[0]
    assert unidentifiable.success and unidentifiable.warnings


def test_sigma_inconsistent_with_residual_scatter_is_flagged():
    x = np.logspace(-2, 2, 10)
    y = ic50_curve(x) + np.random.default_rng(2).normal(0.0, 1.0, x.size)
    with pytest.warns(UserWarning, match="quality warnings"):
        fit = bc.fit(
            bc.DoseResponseData(frame(x, y, sigma=0.01)), "ic50", fixed=FIXED
        ).fits[0]
    assert any("inconsistent with the supplied sigma" in w for w in fit.warnings)


def test_bounds_and_fixed_values_are_respected():
    x = np.logspace(-2, 2, 12)
    data = bc.DoseResponseData(frame(x, ic50_curve(x, IC50=30.0, hill_slope=2.0)))
    bounds = {"IC50": (0.1, 10.0), "hill_slope": (0.1, 1.5)}
    fit = bc.fit(data, "ic50", fixed={"ymin": 0.0, "ymax": 100.0}, bounds=bounds).fits[
        0
    ]
    assert fit.values["ymin"] == 0.0 and fit.values["ymax"] == 100.0
    assert fit.values["IC50"] == pytest.approx(10.0)
    assert 0.1 <= fit.values["hill_slope"] <= 1.5
    assert fit.free == ("IC50", "hill_slope")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model": "dir_specific"}, ValueError, r"requires fixed values for \['LsT'\]"),
        ({"fixed": {"amplitude": 1.0}}, KeyError, "amplitude"),
        ({"fixed": {"IC50": 0.0}}, ValueError, "IC50 must be positive"),
        (
            {"bounds": {"IC50": (-1.0, 10.0)}},
            ValueError,
            "Bounds of IC50 must be positive",
        ),
        ({"bounds": {"IC50": (10.0, 1.0)}}, ValueError, "below its upper bound"),
        (
            {"fixed": {"IC50": 1.0}, "bounds": {"IC50": (0.1, 10.0)}},
            ValueError,
            "both fixed and bounded",
        ),
        ({"errors": "ignore"}, ValueError, "errors must be"),
    ],
)
def test_invalid_settings_are_rejected_before_fitting(kwargs, error, message):
    x = np.logspace(-2, 2, 6)
    data = bc.DoseResponseData(frame(x, ic50_curve(x)))
    with pytest.raises(error, match=message):
        bc.fit(data, **{"model": "ic50", **kwargs})


def test_errors_are_raised_or_collected_per_experiment():
    x = np.logspace(-2, 2, 6)
    table = pd.concat(
        [
            frame(x, ic50_curve(x), experiment_id="good"),
            frame(x[:1], ic50_curve(x[:1]), experiment_id="short"),
        ]
    )
    data = bc.DoseResponseData(table)
    with pytest.raises(ValueError, match="needs more than 1 concentrations"):
        bc.fit(data, "ic50", fixed=FIXED)
    results = bc.fit(data, "ic50", fixed=FIXED, errors="collect")
    by_experiment = {fit.experiment_id: fit for fit in results.fits}
    assert by_experiment["good"].success
    assert not by_experiment["short"].success
    assert by_experiment["short"].message.startswith("ValueError: Fitting 1 parameters")
    assert results.summary().loc[0, ["N_fit", "N_failed"]].tolist() == [1, 1]


def test_all_fixed_parameters_evaluate_without_optimizing():
    x = np.logspace(-2, 2, 6)
    data = bc.DoseResponseData(frame(x, ic50_curve(x)))
    fit = bc.fit(data, "ic50", fixed={**FIXED, "IC50": 1.7}).fits[0]
    assert fit.success and fit.free == () and fit.covariance is None
    assert fit.rss == pytest.approx(0.0, abs=1e-20)
    assert fit.warnings == ()


@pytest.mark.parametrize(
    ("model", "fixed", "level", "seed"),
    [
        ("comp_3st_specific", COMPETITIVE, 100.0, 2),
        ("comp_4st_specific", {**COMPETITIVE, "Kd3": 0.5}, 0.0, 6),
    ],
)
def test_flat_competition_data_with_free_plateaus_is_flagged(model, fixed, level, seed):
    # Kd runs to an extreme, where free receptor is a tiny root of its balance.
    x, _ = observations(model, 2.0, fixed)
    y = level + np.random.default_rng(seed).normal(0.0, 2.0, x.size)
    with pytest.warns(UserWarning, match="quality warnings"):
        fit = bc.fit(bc.DoseResponseData(frame(x, y)), model, fixed=fixed).fits[0]
    assert fit.success and fit.warnings
