"""Matplotlib plots of fitted dose-response curves."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from scipy.stats import t as student_t

from bindcurve.data import replicate_means
from bindcurve.results import FitResult, FitResults

ErrorBars = Literal["sd", "sem"] | None


def plot_fits(
    results: FitResults,
    *,
    compounds: str | Iterable[str] | None = None,
    experiments: str | Iterable[str] | None = None,
    errorbars: ErrorBars = "sd",
    band: bool = False,
    confidence: float = 0.95,
    colors: object = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot replicate means and the fitted curve of every experiment.

    Parameters
    ----------
    results
        Fitted results; failed fits are skipped.
    compounds, experiments
        Restrict the plot to these compounds and experiments.
    errorbars
        Spread of the technical replicates around each mean.
    band
        Draw a pointwise confidence band from each fit's covariance.
    confidence
        Confidence level of the band.
    colors
        One color, or one color per fit. Defaults to the axes color cycle.
    ax
        Axes to draw on; a new figure is created by default.
    """
    ax = ax or plt.subplots()[1]
    fits = _selected_fits(results, compounds, experiments)
    table = results.data.table
    for fit, color in zip(fits, _colors(ax, colors, len(fits)), strict=True):
        observed = table[
            (table["compound_id"] == fit.compound_id)
            & (table["experiment_id"] == fit.experiment_id)
        ]
        means = replicate_means(observed, ["concentration"])
        _errorbar(ax, means, errorbars, color)
        x = _grid(means["concentration"])
        if band:
            low, high = _confidence_band(fit, x, confidence)
            ax.fill_between(x, low, high, color=color, alpha=0.25, linewidth=0)
        ax.plot(x, fit.predict(x), color=color, label=_label(fit, fits))
    ax.set_xscale("log")
    return ax


def plot_compounds(
    results: FitResults,
    *,
    compounds: str | Iterable[str] | None = None,
    errorbars: ErrorBars = "sd",
    colors: object = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot one summary curve per compound.

    Markers are grand means of the experiment means, so every experiment counts
    equally; error bars show their spread across experiments. The curve is the
    model at `FitResults.parameters`, i.e. at the reported summary values.

    Parameters
    ----------
    results
        Fitted results.
    compounds
        Restrict the plot to these compounds.
    errorbars
        Spread of the experiment means around each grand mean.
    colors
        One color, or one color per compound.
    ax
        Axes to draw on; a new figure is created by default.
    """
    ax = ax or plt.subplots()[1]
    selected = _compounds(results, compounds)
    table = results.data.table
    for compound_id, color in zip(
        selected, _colors(ax, colors, len(selected)), strict=True
    ):
        observed = table[table["compound_id"] == compound_id]
        experiment_means = replicate_means(observed, ["experiment_id", "concentration"])
        means = replicate_means(experiment_means, ["concentration"])
        _errorbar(ax, means, errorbars, color)
        if any(fit.success for fit in results.fits if fit.compound_id == compound_id):
            x = _grid(means["concentration"])
            y = results.model.evaluate(x, **results.parameters(compound_id))
            ax.plot(x, y, color=color, label=compound_id)
    ax.set_xscale("log")
    return ax


def plot_residuals(
    results: FitResults,
    *,
    compounds: str | Iterable[str] | None = None,
    experiments: str | Iterable[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot observed minus fitted replicate means against concentration."""
    ax = ax or plt.subplots()[1]
    fits = _selected_fits(results, compounds, experiments)
    table = results.data.table
    for fit in fits:
        observed = table[
            (table["compound_id"] == fit.compound_id)
            & (table["experiment_id"] == fit.experiment_id)
        ]
        means = replicate_means(observed, ["concentration"])
        residual = means["response"] - fit.predict(means["concentration"].to_numpy())
        ax.scatter(means["concentration"], residual, label=_label(fit, fits))
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    return ax


def _compounds(results: FitResults, compounds: str | Iterable[str] | None) -> list[str]:
    """Selected compound IDs; unknown IDs raise KeyError."""
    data = results.data if compounds is None else results.data.select(compounds)
    return data.compounds


def _selected_fits(
    results: FitResults,
    compounds: str | Iterable[str] | None,
    experiments: str | Iterable[str] | None,
) -> list[FitResult]:
    compounds = _compounds(results, compounds)
    if isinstance(experiments, str):
        experiments = [experiments]
    return [
        fit
        for fit in results.fits
        if fit.success
        and fit.compound_id in compounds
        and (experiments is None or fit.experiment_id in experiments)
    ]


def _label(fit: FitResult, fits: list[FitResult]) -> str:
    """Experiment ID, prefixed by the compound when several are plotted."""
    if len({other.compound_id for other in fits}) > 1:
        return f"{fit.compound_id} {fit.experiment_id}"
    return fit.experiment_id


def _colors(ax: Axes, colors: object, n: int) -> list[object]:
    if colors is None:
        return [ax._get_lines.get_next_color() for _ in range(n)]
    if is_color_like(colors):
        return [colors] * n
    colors = list(colors)
    if len(colors) != n:
        raise ValueError(f"Expected {n} colors, got {len(colors)}.")
    return colors


def _grid(concentration: pd.Series, n: int = 200) -> np.ndarray:
    return np.geomspace(concentration.min(), concentration.max(), n)


def _errorbar(
    ax: Axes, means: pd.DataFrame, errorbars: ErrorBars, color: object
) -> None:
    if errorbars not in ("sd", "sem", None):
        raise ValueError("errorbars must be 'sd', 'sem' or None.")
    yerr = None if errorbars is None else means[errorbars].fillna(0.0).to_numpy()
    ax.errorbar(
        means["concentration"].to_numpy(),
        means["response"].to_numpy(),
        yerr=yerr,
        fmt="o",
        color=color,
        markersize=5,
        capsize=3,
    )


def _confidence_band(
    fit: FitResult, x: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise confidence band of the fitted curve from its covariance."""
    if fit.covariance is None:
        raise ValueError(
            f"{fit.compound_id} {fit.experiment_id} has no covariance for a band."
        )
    y = fit.predict(x)
    jacobian = np.empty((x.size, len(fit.free)))
    for i, name in enumerate(fit.free):
        value = fit.values[name]
        # Concentrations are stepped multiplicatively and stay positive.
        if fit.model.parameter(name).concentration:
            up, down = value * (1.0 + 1e-6), value / (1.0 + 1e-6)
        else:
            step = 1e-6 * max(abs(value), 1.0)
            up, down = value + step, value - step
        jacobian[:, i] = (
            fit.model.evaluate(x, **{**fit.values, name: up})
            - fit.model.evaluate(x, **{**fit.values, name: down})
        ) / (up - down)
    variance = np.einsum("ij,jk,ik->i", jacobian, fit.covariance, jacobian)
    # The covariance is scaled by the estimated residual variance: Student t.
    dof = fit.n_data - len(fit.free)
    multiplier = student_t.ppf(0.5 + confidence / 2.0, dof)
    half_width = multiplier * np.sqrt(np.maximum(variance, 0.0))
    return y - half_width, y + half_width
