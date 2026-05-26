from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.plotting.common import (
    CurveSeries,
    XScale,
    _evaluate_fit,
    _filter_experiments,
    _get_axes,
    _make_plot_grid_from_table,
    _matching_fits,
    _normalize_error_style,
    _resolve_compound_ids,
    _resolve_series_colors,
)
from bindcurve.plotting.confidence import _plot_series_confidence_band
from bindcurve.plotting.observations import (
    _aggregate_observations,
    _observation_table_for_fit,
    _plot_series_observation_group,
)
from bindcurve.results import FitResult, FitResults


def _series_label_for_fit(
    fit: FitResult,
    *,
    n_compounds: int,
) -> str:
    experiment = fit.experiment_id or fit.model_name
    if n_compounds > 1:
        return f"{fit.compound_id} {experiment}"
    return str(experiment)


def _build_fit_series(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_ids: Sequence[str],
    experiments: Iterable[str] | None,
) -> list[CurveSeries]:
    fits = _matching_fits(
        results,
        compound_ids=compound_ids,
        experiments=experiments,
    )
    series = []
    for fit in fits:
        observations = _observation_table_for_fit(data, fit)
        observation_groups = [] if observations.empty else [observations]
        series.append(
            CurveSeries(
                label=_series_label_for_fit(fit, n_compounds=len(compound_ids)),
                compound_id=str(fit.compound_id),
                observation_groups=observation_groups,
                fit=fit,
            )
        )
    return series


def _plot_series_curve(
    ax: Axes,
    fit: FitResult,
    grid: np.ndarray,
    *,
    label: str,
    color: object,
    show_markers: bool,
    marker_kind: str,
    marker_size: float,
    curve_width: float,
    curve_style: str,
) -> None:
    line_kwargs: dict[str, object] = {
        "label": label,
        "color": color,
        "linewidth": curve_width,
        "linestyle": curve_style,
    }
    if show_markers:
        # Keep markers in the legend handle, not on the fitted line itself.
        line_kwargs.update(
            {
                "marker": marker_kind,
                "markersize": marker_size,
                "markerfacecolor": color,
                "markeredgecolor": color,
                "markevery": [],
            }
        )
    ax.plot(grid, _evaluate_fit(fit, grid), **line_kwargs)


def _residual_table_for_fit(
    table: pd.DataFrame,
    fit: FitResult,
    *,
    aggregate: bool,
) -> pd.DataFrame:
    fit_table = table[table["compound_id"].astype(str) == str(fit.compound_id)]
    if fit.experiment_id is not None:
        fit_table = fit_table[
            fit_table["experiment_id"].astype(str) == str(fit.experiment_id)
        ]

    if fit_table.empty:
        return pd.DataFrame(
            columns=["concentration", "response", "predicted", "residual"]
        )

    if aggregate:
        fit_table = _aggregate_observations(fit_table, by_experiment=False)

    plotted = fit_table.copy()
    predicted = np.asarray(
        _evaluate_fit(fit, plotted["concentration"].to_numpy()),
        dtype=float,
    )
    plotted["predicted"] = predicted
    plotted["residual"] = plotted["response"].to_numpy(dtype=float) - predicted
    return plotted


def plot_fit_lines(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    x_grid: np.ndarray | None = None,
    n_points: int = 300,
    xscale: XScale = "log",
    by_experiment: bool = True,
    label: str | None = None,
    **plot_kwargs,
) -> Axes:
    """Plot fitted dose-response curves onto an existing Matplotlib axes."""
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)

    # Use the first compound to determine the default grid if not provided,
    # or better, use the global range.
    grid = x_grid
    if grid is None:
        table = data.table[data.table["compound_id"].isin(resolved_compound_ids)]
        xmin = float(table["concentration"].min())
        xmax = float(table["concentration"].max())
        if xscale == "log":
            grid = np.logspace(np.log10(xmin), np.log10(xmax), n_points)
        else:
            grid = np.linspace(xmin, xmax, n_points)

    fits = _matching_fits(
        results,
        compound_ids=resolved_compound_ids,
        experiments=experiments,
    )

    seen_labels = set()
    for fit in fits:
        y = _evaluate_fit(fit, grid)

        plot_label = label
        if plot_label is None:
            experiment = fit.experiment_id or fit.model_name
            if len(resolved_compound_ids) > 1:
                plot_label = (
                    f"{fit.compound_id} {experiment}"
                    if by_experiment
                    else str(fit.compound_id)
                )
            elif by_experiment:
                plot_label = str(experiment)
            else:
                plot_label = str(fit.compound_id)

        if plot_label in seen_labels:
            plot_label = None
        elif plot_label is not None:
            seen_labels.add(plot_label)

        ax.plot(grid, y, label=plot_label, **plot_kwargs)

    if xscale is not None:
        ax.set_xscale(xscale)
    return ax


def plot_residuals(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    aggregate: bool = True,
    xscale: XScale = "log",
    zero_line: bool = True,
    label: str | None = None,
    zero_line_kwargs: dict | None = None,
    **scatter_kwargs,
) -> Axes:
    """Plot fit residuals against concentration on an existing axes.

    Residuals are computed as ``observed - predicted``. By default, technical
    replicates are aggregated in the same way as fitted observations.
    """
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)
    
    fits = _matching_fits(
        results,
        compound_ids=resolved_compound_ids,
        experiments=experiments,
    )

    if not fits:
        return ax

    default_scatter_kwargs = {"marker": "o"}
    default_scatter_kwargs.update(scatter_kwargs)

    for fit in fits:
        residuals = _residual_table_for_fit(data.table, fit, aggregate=aggregate)
        if residuals.empty:
            continue

        residual_label = label
        if residual_label is None:
            experiment = fit.experiment_id or fit.model_name
            if len(resolved_compound_ids) > 1:
                residual_label = f"{fit.compound_id} {experiment}"
            else:
                residual_label = str(experiment)

        ax.scatter(
            residuals["concentration"],
            residuals["residual"],
            label=residual_label,
            **default_scatter_kwargs,
        )

    if zero_line:
        default_zero_line_kwargs = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7}
        default_zero_line_kwargs.update(zero_line_kwargs or {})
        ax.axhline(0.0, **default_zero_line_kwargs)

    if xscale is not None:
        ax.set_xscale(xscale)
    return ax


def plot_fits(
    data: DoseResponseData,
    results: FitResults,
    *,
    compounds: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    show_markers: bool = True,
    marker_kind: str = "o",
    marker_size: float = 5.0,
    show_curves: bool = True,
    curve_width: float = 1.0,
    curve_style: str = "-",
    show_errorbars: bool = True,
    errorbar_kind: str = "sd",
    errorbar_linewidth: float = 1.0,
    errorbar_capsize: float = 3.0,
    colors: object | None = None,
    x_grid: np.ndarray | None = None,
    n_points: int = 300,
    xscale: XScale = "log",
    confidence_band: bool = False,
    confidence_level: float = 0.95,
    confidence_band_kwargs: dict | None = None,
) -> Axes:
    """Plot observations and fitted curves for each independent experiment.

    Optional confidence bands are covariance-based pointwise confidence bands
    around the fitted mean curve for each experiment-level fit.
    """
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compounds)
    error_style = _normalize_error_style(errorbar_kind) if show_errorbars else None
    series = _build_fit_series(
        data,
        results,
        compound_ids=resolved_compound_ids,
        experiments=experiments,
    )
    resolved_colors = _resolve_series_colors(ax, n_series=len(series), colors=colors)
    selected_table = data.table[
        data.table["compound_id"].astype(str).isin(resolved_compound_ids)
    ]
    selected_table = _filter_experiments(selected_table, experiments)

    for spec, color in zip(series, resolved_colors, strict=False):
        spec.color = color

    for spec in series:
        label_on_curve = show_curves and spec.fit is not None
        observations_visible = show_markers or error_style is not None
        if observations_visible:
            label_used = False
            for group in spec.observation_groups:
                group_label = "_nolegend_"
                if not label_on_curve and not label_used:
                    group_label = spec.label
                plotted = _plot_series_observation_group(
                    ax,
                    group,
                    label=group_label,
                    color=spec.color,
                    show_markers=show_markers,
                    marker_kind=marker_kind,
                    marker_size=marker_size,
                    error_style=error_style,
                    errorbar_linewidth=errorbar_linewidth,
                    errorbar_capsize=errorbar_capsize,
                )
                if plotted and group_label != "_nolegend_":
                    label_used = True

        if spec.fit is None:
            continue
        if selected_table.empty:
            continue
        grid = _make_plot_grid_from_table(
            selected_table,
            x_grid=x_grid,
            n_points=n_points,
            xscale=xscale,
        )
        if confidence_band:
            _plot_series_confidence_band(
                ax,
                spec.fit,
                grid,
                color=spec.color,
                confidence_level=confidence_level,
                confidence_band_kwargs=confidence_band_kwargs,
            )
        if show_curves:
            _plot_series_curve(
                ax,
                spec.fit,
                grid,
                label=spec.label,
                color=spec.color,
                show_markers=show_markers,
                marker_kind=marker_kind,
                marker_size=marker_size,
                curve_width=curve_width,
                curve_style=curve_style,
            )

    if xscale is not None:
        ax.set_xscale(xscale)
    return ax
