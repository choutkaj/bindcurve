from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.plotting.common import (
    CurveSeries,
    DoseRepresentation,
    XScale,
    _get_axes,
    _make_plot_grid_from_table,
    _normalize_dose_representation,
    _normalize_error_style,
    _plot_series_curve,
    _resolve_compound_ids,
    _resolve_series_colors,
)
from bindcurve.plotting.observations import (
    _observation_groups_for_compound,
    _plot_series_observations,
)
from bindcurve.results import FitResults


def _build_compound_series(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_ids: Sequence[str],
    dose_representation: DoseRepresentation,
) -> list[CurveSeries]:
    series = []
    table = data.table
    fits_by_compound = results._fits_by_compound()
    for compound_id in compound_ids:
        observation_groups = _observation_groups_for_compound(
            table,
            compound_id=str(compound_id),
            dose_representation=dose_representation,
        )
        fits = tuple(
            fit for fit in fits_by_compound.get(str(compound_id), []) if fit.success
        )
        if not observation_groups and not fits:
            continue
        series.append(
            CurveSeries(
                label=str(compound_id),
                compound_id=str(compound_id),
                observation_groups=observation_groups,
                fits=fits,
            )
        )
    return series


def plot_compounds(
    data: DoseResponseData,
    results: FitResults,
    *,
    compounds: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    show_markers: bool = True,
    marker_kind: str = "o",
    marker_size: float = 5.0,
    show_curves: bool = True,
    curve_width: float = 1.0,
    curve_style: str = "-",
    dose_representation: str = "mean",
    show_errorbars: bool = True,
    errorbar_kind: str = "sd",
    errorbar_linewidth: float = 1.0,
    errorbar_capsize: float = 3.0,
    colors: object | None = None,
    x_grid: np.ndarray | None = None,
    n_points: int = 300,
    xscale: XScale = "log",
) -> Axes:
    """Plot one summary dose-response curve per compound.

    A plotted series is one compound. Markers and fitted curve share one label
    and one base color by default. The curve is the model evaluated at
    ``FitResults.parameter_values()``, so it passes through the same summary
    concentration parameters that ``report()`` prints; failed fits are
    excluded. Plotting never fits or modifies data. Grand-mean or
    experiment-level observations are selected with ``dose_representation``.
    """
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compounds)
    representation = _normalize_dose_representation(dose_representation)
    error_style = _normalize_error_style(errorbar_kind) if show_errorbars else None
    series = _build_compound_series(
        data,
        results,
        compound_ids=resolved_compound_ids,
        dose_representation=representation,
    )
    resolved_colors = _resolve_series_colors(ax, n_series=len(series), colors=colors)
    for spec, color in zip(series, resolved_colors, strict=False):
        spec.color = color

    for spec in series:
        label_on_curve = show_curves and bool(spec.fits)
        _plot_series_observations(
            ax,
            spec,
            label_on_curve=label_on_curve,
            show_markers=show_markers,
            marker_kind=marker_kind,
            marker_size=marker_size,
            error_style=error_style,
            errorbar_linewidth=errorbar_linewidth,
            errorbar_capsize=errorbar_capsize,
        )

        if show_curves and spec.fits:
            compound_table = pd.concat(spec.observation_groups, ignore_index=True)
            if compound_table.empty:
                continue
            grid = _make_plot_grid_from_table(
                compound_table,
                x_grid=x_grid,
                n_points=n_points,
                xscale=xscale,
            )
            response = results.model.evaluate(
                grid, **results.parameter_values(spec.compound_id)
            )
            _plot_series_curve(
                ax,
                grid,
                response,
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
