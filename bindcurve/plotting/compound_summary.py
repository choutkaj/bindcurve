from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.fitting import FitCalculator, FitSettings
from bindcurve.modeling import get_model
from bindcurve.plotting.common import (
    CurveSeries,
    DoseRepresentation,
    XScale,
    _filter_experiments,
    _get_axes,
    _make_plot_grid_from_table,
    _matching_fits,
    _normalize_dose_representation,
    _normalize_error_style,
    _resolve_compound_ids,
    _resolve_series_colors,
)
from bindcurve.plotting.fits import _plot_series_curve
from bindcurve.plotting.observations import (
    _aggregate_grand_mean,
    _aggregate_within_experiment,
    _observation_groups_for_compound,
    _plot_series_observation_group,
)
from bindcurve.results import FitResult, FitResults


def _infer_model_name(fits: Sequence[FitResult]) -> str | None:
    model_names = {fit.model_name for fit in fits}
    if not model_names:
        return None
    if len(model_names) != 1:
        raise ValueError("plot_compounds requires fit results from exactly one model.")
    return next(iter(model_names))


def _infer_fixed_parameters(fits: Sequence[FitResult]) -> dict[str, float]:
    fixed_parameters: dict[str, float] = {}
    if not fits:
        return fixed_parameters

    parameter_names = sorted({name for fit in fits for name in fit.parameters})
    for name in parameter_names:
        estimates = [fit.parameters[name] for fit in fits if name in fit.parameters]
        if not estimates or any(estimate.vary for estimate in estimates):
            continue

        values = np.asarray([estimate.value for estimate in estimates], dtype=float)
        if np.allclose(values, values[0], rtol=1.0e-9, atol=1.0e-12):
            fixed_parameters[name] = float(values[0])

    return fixed_parameters


def _infer_bounds(
    fits: Sequence[FitResult],
) -> dict[str, tuple[float | None, float | None]]:
    for fit in fits:
        if fit.lmfit_result is None:
            continue

        bounds: dict[str, tuple[float | None, float | None]] = {}
        for name, parameter in fit.lmfit_result.params.items():
            lower = None if not np.isfinite(parameter.min) else float(parameter.min)
            upper = None if not np.isfinite(parameter.max) else float(parameter.max)
            if lower is None and upper is None:
                continue
            bounds[name] = (lower, upper)
        return bounds

    return {}


def _make_master_fit_table(
    table: pd.DataFrame,
    *,
    compound_id: str,
) -> pd.DataFrame:
    grand_mean = _aggregate_grand_mean(_aggregate_within_experiment(table)).copy()
    grand_mean["compound_id"] = compound_id
    grand_mean["experiment_id"] = "grand_mean"
    grand_mean["replicate_id"] = "grand_mean"
    return grand_mean


def _master_fit_results(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_ids: Sequence[str],
    experiments: Iterable[str] | None,
) -> FitResults:
    filtered_table = data.table[
        data.table["compound_id"].astype(str).isin(compound_ids)
    ]
    filtered_table = _filter_experiments(filtered_table, experiments)
    if filtered_table.empty:
        return FitResults(fit_results=[])

    source_fits = _matching_fits(
        results,
        compound_ids=compound_ids,
        experiments=experiments,
    )
    model_name = _infer_model_name(source_fits)
    if model_name is None:
        return FitResults(fit_results=[])

    master_rows = []
    for compound_id in compound_ids:
        compound_rows = filtered_table[
            filtered_table["compound_id"].astype(str) == str(compound_id)
        ]
        if compound_rows.empty:
            continue
        master_rows.append(
            _make_master_fit_table(compound_rows, compound_id=str(compound_id))
        )

    if not master_rows:
        return FitResults(fit_results=[])

    summary_data = DoseResponseData.from_dataframe(
        pd.concat(master_rows, ignore_index=True),
        metadata=dict(data.metadata),
    )
    calculator = FitCalculator(
        model=get_model(model_name),
        settings=FitSettings(errors="collect"),
    )
    return calculator.fit(
        summary_data,
        compounds=list(compound_ids),
        bounds=_infer_bounds(source_fits),
        fixed=_infer_fixed_parameters(source_fits),
    )


def _build_compound_series(
    data: DoseResponseData,
    summary_results: FitResults,
    *,
    compound_ids: Sequence[str],
    dose_representation: DoseRepresentation,
) -> list[CurveSeries]:
    fit_by_compound = {
        str(fit.compound_id): fit for fit in summary_results.successful()
    }
    series = []
    for compound_id in compound_ids:
        observation_groups = _observation_groups_for_compound(
            data,
            compound_id=str(compound_id),
            dose_representation=dose_representation,
        )
        fit = fit_by_compound.get(str(compound_id))
        if not observation_groups and fit is None:
            continue
        series.append(
            CurveSeries(
                label=str(compound_id),
                compound_id=str(compound_id),
                observation_groups=observation_groups,
                fit=fit,
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
    and one base color by default. Grand-mean observations plus a plotting-only
    master fit are used for ``dose_representation="mean"``. Experiment-level
    means can be shown instead with ``dose_representation="experiments"`` while
    keeping the same one-curve-per-compound master fit.
    """
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compounds)
    representation = _normalize_dose_representation(dose_representation)
    error_style = _normalize_error_style(errorbar_kind) if show_errorbars else None
    summary_results = (
        _master_fit_results(
            data,
            results,
            compound_ids=resolved_compound_ids,
            experiments=None,
        )
        if show_curves
        else FitResults(fit_results=[])
    )
    series = _build_compound_series(
        data,
        summary_results,
        compound_ids=resolved_compound_ids,
        dose_representation=representation,
    )
    resolved_colors = _resolve_series_colors(ax, n_series=len(series), colors=colors)
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

        if show_curves and spec.fit is not None:
            compound_table = data.table[
                data.table["compound_id"].astype(str) == spec.compound_id
            ]
            if compound_table.empty:
                continue
            grid = _make_plot_grid_from_table(
                compound_table,
                x_grid=x_grid,
                n_points=n_points,
                xscale=xscale,
            )
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
