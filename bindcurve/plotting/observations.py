from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets.aggregation import aggregate_responses
from bindcurve.plotting.common import (
    CurveSeries,
    DoseRepresentation,
    ErrorStyle,
)
from bindcurve.results import FitResult


def _aggregate_within_experiment(table: pd.DataFrame) -> pd.DataFrame:
    return aggregate_responses(
        table,
        by=["experiment_id", "concentration"],
        count_name="n_experiment_replicates",
    )


def _aggregate_grand_mean(experiment_means: pd.DataFrame) -> pd.DataFrame:
    return aggregate_responses(
        experiment_means, by=["concentration"], count_name="n_experiments"
    )


def _observation_table_for_fit(
    table: pd.DataFrame,
    fit: FitResult,
) -> pd.DataFrame:
    fit_table = table[table["compound_id"].astype(str) == str(fit.compound_id)]
    if fit.experiment_id is not None:
        fit_table = fit_table[
            fit_table["experiment_id"].astype(str) == str(fit.experiment_id)
        ]
    if fit_table.empty:
        return pd.DataFrame()
    if fit.experiment_id is None:
        return _aggregate_grand_mean(_aggregate_within_experiment(fit_table))
    return _aggregate_within_experiment(fit_table)


def _observation_groups_for_compound(
    table: pd.DataFrame,
    *,
    compound_id: str,
    dose_representation: DoseRepresentation,
) -> list[pd.DataFrame]:
    compound_table = table[table["compound_id"].astype(str) == str(compound_id)]
    if compound_table.empty:
        return []

    experiment_means = _aggregate_within_experiment(compound_table)
    if dose_representation == "mean":
        return [_aggregate_grand_mean(experiment_means)]

    return [
        group.reset_index(drop=True)
        for _, group in experiment_means.groupby("experiment_id", sort=True)
    ]


def _plot_series_observation_group(
    ax: Axes,
    group: pd.DataFrame,
    *,
    label: str,
    color: object,
    show_markers: bool,
    marker_kind: str,
    marker_size: float,
    error_style: ErrorStyle,
    errorbar_linewidth: float,
    errorbar_capsize: float,
) -> bool:
    yerr = None
    if error_style == "sem":
        yerr = group["response_sem"].fillna(0.0)
    elif error_style == "sd":
        yerr = group["response_sd"].fillna(0.0)

    if not show_markers and yerr is None:
        return False

    errorbar_kwargs: dict[str, object] = {
        "fmt": marker_kind if show_markers else "none",
        "linestyle": "none",
        "label": label,
        "color": color,
        "ecolor": color,
    }
    if show_markers:
        errorbar_kwargs.update(
            {
                "markersize": marker_size,
                "markerfacecolor": color,
                "markeredgecolor": color,
            }
        )
    if yerr is not None:
        errorbar_kwargs.update(
            {
                "elinewidth": errorbar_linewidth,
                "capsize": errorbar_capsize,
            }
        )

    ax.errorbar(
        group["concentration"],
        group["response"],
        yerr=yerr,
        **errorbar_kwargs,
    )
    return True


def _plot_series_observations(
    ax: Axes,
    series: CurveSeries,
    *,
    label_on_curve: bool,
    show_markers: bool,
    marker_kind: str,
    marker_size: float,
    error_style: ErrorStyle,
    errorbar_linewidth: float,
    errorbar_capsize: float,
) -> None:
    """Render observation groups with one legend entry per logical series."""
    if not show_markers and error_style is None:
        return
    label_used = label_on_curve
    for group in series.observation_groups:
        label = "_nolegend_" if label_used else series.label
        plotted = _plot_series_observation_group(
            ax,
            group,
            label=label,
            color=series.color,
            show_markers=show_markers,
            marker_kind=marker_kind,
            marker_size=marker_size,
            error_style=error_style,
            errorbar_linewidth=errorbar_linewidth,
            errorbar_capsize=errorbar_capsize,
        )
        label_used = label_used or plotted
