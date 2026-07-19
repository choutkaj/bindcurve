from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.plotting.common import (
    DoseRepresentation,
    ErrorStyle,
)
from bindcurve.results import FitResult


def _aggregate_observations(
    table: pd.DataFrame,
    *,
    by_experiment: bool,
) -> pd.DataFrame:
    experiment_means = _aggregate_within_experiment(table)
    if by_experiment:
        return experiment_means
    return _aggregate_grand_mean(experiment_means)


def _aggregate_within_experiment(table: pd.DataFrame) -> pd.DataFrame:
    grouped = table.groupby(
        ["experiment_id", "concentration"], as_index=False
    )["response"]
    aggregated = grouped.agg(
        response="mean",
        response_sd="std",
        n_experiment_replicates="count",
    )
    aggregated["response_sem"] = aggregated["response_sd"] / np.sqrt(
        aggregated["n_experiment_replicates"]
    )
    return aggregated.sort_values(["experiment_id", "concentration"])


def _aggregate_grand_mean(experiment_means: pd.DataFrame) -> pd.DataFrame:
    grouped = experiment_means.groupby("concentration", as_index=False)["response"]
    aggregated = grouped.agg(response="mean", response_sd="std", n_experiments="count")
    aggregated["response_sem"] = aggregated["response_sd"] / np.sqrt(
        aggregated["n_experiments"]
    )
    return aggregated.sort_values("concentration")

def _observation_table_for_fit(
    data: DoseResponseData,
    fit: FitResult,
) -> pd.DataFrame:
    fit_table = data.table[
        data.table["compound_id"].astype(str) == str(fit.compound_id)
    ]
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
    data: DoseResponseData,
    *,
    compound_id: str,
    dose_representation: DoseRepresentation,
) -> list[pd.DataFrame]:
    compound_table = data.table[
        data.table["compound_id"].astype(str) == str(compound_id)
    ]
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
