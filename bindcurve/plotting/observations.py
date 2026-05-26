from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.plotting.common import (
    DoseRepresentation,
    ErrorStyle,
    _filter_experiments,
    _get_axes,
    _resolve_compound_ids,
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

def plot_observations(
    data: DoseResponseData,
    *,
    compound_id: str | None = None,
    ax: Axes | None = None,
    aggregate: bool = True,
    by_experiment: bool = True,
    error: ErrorStyle = "sem",
    experiments: Iterable[str] | None = None,
    label: str | None = None,
    **errorbar_kwargs,
) -> Axes:
    """Plot dose-response observations onto an existing Matplotlib axes.

    Parameters
    ----------
    data
        Dose-response data to plot.
    compound_id
        Compound(s) to plot. If ``None`` (default), all compounds in ``data``
        are plotted.
    ax
        Existing Matplotlib axes. If omitted, a new figure and axes are created.
    aggregate
        If ``True``, responses are averaged before plotting. If ``False``, raw
        observations are plotted without error bars.
    by_experiment
        If aggregating, either keep one averaged series per independent
        experiment or collapse experiment-level means into one grand-mean
        series per compound.
    error
        Error bar to show for aggregated observations. Use ``"sem"``, ``"sd"``,
        or ``None``.
    experiments
        Optional subset of experiment identifiers to plot.
    label
        Optional label override. If omitted and plotting by experiment, each
        experiment is labeled separately.
    **errorbar_kwargs
        Additional keyword arguments passed to ``Axes.errorbar``.
    """
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)

    for cid in resolved_compound_ids:
        compound = data.select_compound(cid)
        table = _filter_experiments(compound.table, experiments)

        if table.empty:
            continue

        default_kwargs = {"fmt": "o", "linestyle": "none"}
        default_kwargs.update(errorbar_kwargs)

        if not aggregate:
            ax.errorbar(
                table["concentration"],
                table["response"],
                yerr=None,
                label=label,
                **default_kwargs,
            )
            continue

        aggregated = _aggregate_observations(table, by_experiment=by_experiment)
        groups: Iterable[tuple[str | None, pd.DataFrame]]
        if by_experiment:
            groups = aggregated.groupby("experiment_id", sort=True)
        else:
            groups = [(None, aggregated)]

        for experiment_id, group in groups:
            yerr = None
            if error == "sem":
                yerr = group["response_sem"].fillna(0.0)
            elif error == "sd":
                yerr = group["response_sd"].fillna(0.0)
            elif error is not None:
                raise ValueError("error must be 'sem', 'sd', or None.")

            plot_label = label
            if plot_label is None:
                if len(resolved_compound_ids) > 1:
                    plot_label = f"{cid} {experiment_id}" if by_experiment else str(cid)
                elif by_experiment:
                    plot_label = str(experiment_id)
                else:
                    plot_label = str(cid)

            # If we are plotting multiple experiments for the same compound in one call,
            # and by_experiment is True, each gets its own label.
            # If by_experiment is False, they are already aggregated into one group.
            
            ax.errorbar(
                group["concentration"],
                group["response"],
                yerr=yerr,
                label=plot_label,
                **default_kwargs,
            )

    return ax


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
