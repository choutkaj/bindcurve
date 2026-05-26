from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like

from bindcurve.datasets import DoseResponseData
from bindcurve.modeling import get_model
from bindcurve.results import FitResult, FitResults

ErrorStyle = Literal["sem", "sd", None]
XScale = Literal["log", "linear", None]
DoseRepresentation = Literal["mean", "experiments"]


@dataclass
class CurveSeries:
    """One logical plotted series for high-level wrapper plots."""

    label: str
    compound_id: str
    observation_groups: list[pd.DataFrame]
    fit: FitResult | None = None
    color: object | None = None



def _get_axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created_ax = plt.subplots()
    return created_ax


def _resolve_compound_ids(
    data: DoseResponseData, compound_id: str | Iterable[str] | None
) -> list[str]:
    if compound_id is not None:
        if isinstance(compound_id, str):
            return [compound_id]
        return [str(cid) for cid in compound_id]
    return data.compounds


def _filter_experiments(
    table: pd.DataFrame,
    experiments: Iterable[str] | None,
) -> pd.DataFrame:
    if experiments is None:
        return table
    requested = {str(experiment) for experiment in experiments}
    return table[table["experiment_id"].astype(str).isin(requested)]


def _make_x_grid(
    data: DoseResponseData,
    *,
    compound_id: str,
    x_grid: np.ndarray | None,
    n_points: int,
    xscale: XScale,
) -> np.ndarray:
    if x_grid is not None:
        return np.asarray(x_grid, dtype=float)

    compound = data.select_compound(compound_id)
    xmin = float(compound.table["concentration"].min())
    xmax = float(compound.table["concentration"].max())
    if xscale == "log":
        return np.logspace(np.log10(xmin), np.log10(xmax), n_points)
    return np.linspace(xmin, xmax, n_points)


def _matching_fits(
    results: FitResults,
    *,
    compound_ids: Iterable[str],
    experiments: Iterable[str] | None,
) -> list[FitResult]:
    requested_compounds = set(compound_ids)
    requested_experiments = (
        None if experiments is None else {str(experiment) for experiment in experiments}
    )
    fits = []
    for fit in results.successful():
        if fit.compound_id not in requested_compounds:
            continue
        if (
            requested_experiments is not None
            and str(fit.experiment_id) not in requested_experiments
        ):
            continue
        fits.append(fit)
    return fits


def _evaluate_model(
    fit: FitResult,
    x: np.ndarray | float,
    parameters: dict[str, float],
) -> np.ndarray:
    model = get_model(fit.model_name)
    x_array = np.asarray(x, dtype=float)
    x_transformed = model.transform_x(x_array)
    return model.evaluate(x_transformed, **parameters)


def _evaluate_fit(fit: FitResult, x: np.ndarray | float) -> np.ndarray:
    parameters = {name: estimate.value for name, estimate in fit.parameters.items()}
    return _evaluate_model(fit, x, parameters)


def _normalize_error_style(error_style: str | None) -> ErrorStyle:
    if error_style is None:
        return None
    normalized = str(error_style).lower()
    if normalized not in {"sd", "sem"}:
        raise ValueError("errorbar_kind must be 'sd', 'sem', or None.")
    return normalized


def _normalize_dose_representation(
    dose_representation: str,
) -> DoseRepresentation:
    normalized = str(dose_representation).lower()
    if normalized not in {"mean", "experiments"}:
        raise ValueError("dose_representation must be 'mean' or 'experiments'.")
    return normalized


def _make_plot_grid_from_table(
    table: pd.DataFrame,
    *,
    x_grid: np.ndarray | None,
    n_points: int,
    xscale: XScale,
) -> np.ndarray:
    if x_grid is not None:
        return np.asarray(x_grid, dtype=float)

    xmin = float(table["concentration"].min())
    xmax = float(table["concentration"].max())
    if xscale == "log":
        return np.logspace(np.log10(xmin), np.log10(xmax), n_points)
    return np.linspace(xmin, xmax, n_points)


def _resolve_series_colors(
    ax: Axes,
    *,
    n_series: int,
    colors: object | None,
) -> list[object]:
    if n_series == 0:
        return []
    if colors is None:
        return [ax._get_lines.get_next_color() for _ in range(n_series)]
    if is_color_like(colors):
        return [colors] * n_series

    try:
        resolved = list(colors)
    except TypeError as exc:  # pragma: no cover - defensive typing guard
        raise TypeError(
            "colors must be a single Matplotlib color or a sequence of colors."
        ) from exc

    if len(resolved) != n_series:
        raise ValueError(
            f"colors must contain exactly {n_series} entries for the plotted series; "
            f"got {len(resolved)}."
        )
    if not all(is_color_like(color) for color in resolved):
        raise ValueError("Every entry in colors must be a valid Matplotlib color.")
    return resolved
