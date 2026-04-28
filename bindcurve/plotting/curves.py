from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.modeling import get_model
from bindcurve.results import FitResult, FitResults

ErrorStyle = Literal["sem", "sd", None]
XScale = Literal["log", "linear", None]
AsymptoteName = Literal["ymin", "ymax"]
CurvePointSpec = float | tuple[float, str] | dict[str, object]


@dataclass(frozen=True)
class CurvePoint:
    """Point to annotate on a fitted curve."""

    x: float
    label: str | None = None


def _get_axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created_ax = plt.subplots()
    return created_ax


def _resolve_compound_id(data: DoseResponseData, compound_id: str | None) -> str:
    if compound_id is not None:
        return str(compound_id)
    compounds = data.compounds
    if len(compounds) != 1:
        raise ValueError(
            "compound_id must be provided when data contain multiple compounds."
        )
    return compounds[0]


def _filter_experiments(
    table: pd.DataFrame,
    experiments: Iterable[str] | None,
) -> pd.DataFrame:
    if experiments is None:
        return table
    requested = {str(experiment) for experiment in experiments}
    return table[table["experiment_id"].astype(str).isin(requested)]


def _aggregate_observations(
    table: pd.DataFrame,
    *,
    by_experiment: bool,
) -> pd.DataFrame:
    group_cols = ["concentration"]
    if by_experiment:
        group_cols = ["experiment_id", "concentration"]

    grouped = table.groupby(group_cols, as_index=False)["response"]
    aggregated = grouped.agg(response="mean", response_sd="std", n="count")
    aggregated["response_sem"] = aggregated["response_sd"] / np.sqrt(aggregated["n"])
    return aggregated.sort_values(group_cols)


def _set_axis_labels(data: DoseResponseData, ax: Axes) -> None:
    concentration_label = "concentration"
    response_label = "response"
    if data.concentration_unit is not None:
        concentration_label = f"concentration ({data.concentration_unit})"
    if data.response_unit is not None:
        response_label = f"response ({data.response_unit})"
    ax.set_xlabel(concentration_label)
    ax.set_ylabel(response_label)


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
        Compound to plot. Required when ``data`` contain multiple compounds.
    ax
        Existing Matplotlib axes. If omitted, a new figure and axes are created.
    aggregate
        If ``True``, responses are averaged before plotting. If ``False``, raw
        observations are plotted without error bars.
    by_experiment
        If aggregating, aggregate separately within each independent experiment.
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
    resolved_compound_id = _resolve_compound_id(data, compound_id)
    compound = data.select_compound(resolved_compound_id)
    table = _filter_experiments(compound.table, experiments)

    if table.empty:
        raise ValueError("No observations remain after filtering.")

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
        _set_axis_labels(data, ax)
        return ax

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
        if plot_label is None and by_experiment:
            plot_label = str(experiment_id)

        ax.errorbar(
            group["concentration"],
            group["response"],
            yerr=yerr,
            label=plot_label,
            **default_kwargs,
        )

    _set_axis_labels(data, ax)
    return ax


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
    compound_id: str,
    experiments: Iterable[str] | None,
) -> list[FitResult]:
    requested = None if experiments is None else {str(experiment) for experiment in experiments}
    fits = []
    for fit in results.successful():
        if fit.compound_id != compound_id:
            continue
        if requested is not None and str(fit.experiment_id) not in requested:
            continue
        fits.append(fit)
    if not fits:
        raise ValueError("No successful fits match the requested filters.")
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


def _coerce_curve_points(points: Iterable[CurvePointSpec]) -> list[CurvePoint]:
    coerced = []
    for point in points:
        if isinstance(point, dict):
            if "x" not in point:
                raise ValueError("Curve point dictionaries must contain an 'x' key.")
            coerced.append(
                CurvePoint(
                    x=float(point["x"]),
                    label=None if point.get("label") is None else str(point["label"]),
                )
            )
        elif isinstance(point, tuple):
            if len(point) != 2:
                raise ValueError("Curve point tuples must be (x, label).")
            coerced.append(CurvePoint(x=float(point[0]), label=str(point[1])))
        else:
            coerced.append(CurvePoint(x=float(point), label=None))
    return coerced


def _confidence_multiplier(confidence_level: float) -> float:
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)


def _get_lmfit_covariance(fit: FitResult) -> tuple[list[str], np.ndarray]:
    lmfit_result = fit.lmfit_result
    if lmfit_result is None:
        raise ValueError("Cannot plot confidence bands without lmfit_result.")

    covariance = getattr(lmfit_result, "covar", None)
    if covariance is None:
        raise ValueError("Cannot plot confidence bands without a covariance matrix.")

    variable_names = list(getattr(lmfit_result, "var_names", []))
    if not variable_names:
        raise ValueError("Cannot plot confidence bands without lmfit variable names.")

    covariance_array = np.asarray(covariance, dtype=float)
    expected_shape = (len(variable_names), len(variable_names))
    if covariance_array.shape != expected_shape:
        raise ValueError(
            "Covariance matrix shape does not match lmfit variable names: "
            f"{covariance_array.shape} != {expected_shape}."
        )
    return variable_names, covariance_array


def _fit_confidence_band(
    fit: FitResult,
    x: np.ndarray,
    *,
    confidence_level: float,
    finite_difference_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variable_names, covariance = _get_lmfit_covariance(fit)
    z_value = _confidence_multiplier(confidence_level)
    x = np.asarray(x, dtype=float)
    parameters = {name: estimate.value for name, estimate in fit.parameters.items()}
    y = np.asarray(_evaluate_model(fit, x, parameters), dtype=float)

    jacobian = np.empty((x.size, len(variable_names)), dtype=float)
    for index, name in enumerate(variable_names):
        if name not in parameters:
            raise ValueError(f"Fit result is missing varying parameter {name!r}.")
        value = parameters[name]
        step = finite_difference_step * max(1.0, abs(value))
        plus_parameters = dict(parameters)
        minus_parameters = dict(parameters)
        plus_parameters[name] = value + step
        minus_parameters[name] = value - step
        y_plus = np.asarray(_evaluate_model(fit, x, plus_parameters), dtype=float)
        y_minus = np.asarray(_evaluate_model(fit, x, minus_parameters), dtype=float)
        jacobian[:, index] = (y_plus - y_minus) / (2.0 * step)

    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    band = z_value * np.sqrt(np.maximum(variance, 0.0))
    return y, y - band, y + band


def plot_fits(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    x_grid: np.ndarray | None = None,
    n_points: int = 300,
    xscale: XScale = "log",
    label: str | None = None,
    **plot_kwargs,
) -> Axes:
    """Plot fitted dose-response curves onto an existing Matplotlib axes."""
    ax = _get_axes(ax)
    resolved_compound_id = _resolve_compound_id(data, compound_id)
    grid = _make_x_grid(
        data,
        compound_id=resolved_compound_id,
        x_grid=x_grid,
        n_points=n_points,
        xscale=xscale,
    )

    fits = _matching_fits(
        results,
        compound_id=resolved_compound_id,
        experiments=experiments,
    )

    for fit in fits:
        y = _evaluate_fit(fit, grid)

        plot_label = label
        if plot_label is None:
            plot_label = str(fit.experiment_id or fit.model_name)

        ax.plot(grid, y, label=plot_label, **plot_kwargs)

    if xscale is not None:
        ax.set_xscale(xscale)
    _set_axis_labels(data, ax)
    return ax


def plot_confidence_bands(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    x_grid: np.ndarray | None = None,
    n_points: int = 300,
    xscale: XScale = "log",
    confidence_level: float = 0.95,
    finite_difference_step: float = 1.0e-6,
    label: str | None = None,
    **fill_between_kwargs,
) -> Axes:
    """Plot approximate covariance-based confidence bands for fitted curves."""
    ax = _get_axes(ax)
    resolved_compound_id = _resolve_compound_id(data, compound_id)
    grid = _make_x_grid(
        data,
        compound_id=resolved_compound_id,
        x_grid=x_grid,
        n_points=n_points,
        xscale=xscale,
    )
    fits = _matching_fits(
        results,
        compound_id=resolved_compound_id,
        experiments=experiments,
    )

    default_kwargs = {"alpha": 0.2, "linewidth": 0.0}
    default_kwargs.update(fill_between_kwargs)

    for fit in fits:
        _, lower, upper = _fit_confidence_band(
            fit,
            grid,
            confidence_level=confidence_level,
            finite_difference_step=finite_difference_step,
        )

        band_kwargs = dict(default_kwargs)
        if "label" not in band_kwargs:
            band_label = label
            if band_label is None:
                experiment = fit.experiment_id or fit.model_name
                percent = 100.0 * confidence_level
                band_label = f"{experiment} {percent:g}% confidence band"
            band_kwargs["label"] = band_label

        ax.fill_between(grid, lower, upper, **band_kwargs)

    if xscale is not None:
        ax.set_xscale(xscale)
    _set_axis_labels(data, ax)
    return ax


def plot_asymptotes(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    parameters: Sequence[AsymptoteName] = ("ymin", "ymax"),
    label: bool = True,
    **line_kwargs,
) -> Axes:
    """Plot model asymptotes as horizontal lines on an existing axes."""
    ax = _get_axes(ax)
    resolved_compound_id = _resolve_compound_id(data, compound_id)
    fits = _matching_fits(
        results,
        compound_id=resolved_compound_id,
        experiments=experiments,
    )

    default_kwargs = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7}
    default_kwargs.update(line_kwargs)

    n_drawn = 0
    for fit in fits:
        for parameter in parameters:
            if parameter not in fit.parameters:
                continue
            asymptote_label = None
            if label:
                experiment = fit.experiment_id or fit.model_name
                asymptote_label = f"{experiment} {parameter}"
            ax.axhline(
                fit.parameters[parameter].value,
                label=asymptote_label,
                **default_kwargs,
            )
            n_drawn += 1

    if n_drawn == 0:
        raise ValueError("No requested asymptote parameters were available to plot.")

    _set_axis_labels(data, ax)
    return ax


def plot_curve_points(
    data: DoseResponseData,
    results: FitResults,
    *,
    points: Iterable[CurvePointSpec],
    compound_id: str | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    annotate: bool = True,
    annotation_offset: tuple[float, float] = (6.0, 6.0),
    point_kwargs: dict | None = None,
    annotation_kwargs: dict | None = None,
) -> Axes:
    """Plot arbitrary labeled points evaluated on fitted curves."""
    ax = _get_axes(ax)
    resolved_compound_id = _resolve_compound_id(data, compound_id)
    fits = _matching_fits(
        results,
        compound_id=resolved_compound_id,
        experiments=experiments,
    )
    curve_points = _coerce_curve_points(points)

    default_point_kwargs = {"marker": "o", "zorder": 5}
    default_point_kwargs.update(point_kwargs or {})
    default_annotation_kwargs = {"textcoords": "offset points"}
    default_annotation_kwargs.update(annotation_kwargs or {})

    for fit in fits:
        for point in curve_points:
            y_value = float(np.asarray(_evaluate_fit(fit, point.x)))
            ax.scatter([point.x], [y_value], **default_point_kwargs)

            if not annotate or point.label is None:
                continue

            label = point.label
            if len(fits) > 1:
                experiment = fit.experiment_id or fit.model_name
                label = f"{label} ({experiment})"

            ax.annotate(
                label,
                xy=(point.x, y_value),
                xytext=annotation_offset,
                **default_annotation_kwargs,
            )

    _set_axis_labels(data, ax)
    return ax


def plot_curves(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    aggregate: bool = True,
    by_experiment: bool = True,
    error: ErrorStyle = "sem",
    x_grid: np.ndarray | None = None,
    n_points: int = 300,
    xscale: XScale = "log",
    confidence_band: bool = False,
    confidence_level: float = 0.95,
    show_asymptotes: bool = False,
    curve_points: Iterable[CurvePointSpec] | None = None,
    observation_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
    confidence_band_kwargs: dict | None = None,
    asymptote_kwargs: dict | None = None,
    curve_point_kwargs: dict | None = None,
) -> Axes:
    """Plot observations, fitted curves, and optional model annotations."""
    ax = _get_axes(ax)
    resolved_compound_id = _resolve_compound_id(data, compound_id)

    plot_observations(
        data,
        compound_id=resolved_compound_id,
        ax=ax,
        aggregate=aggregate,
        by_experiment=by_experiment,
        error=error,
        experiments=experiments,
        **(observation_kwargs or {}),
    )
    if confidence_band:
        plot_confidence_bands(
            data,
            results,
            compound_id=resolved_compound_id,
            ax=ax,
            experiments=experiments,
            x_grid=x_grid,
            n_points=n_points,
            xscale=xscale,
            confidence_level=confidence_level,
            **(confidence_band_kwargs or {}),
        )
    plot_fits(
        data,
        results,
        compound_id=resolved_compound_id,
        ax=ax,
        experiments=experiments,
        x_grid=x_grid,
        n_points=n_points,
        xscale=xscale,
        **(fit_kwargs or {}),
    )
    if show_asymptotes:
        plot_asymptotes(
            data,
            results,
            compound_id=resolved_compound_id,
            ax=ax,
            experiments=experiments,
            **(asymptote_kwargs or {}),
        )
    if curve_points is not None:
        plot_curve_points(
            data,
            results,
            points=curve_points,
            compound_id=resolved_compound_id,
            ax=ax,
            experiments=experiments,
            **(curve_point_kwargs or {}),
        )
    return ax
