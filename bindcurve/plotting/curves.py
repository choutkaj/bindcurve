from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from scipy.stats import t as student_t

from bindcurve.datasets import DoseResponseData
from bindcurve.fitting import FitCalculator, FitSettings
from bindcurve.modeling import get_model
from bindcurve.results import FitResult, FitResults

ErrorStyle = Literal["sem", "sd", None]
XScale = Literal["log", "linear", None]
AsymptoteName = Literal["ymin", "ymax"]
CurvePointSpec = float | tuple[float, str] | dict[str, object]
DoseRepresentation = Literal["mean", "experiments"]


@dataclass(frozen=True)
class CurvePoint:
    """Point to annotate on a fitted curve."""

    x: float
    label: str | None = None


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
    grouped = table.groupby(["experiment_id", "concentration"], as_index=False)["response"]
    aggregated = grouped.agg(response="mean", response_sd="std", n_experiment_replicates="count")
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


def _set_axis_labels(data: DoseResponseData, ax: Axes) -> None:
    ax.set_xlabel("concentration")
    ax.set_ylabel("response")


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
        if requested_experiments is not None and str(fit.experiment_id) not in requested_experiments:
            continue
        fits.append(fit)
    return fits


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
    filtered_table = data.table[data.table["compound_id"].astype(str).isin(compound_ids)]
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
        raise TypeError("colors must be a single Matplotlib color or a sequence of colors.") from exc

    if len(resolved) != n_series:
        raise ValueError(
            f"colors must contain exactly {n_series} entries for the plotted series; "
            f"got {len(resolved)}."
        )
    if not all(is_color_like(color) for color in resolved):
        raise ValueError("Every entry in colors must be a valid Matplotlib color.")
    return resolved


def _series_label_for_fit(
    fit: FitResult,
    *,
    n_compounds: int,
) -> str:
    experiment = fit.experiment_id or fit.model_name
    if n_compounds > 1:
        return f"{fit.compound_id} {experiment}"
    return str(experiment)


def _observation_table_for_fit(
    data: DoseResponseData,
    fit: FitResult,
) -> pd.DataFrame:
    fit_table = data.table[data.table["compound_id"].astype(str) == str(fit.compound_id)]
    if fit.experiment_id is not None:
        fit_table = fit_table[fit_table["experiment_id"].astype(str) == str(fit.experiment_id)]
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
    compound_table = data.table[data.table["compound_id"].astype(str) == str(compound_id)]
    if compound_table.empty:
        return []

    experiment_means = _aggregate_within_experiment(compound_table)
    if dose_representation == "mean":
        return [_aggregate_grand_mean(experiment_means)]

    return [
        group.reset_index(drop=True)
        for _, group in experiment_means.groupby("experiment_id", sort=True)
    ]


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


def _plot_series_confidence_band(
    ax: Axes,
    fit: FitResult,
    grid: np.ndarray,
    *,
    color: object,
    confidence_level: float,
    confidence_band_kwargs: dict | None,
) -> None:
    _, lower, upper = _fit_confidence_band(
        fit,
        grid,
        confidence_level=confidence_level,
        finite_difference_step=1.0e-2,
    )

    fill_kwargs: dict[str, object] = {
        "alpha": 0.25,
        "linewidth": 0.8,
        "color": color,
        "label": "_nolegend_",
    }
    if confidence_band_kwargs:
        fill_kwargs.update(
            {
                key: value
                for key, value in confidence_band_kwargs.items()
                if key != "label"
            }
        )
    ax.fill_between(grid, lower, upper, **fill_kwargs)


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
        # Keep markers in the legend handle while leaving the fitted line itself marker-free.
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


def _confidence_multiplier(confidence_level: float, *, degrees_of_freedom: int) -> float:
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if degrees_of_freedom <= 0:
        raise ValueError("Confidence bands require positive residual degrees of freedom.")
    return float(student_t.ppf(0.5 + confidence_level / 2.0, df=degrees_of_freedom))


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
    lmfit_result = fit.lmfit_result
    if lmfit_result is None:
        raise ValueError("Cannot plot confidence bands without lmfit_result.")

    degrees_of_freedom = int(lmfit_result.ndata) - int(lmfit_result.nvarys)
    multiplier = _confidence_multiplier(
        confidence_level,
        degrees_of_freedom=degrees_of_freedom,
    )
    x = np.asarray(x, dtype=float)
    parameters = {name: estimate.value for name, estimate in fit.parameters.items()}
    y = np.asarray(_evaluate_model(fit, x, parameters), dtype=float)

    jacobian = np.empty((x.size, len(variable_names)), dtype=float)
    for index, name in enumerate(variable_names):
        if name not in parameters:
            raise ValueError(f"Fit result is missing varying parameter {name!r}.")
        value = parameters[name]
        stderr = fit.parameters[name].stderr
        if stderr is not None and np.isfinite(stderr) and stderr > 0.0:
            step = finite_difference_step * stderr
        else:
            step = finite_difference_step * max(1.0, abs(value))
        if step == 0.0:
            step = finite_difference_step
        plus_parameters = dict(parameters)
        minus_parameters = dict(parameters)
        plus_parameters[name] = value + step
        minus_parameters[name] = value - step
        y_plus = np.asarray(_evaluate_model(fit, x, plus_parameters), dtype=float)
        y_minus = np.asarray(_evaluate_model(fit, x, minus_parameters), dtype=float)
        jacobian[:, index] = (y_plus - y_minus) / (2.0 * step)

    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    band = multiplier * np.sqrt(np.maximum(variance, 0.0))
    return y, y - band, y + band


def _residual_table_for_fit(
    table: pd.DataFrame,
    fit: FitResult,
    *,
    aggregate: bool,
) -> pd.DataFrame:
    fit_table = table[table["compound_id"].astype(str) == str(fit.compound_id)]
    if fit.experiment_id is not None:
        fit_table = fit_table[fit_table["experiment_id"].astype(str) == str(fit.experiment_id)]

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
    finite_difference_step: float = 1.0e-2,
    label: str | None = None,
    **fill_between_kwargs,
) -> Axes:
    """Plot covariance-based pointwise confidence bands for fitted mean curves.

    These bands reflect uncertainty in the fitted mean response due to
    parameter uncertainty in each experiment-level fit. They are not
    prediction intervals for future observations.
    """
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)
    
    # Grid logic duplicated or should be shared
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

    default_kwargs = {"alpha": 0.25, "linewidth": 0.8}
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
                if len(resolved_compound_ids) > 1:
                    band_label = f"{fit.compound_id} {experiment} {percent:g}% confidence band"
                else:
                    band_label = f"{experiment} {percent:g}% confidence band"
            band_kwargs["label"] = band_label

        ax.fill_between(grid, lower, upper, **band_kwargs)

    if xscale is not None:
        ax.set_xscale(xscale)
    _set_axis_labels(data, ax)
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
    ax.set_xlabel("concentration")
    ax.set_ylabel("residual")
    return ax


def plot_asymptotes(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    parameters: Sequence[AsymptoteName] = ("ymin", "ymax"),
    label: bool = True,
    **line_kwargs,
) -> Axes:
    """Plot model asymptotes as horizontal lines on an existing axes."""
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)
    fits = _matching_fits(
        results,
        compound_ids=resolved_compound_ids,
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
                if len(resolved_compound_ids) > 1:
                    asymptote_label = f"{fit.compound_id} {experiment} {parameter}"
                else:
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
    compound_id: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    annotate: bool = True,
    annotation_offset: tuple[float, float] = (6.0, 6.0),
    point_kwargs: dict | None = None,
    annotation_kwargs: dict | None = None,
) -> Axes:
    """Plot arbitrary labeled points evaluated on fitted curves."""
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)
    fits = _matching_fits(
        results,
        compound_ids=resolved_compound_ids,
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
            if len(fits) > 1 or len(resolved_compound_ids) > 1:
                experiment = fit.experiment_id or fit.model_name
                if len(resolved_compound_ids) > 1:
                    label = f"{label} ({fit.compound_id} {experiment})"
                else:
                    label = f"{label} ({experiment})"

            ax.annotate(
                label,
                xy=(point.x, y_value),
                xytext=annotation_offset,
                **default_annotation_kwargs,
            )

    _set_axis_labels(data, ax)
    return ax


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
    _set_axis_labels(data, ax)
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
    _set_axis_labels(data, ax)
    return ax


