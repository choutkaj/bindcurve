from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import t as student_t

from bindcurve.datasets import DoseResponseData
from bindcurve.plotting.common import (
    XScale,
    _evaluate_model,
    _get_axes,
    _matching_fits,
    _resolve_compound_ids,
)
from bindcurve.results import FitResult, FitResults


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


def _confidence_multiplier(
    confidence_level: float, *, degrees_of_freedom: int
) -> float:
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if degrees_of_freedom <= 0:
        raise ValueError(
            "Confidence bands require positive residual degrees of freedom."
        )
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
                    band_label = (
                        f"{fit.compound_id} {experiment} "
                        f"{percent:g}% confidence band"
                    )
                else:
                    band_label = f"{experiment} {percent:g}% confidence band"
            band_kwargs["label"] = band_label

        ax.fill_between(grid, lower, upper, **band_kwargs)

    if xscale is not None:
        ax.set_xscale(xscale)
    return ax
