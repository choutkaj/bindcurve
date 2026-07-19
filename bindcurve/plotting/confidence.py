from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import t as student_t

from bindcurve.plotting.common import (
    _evaluate_model,
)
from bindcurve.results import FitResult


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


def _get_covariance(fit: FitResult) -> tuple[list[str], np.ndarray]:
    if fit.covariance is None:
        raise ValueError("Cannot plot confidence bands without a covariance matrix.")
    variable_names = list(fit.variable_names)
    if not variable_names:
        raise ValueError("Cannot plot confidence bands without variable names.")

    covariance_array = np.asarray(fit.covariance, dtype=float)
    expected_shape = (len(variable_names), len(variable_names))
    if covariance_array.shape != expected_shape:
        raise ValueError(
            "Covariance matrix shape does not match variable names: "
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
    variable_names, covariance = _get_covariance(fit)
    finite_difference_step = float(finite_difference_step)
    if not np.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive.")
    if fit.metrics is None:
        raise ValueError("Cannot plot confidence bands without fit metrics.")
    degrees_of_freedom = (
        fit.metrics.n_data - fit.metrics.n_varying_parameters
    )
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
        estimate = fit.parameters[name]
        distance_up = estimate.max - value
        distance_down = value - estimate.min
        symmetric_step = min(step, distance_up, distance_down)
        if symmetric_step > 0.0:
            plus_parameters = dict(parameters)
            minus_parameters = dict(parameters)
            plus_parameters[name] = value + symmetric_step
            minus_parameters[name] = value - symmetric_step
            y_plus = np.asarray(
                _evaluate_model(fit, x, plus_parameters),
                dtype=float,
            )
            y_minus = np.asarray(
                _evaluate_model(fit, x, minus_parameters),
                dtype=float,
            )
            jacobian[:, index] = (
                y_plus - y_minus
            ) / (2.0 * symmetric_step)
            continue

        if distance_up > 0.0:
            forward_step = min(step, distance_up)
            plus_parameters = dict(parameters)
            plus_parameters[name] = value + forward_step
            y_plus = np.asarray(
                _evaluate_model(fit, x, plus_parameters),
                dtype=float,
            )
            jacobian[:, index] = (y_plus - y) / forward_step
            continue

        if distance_down > 0.0:
            backward_step = min(step, distance_down)
            minus_parameters = dict(parameters)
            minus_parameters[name] = value - backward_step
            y_minus = np.asarray(
                _evaluate_model(fit, x, minus_parameters),
                dtype=float,
            )
            jacobian[:, index] = (y - y_minus) / backward_step
            continue

        raise ValueError(f"Varying parameter {name!r} has no feasible perturbation.")

    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    band = multiplier * np.sqrt(np.maximum(variance, 0.0))
    return y, y - band, y + band
