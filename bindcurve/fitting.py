"""Experiment-level least-squares fitting."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from bindcurve.data import DoseResponseData, replicate_means
from bindcurve.models import Model, Parameter, get_model
from bindcurve.models.base import PLATEAUS
from bindcurve.results import FitResult, FitResults

# Concentrations are searched as log10 values within +-100 decades: far beyond
# any physical concentration, while runaway fits stay finite.
LOG10_LIMIT = 100.0
PLATEAU_NAMES = frozenset(p.name for p in PLATEAUS)

Bounds = Mapping[str, tuple[float | None, float | None]]


def fit(
    data: DoseResponseData,
    model: str | Model,
    *,
    fixed: Mapping[str, float] | None = None,
    bounds: Bounds | None = None,
    errors: Literal["raise", "collect"] = "raise",
) -> FitResults:
    """Fit a model separately to every experiment of every compound.

    Technical replicates are averaged per concentration, each mean is weighted
    by its replicate count, and standard errors are scaled by the residual
    scatter.

    Parameters
    ----------
    data
        Observations to fit.
    model
        A built-in model name such as ``"ic50"`` (see `get_model`) or a model.
    fixed
        Values held fixed in every fit. Assay constants of binding models,
        such as ``RT``, ``LsT`` and ``Kds``, must be given here.
    bounds
        ``{name: (lower, upper)}`` limits of fitted parameters; ``None``
        leaves a side unbounded.
    errors
        Re-raise errors, or record them as failed fits and continue.

    Returns
    -------
    FitResults

    Warns
    -----
    UserWarning
        If converged fits carry quality warnings, see `FitResult.warnings`.
    """
    model = get_model(model) if isinstance(model, str) else model
    fixed = {name: float(value) for name, value in (fixed or {}).items()}
    bounds = dict(bounds or {})
    if errors not in ("raise", "collect"):
        raise ValueError("errors must be 'raise' or 'collect'.")
    _check_settings(model, fixed, bounds)

    fits = []
    experiments = data.table.groupby(["compound_id", "experiment_id"], sort=True)
    for (compound_id, experiment_id), table in experiments:
        identity = {"compound_id": compound_id, "experiment_id": experiment_id}
        try:
            fits.append(_fit_experiment(model, table, fixed, bounds, identity))
        except Exception as error:
            if errors == "raise":
                raise
            message = f"{type(error).__name__}: {error}"
            fits.append(
                FitResult(**identity, model=model, success=False, message=message)
            )

    n_flagged = sum(bool(fit.warnings) for fit in fits)
    if n_flagged:
        warnings.warn(
            f"{n_flagged} of {len(fits)} fits have quality warnings; see "
            "FitResults.experiments().",
            UserWarning,
            stacklevel=2,
        )
    return FitResults(model=model, data=data, fixed=fixed, fits=tuple(fits))


def _fit_experiment(
    model: Model,
    table: pd.DataFrame,
    fixed: dict[str, float],
    bounds: Bounds,
    identity: dict[str, str],
) -> FitResult:
    means = replicate_means(table, ["concentration"])
    x = means["concentration"].to_numpy()
    y = means["response"].to_numpy()
    # A mean of n replicates has variance sigma**2 / n. Only these relative
    # weights are known; they are 1 when replicate counts are equal.
    n = means["n"].to_numpy(dtype=float)
    relative_sd = np.sqrt(n.mean() / n)

    free = [p for p in model.parameters if p.name not in fixed]
    if len(x) <= len(free):
        raise ValueError(
            f"Fitting {len(free)} parameters needs more than {len(x)} concentrations."
        )

    # Responses and plateaus are measured from the lowest mean in units of the
    # response range. The solver's absolute tolerances then hold at any
    # response scale, and predictions avoid cancellation against a baseline.
    offset = float(np.min(y))
    coordinates = _Coordinates(offset, float(np.ptp(y)) or abs(offset) or 1.0)
    y_scaled = (y - offset) / coordinates.scale
    fixed_plateaus = {
        name: coordinates.coordinate(model.parameter(name), value)
        for name, value in fixed.items()
        if name in PLATEAU_NAMES
    }

    def values_at(theta: np.ndarray) -> dict[str, float]:
        values = dict(fixed)
        for p, t in zip(free, theta, strict=True):
            values[p.name] = coordinates.value(p, t)
        return values

    def residual(theta: np.ndarray) -> np.ndarray:
        # Model.evaluate is ymin + (ymax - ymin) * fraction; evaluated here with
        # the plateau coordinates. Parameters are validated at their values.
        plateaus = dict(fixed_plateaus)
        for p, t in zip(free, theta, strict=True):
            if p.name in PLATEAU_NAMES:
                plateaus[p.name] = t
        x_checked, values = model._checked(x, values_at(theta))
        fraction = model._fraction(x_checked, **values)
        low, high = plateaus["ymin"], plateaus["ymax"]
        return (y_scaled - low - (high - low) * fraction) / relative_sd

    covariance = None
    if free:
        guess = model.guess(x, y)
        missing = [p.name for p in free if p.name not in guess]
        if missing:
            raise ValueError(f"No initial values for {missing}; fix them instead.")
        start = [coordinates.coordinate(p, guess[p.name]) for p in free]
        lower, upper = np.array(
            [_search_range(p, bounds.get(p.name), coordinates) for p in free]
        ).T
        solution = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=2000 * (len(free) + 1),
        )
        theta, success, message = solution.x, bool(solution.success), solution.message
        covariance = _covariance(solution.jac)
    else:
        theta, success, message = np.array([]), True, "All parameters were fixed."

    values = values_at(theta)
    raw = y - model.evaluate(x, **values)
    if covariance is not None:
        covariance *= np.sum(residual(theta) ** 2) / (len(x) - len(free))
        scale = [coordinates.derivative(p, values[p.name]) for p in free]
        covariance *= np.outer(scale, scale)
    stderr = (
        {}
        if covariance is None
        else {p.name: float(np.sqrt(covariance[i, i])) for i, p in enumerate(free)}
    )
    rss = float(np.sum(raw**2))
    total = float(np.sum((y - y.mean()) ** 2))
    result = FitResult(
        **identity,
        model=model,
        success=success,
        message=message,
        values=values,
        stderr=stderr,
        free=tuple(p.name for p in free),
        covariance=covariance,
        n_data=len(x),
        rss=rss,
        r_squared=1.0 - rss / total if total > 0 else None,
    )
    return replace(result, warnings=_quality_warnings(result, x)) if success else result


def _check_settings(model: Model, fixed: dict[str, float], bounds: Bounds) -> None:
    names = {p.name for p in model.parameters}
    unknown = sorted((set(fixed) | set(bounds)) - names)
    if unknown:
        raise KeyError(f"Model {model.name!r} has no parameter(s) {unknown}.")
    missing = [p.name for p in model.parameters if p.fixed and p.name not in fixed]
    if missing:
        raise ValueError(f"Model {model.name!r} requires fixed values for {missing}.")
    for name, value in fixed.items():
        p = model.parameter(name)
        if not np.isfinite(value) or not p.min <= value <= p.max:
            raise ValueError(f"Fixed {name} = {value} is outside [{p.min}, {p.max}].")
        if p.concentration and value <= 0.0:
            raise ValueError(f"Fixed {name} must be positive.")
    for name, (lower, upper) in bounds.items():
        if name in fixed:
            raise ValueError(f"{name} cannot be both fixed and bounded.")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                f"The lower bound of {name} must be below its upper bound."
            )
        if model.parameter(name).concentration and (
            (lower is not None and lower < 0.0) or (upper is not None and upper <= 0.0)
        ):
            raise ValueError(f"Bounds of {name} must be positive.")


@dataclass(frozen=True)
class _Coordinates:
    """Optimizer coordinates of order one for parameter values.

    Concentrations are log10 values; plateaus are measured from ``offset`` in
    units of ``scale``, the response range. Other parameters are unchanged.
    """

    offset: float
    scale: float

    def coordinate(self, p: Parameter, value: float) -> float:
        if p.concentration:
            return float(np.log10(value))
        if p.name in PLATEAU_NAMES:
            return (value - self.offset) / self.scale
        return value

    def value(self, p: Parameter, coordinate: float) -> float:
        if p.concentration:
            return float(10.0**coordinate)
        if p.name in PLATEAU_NAMES:
            return float(self.offset + self.scale * coordinate)
        return float(coordinate)

    def derivative(self, p: Parameter, value: float) -> float:
        """``d(value) / d(coordinate)`` at ``value``."""
        if p.concentration:
            return np.log(10.0) * value
        if p.name in PLATEAU_NAMES:
            return self.scale
        return 1.0


def _search_range(
    p: Parameter,
    bound: tuple[float | None, float | None] | None,
    coordinates: _Coordinates,
) -> tuple[float, float]:
    """Optimizer-coordinate limits from physical limits and user bounds."""
    lower, upper = bound or (None, None)
    lower = p.min if lower is None else max(lower, p.min)
    upper = p.max if upper is None else min(upper, p.max)
    if not p.concentration:
        return coordinates.coordinate(p, lower), coordinates.coordinate(p, upper)
    lower = np.log10(lower) if lower > 0.0 else -LOG10_LIMIT
    upper = np.log10(upper) if np.isfinite(upper) else LOG10_LIMIT
    return max(lower, -LOG10_LIMIT), min(upper, LOG10_LIMIT)


def _covariance(jacobian: np.ndarray) -> np.ndarray | None:
    """Return ``inv(J.T @ J)`` via SVD, or None if ``J`` is rank deficient."""
    _, s, vt = np.linalg.svd(jacobian, full_matrices=False)
    if s[-1] <= np.finfo(float).eps * max(jacobian.shape) * s[0]:
        return None
    return (vt.T / s**2) @ vt


def _quality_warnings(fit: FitResult, x: np.ndarray) -> tuple[str, ...]:
    """Flag converged fits whose estimates the data may not support."""
    messages = []
    if fit.free and fit.covariance is None:
        messages.append("Standard errors could not be estimated.")
    for name in fit.free:
        if not fit.model.parameter(name).concentration:
            continue
        value = fit.values[name]
        if not x.min() <= value <= x.max():
            messages.append(f"{name} lies outside the tested concentration range.")
        if fit.stderr.get(name, 0.0) > value:
            messages.append(
                f"{name} is poorly determined: its standard error exceeds the estimate."
            )
    return tuple(messages)
