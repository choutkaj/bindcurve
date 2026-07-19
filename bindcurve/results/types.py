from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np

from bindcurve.modeling.base import BaseDoseResponseModel

ReportUncertainty = Literal["sd", "sem", "ci95"]


@dataclass(frozen=True)
class ParameterEstimate:
    """Estimate for one fitted parameter in public physical coordinates."""

    name: str
    value: float
    stderr: float | None = None
    vary: bool = True
    min: float = -np.inf
    max: float = np.inf

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Parameter estimate names must not be blank.")
        if not np.isfinite(self.value):
            raise ValueError(f"Estimate for {self.name!r} must be finite.")
        if self.stderr is not None and (
            not np.isfinite(self.stderr) or self.stderr < 0.0
        ):
            raise ValueError(
                f"Standard error for {self.name!r} must be finite and non-negative."
            )
        if np.isnan(self.min) or np.isnan(self.max) or self.min > self.max:
            raise ValueError(f"Bounds for {self.name!r} are invalid.")
        if not self.min <= self.value <= self.max:
            raise ValueError(f"Estimate for {self.name!r} lies outside its bounds.")


@dataclass(frozen=True)
class FitMetrics:
    """Numerical diagnostics for one fit."""

    n_data: int
    n_varying_parameters: int
    rss: float
    reduced_rss: float | None
    chi_square: float | None = None
    reduced_chi_square: float | None = None
    aic: float | None = None
    bic: float | None = None
    r_squared: float | None = None

    def __post_init__(self) -> None:
        if self.n_data < 0 or self.n_varying_parameters < 0:
            raise ValueError(
                "Observation and varying-parameter counts must be non-negative."
            )
        for name in ("rss", "reduced_rss", "chi_square", "reduced_chi_square"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"{name} must be finite and non-negative when present."
                )
        if self.r_squared is not None and not np.isfinite(self.r_squared):
            raise ValueError("r_squared must be finite when present.")


@dataclass(frozen=True)
class FitResult:
    """Immutable result for one fitted curve."""

    model: BaseDoseResponseModel
    compound_id: str
    experiment_id: str | None = None
    success: bool = True
    parameters: Mapping[str, ParameterEstimate] = field(default_factory=dict)
    metrics: FitMetrics | None = None
    covariance: np.ndarray | None = None
    variable_names: tuple[str, ...] = ()
    optimizer_message: str | None = None
    failure_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not str(self.compound_id).strip():
            raise ValueError("compound_id must not be blank.")
        parameters = dict(self.parameters)
        for name, estimate in parameters.items():
            if name != estimate.name:
                raise ValueError(
                    f"Parameter mapping key {name!r} does not match estimate name "
                    f"{estimate.name!r}."
                )
        variable_names = tuple(self.variable_names)
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("variable_names must be unique.")
        unknown_variables = set(variable_names) - set(parameters)
        if unknown_variables:
            raise ValueError(
                f"Covariance variable names lack estimates: {sorted(unknown_variables)}"
            )
        nonvarying_variables = [
            name for name in variable_names if not parameters[name].vary
        ]
        if nonvarying_variables:
            raise ValueError(
                "Covariance variable names must refer to varying parameters: "
                f"{nonvarying_variables}"
            )

        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "variable_names", variable_names)
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float).copy()
            expected_shape = (len(variable_names), len(variable_names))
            if covariance.shape != expected_shape:
                raise ValueError(
                    "Covariance shape must match variable_names: "
                    f"{covariance.shape} != {expected_shape}."
                )
            if np.any(~np.isfinite(covariance)):
                raise ValueError("Covariance must contain only finite values.")
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                raise ValueError("Covariance must be symmetric.")
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance", covariance)

    @property
    def model_name(self) -> str:
        """Return the name of the exact model instance used for this fit."""
        return self.model.name

    @classmethod
    def failed(
        cls,
        *,
        model: BaseDoseResponseModel,
        compound_id: str,
        experiment_id: str | None,
        stage: str,
        error: Exception,
    ) -> FitResult:
        """Create a failed result while preserving diagnostic context."""
        return cls(
            model=model,
            compound_id=compound_id,
            experiment_id=experiment_id,
            success=False,
            failure_stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def parameter(self, name: str) -> ParameterEstimate:
        """Return a fitted parameter by name."""
        return self.parameters[name]


@dataclass(frozen=True)
class ParameterSummary:
    """Summary of one native additive parameter across independent fits."""

    compound_id: str
    parameter: str
    N_exp: int
    mean: float
    sd: float | None
    sem: float | None
    ci95_lower: float | None
    ci95_upper: float | None


@dataclass(frozen=True)
class ConcentrationSummary:
    """Summary of one positive concentration-like quantity across fits."""

    compound_id: str
    parameter: str
    log_parameter: str
    N_exp: int
    reportable: bool
    log10_mean: float
    log10_sd: float | None
    log10_sem: float | None
    log10_ci95_lower: float | None
    log10_ci95_upper: float | None

    @property
    def center(self) -> float:
        """Return the linear-scale center derived from the log10 mean."""
        return float(10**self.log10_mean)

    def linear_interval(
        self,
        uncertainty: ReportUncertainty,
    ) -> tuple[float | None, float | None]:
        """Return a derived linear-scale interval."""
        if uncertainty == "sd":
            return _back_transform_log_interval(self.log10_mean, self.log10_sd)
        if uncertainty == "sem":
            return _back_transform_log_interval(self.log10_mean, self.log10_sem)
        if uncertainty == "ci95":
            if self.log10_ci95_lower is None or self.log10_ci95_upper is None:
                return (None, None)
            return (
                float(10**self.log10_ci95_lower),
                float(10**self.log10_ci95_upper),
            )
        raise ValueError("uncertainty must be 'sd', 'sem', or 'ci95'.")

    def log_interval(
        self,
        uncertainty: ReportUncertainty,
    ) -> tuple[float | None, float | None]:
        """Return a log10-scale interval or additive spread."""
        if uncertainty == "sd":
            return _symmetric_interval(self.log10_mean, self.log10_sd)
        if uncertainty == "sem":
            return _symmetric_interval(self.log10_mean, self.log10_sem)
        if uncertainty == "ci95":
            return (self.log10_ci95_lower, self.log10_ci95_upper)
        raise ValueError("uncertainty must be 'sd', 'sem', or 'ci95'.")


SummaryRecord: TypeAlias = ParameterSummary | ConcentrationSummary


def _symmetric_interval(
    center: float,
    spread: float | None,
) -> tuple[float | None, float | None]:
    if spread is None:
        return (None, None)
    return (center - spread, center + spread)


def _back_transform_log_interval(
    center: float,
    spread: float | None,
) -> tuple[float | None, float | None]:
    lower, upper = _symmetric_interval(center, spread)
    if lower is None or upper is None:
        return (None, None)
    return (float(10**lower), float(10**upper))
