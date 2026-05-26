from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import lmfit
import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ConcentrationParameterSpec,
    ParameterSpec,
)


@dataclass(frozen=True)
class ModelEvaluation:
    """Observable response plus any model-specific component arrays."""

    concentration: np.ndarray
    transformed_x: np.ndarray
    response: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)

    def component(self, name: str) -> np.ndarray:
        """Return one named component array."""
        return self.components[name]


class BaseDoseResponseModel(ABC):
    """Base class for dose-response models fitted through lmfit."""

    name: str
    parameter_specs: tuple[ParameterSpec, ...]
    concentration_parameters: frozenset[str] = frozenset()
    concentration_parameter_specs: tuple[ConcentrationParameterSpec, ...] = ()
    required_fixed_parameters: frozenset[str] = frozenset()

    @abstractmethod
    def evaluate(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the model at transformed x values."""

    @abstractmethod
    def guess(self, compound: CompoundData) -> dict[str, float]:
        """Generate initial parameter guesses for one compound or experiment."""

    def transform_x(self, concentration: np.ndarray) -> np.ndarray:
        """Transform positive concentrations into the model's independent variable."""
        return np.asarray(concentration, dtype=float)

    def predict(self, concentration: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the observable response on the raw concentration axis."""
        concentration = np.asarray(concentration, dtype=float)
        x = self.transform_x(concentration)
        return np.asarray(self.evaluate(x, **params), dtype=float)

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        """Return model-specific component arrays on the raw concentration axis."""
        return {}

    def evaluate_components(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> ModelEvaluation:
        """Evaluate the observable response and any model-specific components."""
        concentration = np.asarray(concentration, dtype=float)
        x = self.transform_x(concentration)
        response = np.asarray(self.evaluate(x, **params), dtype=float)
        components = {
            name: np.asarray(values, dtype=float)
            for name, values in self.component_arrays(
                concentration,
                x,
                **params,
            ).items()
        }
        return ModelEvaluation(
            concentration=concentration,
            transformed_x=x,
            response=response,
            components=components,
        )

    def make_lmfit_parameters(
        self,
        guesses: Mapping[str, float],
        *,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> lmfit.Parameters:
        """Create lmfit Parameters from model specs, guesses, and user overrides."""
        fixed = fixed or {}
        bounds = bounds or {}
        parameters = lmfit.Parameters()

        known_names = {spec.name for spec in self.parameter_specs}
        unknown_fixed = set(fixed) - known_names
        if unknown_fixed:
            raise KeyError(f"Unknown fixed parameter(s): {sorted(unknown_fixed)}")

        unknown_bounds = set(bounds) - known_names
        if unknown_bounds:
            raise KeyError(f"Unknown bounded parameter(s): {sorted(unknown_bounds)}")

        missing_required = self.required_fixed_parameters - set(fixed)
        if missing_required:
            raise ValueError(
                "Missing required fixed parameter(s) for "
                f"{self.name!r}: {sorted(missing_required)}"
            )

        for spec in self.parameter_specs:
            value = fixed.get(spec.name, guesses.get(spec.name, spec.initial))
            if value is None:
                raise ValueError(f"No initial value available for {spec.name!r}.")

            lower, upper = bounds.get(spec.name, (spec.min, spec.max))
            if lower is None:
                lower = spec.min
            if upper is None:
                upper = spec.max

            value = float(value)
            lower = float(lower)
            upper = float(upper)

            if spec.min >= STRICTLY_POSITIVE_PARAMETER_MIN:
                if lower <= 0.0:
                    raise ValueError(
                        f"Lower bound for {spec.name!r} must be strictly positive."
                    )
                if value <= 0.0:
                    raise ValueError(
                        f"Initial or fixed value for {spec.name!r} must be "
                        "strictly positive."
                    )
                if upper <= 0.0:
                    raise ValueError(
                        f"Upper bound for {spec.name!r} must be strictly positive."
                    )

            parameters.add(
                spec.name,
                value=value,
                min=lower,
                max=upper,
                vary=spec.name not in fixed and spec.vary,
            )

        return parameters

    def residual(
        self,
        parameters: lmfit.Parameters,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return residuals in lmfit convention."""
        values = {name: parameter.value for name, parameter in parameters.items()}
        residual = self.evaluate(x, **values) - y
        if weights is not None:
            residual = residual * weights
        return residual
