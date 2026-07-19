from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import lmfit
import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.parameters import ParameterSpec


@dataclass(frozen=True)
class ModelEvaluation:
    """Observable response plus any model-specific component arrays."""

    concentration: np.ndarray
    response: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)

    def component(self, name: str) -> np.ndarray:
        """Return one named component array."""
        return self.components[name]


class BaseDoseResponseModel(ABC):
    """Base class for dose-response models fitted through lmfit."""

    name: str
    parameter_specs: tuple[ParameterSpec, ...]

    def __init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model names must be non-empty strings.")
        parameter_names = [spec.name for spec in self.parameter_specs]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError(
                f"Model {self.name!r} contains duplicate parameter names."
            )

    def evaluate(self, concentration: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the observable response on the raw concentration axis."""
        concentration = self._validated_concentration(concentration)
        params = self._validated_parameters(params)
        components = self._normalized_components(
            self._component_arrays(concentration, **params),
            expected_shape=concentration.shape,
        )
        return self._validated_response(
            self.response_from_components(components, **params),
            expected_shape=concentration.shape,
        )

    @abstractmethod
    def guess(self, compound: CompoundData) -> dict[str, float]:
        """Generate initial parameter guesses for one compound or experiment."""

    @abstractmethod
    def _component_arrays(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        """Return model-specific component arrays on the raw concentration axis."""

    def response_from_components(
        self,
        components: Mapping[str, np.ndarray],
        **params: float,
    ) -> np.ndarray:
        """Map a physical binding fraction to the measured response."""
        fraction = np.asarray(components["Fbs"], dtype=float)
        ymin = float(params["ymin"])
        ymax = float(params["ymax"])
        return ymin + (ymax - ymin) * fraction

    def evaluate_components(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> ModelEvaluation:
        """Evaluate the observable response and any model-specific components."""
        concentration = self._validated_concentration(concentration)
        params = self._validated_parameters(params)
        components = self._normalized_components(
            self._component_arrays(concentration, **params),
            expected_shape=concentration.shape,
        )
        response = self._validated_response(
            self.response_from_components(components, **params),
            expected_shape=concentration.shape,
        )
        return ModelEvaluation(
            concentration=concentration,
            response=response,
            components=components,
        )

    @property
    def concentration_parameter_specs(self) -> tuple[ParameterSpec, ...]:
        """Return concentration-like parameter specifications."""
        return tuple(
            spec for spec in self.parameter_specs if spec.kind == "concentration"
        )

    @property
    def required_fixed_parameters(self) -> frozenset[str]:
        """Return parameters that must be provided as fixed assay constants."""
        return frozenset(
            spec.name for spec in self.parameter_specs if spec.required_fixed
        )

    def parameter_spec(self, name: str) -> ParameterSpec:
        """Return one parameter specification by public name."""
        for spec in self.parameter_specs:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown parameter {name!r} for model {self.name!r}.")

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

            value = self._finite_parameter_value(spec.name, value)
            lower = self._parameter_bound(spec.name, lower)
            upper = self._parameter_bound(spec.name, upper)
            if spec.scale == "log10":
                if value <= 0.0:
                    raise ValueError(
                        f"Log-scaled parameter {spec.name!r} must be strictly positive."
                    )
                if lower <= 0.0:
                    raise ValueError(
                        f"Lower bound for log-scaled parameter {spec.name!r} "
                        "must be strictly positive."
                    )
                if upper <= 0.0:
                    raise ValueError(
                        f"Upper bound for log-scaled parameter {spec.name!r} "
                        "must be strictly positive."
                    )
            if lower >= upper:
                raise ValueError(
                    f"Lower bound for {spec.name!r} must be below its upper bound."
                )
            if not lower <= value <= upper:
                raise ValueError(
                    f"Initial or fixed value for {spec.name!r} must lie within bounds."
                )

            optimizer_value = value
            optimizer_lower = lower
            optimizer_upper = upper
            if spec.scale == "log10":
                optimizer_value = float(np.log10(value))
                optimizer_lower = float(np.log10(lower))
                optimizer_upper = float(np.log10(upper))

            parameters.add(
                spec.name,
                value=optimizer_value,
                min=optimizer_lower,
                max=optimizer_upper,
                vary=spec.name not in fixed and spec.vary,
            )

        return parameters

    def residual(
        self,
        parameters: lmfit.Parameters,
        concentration: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return observed-minus-predicted residuals, standardized when σ is known."""
        values = self.decode_parameters(parameters)
        residual = np.asarray(y, dtype=float) - self.evaluate(concentration, **values)
        if sigma is not None:
            residual = residual / np.asarray(sigma, dtype=float)
        return residual

    def decode_parameters(self, parameters: lmfit.Parameters) -> dict[str, float]:
        """Decode optimizer coordinates to public physical parameter values."""
        values: dict[str, float] = {}
        for spec in self.parameter_specs:
            value = float(parameters[spec.name].value)
            values[spec.name] = float(10**value) if spec.scale == "log10" else value
        return values

    def parameter_jacobian(
        self,
        parameters: lmfit.Parameters,
        variable_names: tuple[str, ...],
    ) -> np.ndarray:
        """Return d(public parameter)/d(optimizer coordinate) for varying parameters."""
        diagonal = []
        for name in variable_names:
            spec = self.parameter_spec(name)
            if spec.scale == "log10":
                value = float(10 ** parameters[name].value)
                diagonal.append(float(np.log(10.0) * value))
            else:
                diagonal.append(1.0)
        return np.diag(np.asarray(diagonal, dtype=float))

    @staticmethod
    def _validated_concentration(concentration: np.ndarray) -> np.ndarray:
        values = np.asarray(concentration, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Concentrations must be finite and non-negative.")
        return values

    @staticmethod
    def _normalized_components(
        components: Mapping[str, np.ndarray],
        *,
        expected_shape: tuple[int, ...],
    ) -> dict[str, np.ndarray]:
        normalized = {}
        for name, values in components.items():
            array = np.asarray(values, dtype=float)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Model component {name!r} has shape {array.shape}; "
                    f"expected {expected_shape}."
                )
            if np.any(~np.isfinite(array)):
                raise ValueError(f"Model component {name!r} is not finite.")
            normalized[name] = array
        return normalized

    @staticmethod
    def _validated_response(
        response: np.ndarray,
        *,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        values = np.asarray(response, dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"Model response has shape {values.shape}; expected {expected_shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("Model response must contain only finite values.")
        return values

    def _validated_parameters(
        self,
        params: Mapping[str, float],
    ) -> dict[str, float]:
        expected = {spec.name for spec in self.parameter_specs}
        provided = set(params)
        missing = expected - provided
        if missing:
            raise TypeError(f"Missing model parameter(s): {sorted(missing)}")
        unknown = provided - expected
        if unknown:
            raise TypeError(f"Unknown model parameter(s): {sorted(unknown)}")

        values: dict[str, float] = {}
        for spec in self.parameter_specs:
            value = self._finite_parameter_value(spec.name, params[spec.name])
            if not spec.min <= value <= spec.max:
                raise ValueError(
                    f"Parameter {spec.name!r} must lie within "
                    f"[{spec.min}, {spec.max}]."
                )
            values[spec.name] = value
        return values

    @staticmethod
    def _finite_parameter_value(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Value for {name!r} must be finite.")
        return value

    @staticmethod
    def _parameter_bound(name: str, value: float) -> float:
        value = float(value)
        if np.isnan(value):
            raise ValueError(f"Bound for {name!r} must not be NaN.")
        return value
