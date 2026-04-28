from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import lmfit
import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.parameters import ParameterSpec


class BaseDoseResponseModel(ABC):
    """Base class for dose-response models fitted through lmfit."""

    name: str
    parameter_specs: tuple[ParameterSpec, ...]
    concentration_parameters: frozenset[str] = frozenset()
    response_parameters: frozenset[str] = frozenset()

    @abstractmethod
    def evaluate(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the model at concentrations ``x``."""

    @abstractmethod
    def guess(self, compound: CompoundData) -> dict[str, float]:
        """Generate initial parameter guesses for one compound or experiment."""

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

        for spec in self.parameter_specs:
            value = fixed.get(spec.name, guesses.get(spec.name, spec.initial))
            if value is None:
                raise ValueError(f"No initial value available for {spec.name!r}.")

            lower, upper = bounds.get(spec.name, (spec.min, spec.max))
            if lower is None:
                lower = spec.min
            if upper is None:
                upper = spec.max

            parameters.add(
                spec.name,
                value=float(value),
                min=float(lower),
                max=float(upper),
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

    def parameter_unit(
        self,
        parameter_name: str,
        *,
        concentration_unit: str | None,
        response_unit: str | None,
    ) -> str | None:
        """Return the display unit for a model parameter."""
        if parameter_name in self.concentration_parameters:
            return concentration_unit
        if parameter_name in self.response_parameters:
            return response_unit
        return None
