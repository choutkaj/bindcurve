"""Model and parameter definitions shared by all dose-response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """One model parameter.

    Parameters
    ----------
    name
        Keyword used in model evaluation, ``fixed`` and ``bounds``.
    concentration
        Positive concentration-like parameter. It is optimized on a log10
        coordinate and summarized across experiments on the log10 scale.
    fixed
        Assay constant that must be supplied through ``fit(..., fixed=...)``.
    min, max
        Physical limits of the parameter value.
    """

    name: str
    concentration: bool = False
    fixed: bool = False
    min: float = -np.inf
    max: float = np.inf


PLATEAUS = (Parameter("ymin"), Parameter("ymax"))


class Model(ABC):
    """Dose-response model mapping a fraction in [0, 1] onto two plateaus.

    The response is ``ymin + (ymax - ymin) * fraction(x)``. Subclasses define
    `name`, `parameters` (including ``ymin`` and ``ymax``) and `_fraction`.
    """

    name: str
    parameters: tuple[Parameter, ...]

    def evaluate(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the response at untransformed concentrations ``x >= 0``."""
        x, params = self._checked(x, params)
        fraction = self._fraction(x, **params)
        return params["ymin"] + (params["ymax"] - params["ymin"]) * fraction

    def guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Initial values from replicate means ``y`` at concentrations ``x``.

        The plateaus come from the response range and every concentration
        parameter from the concentration closest to the half response.
        """
        half = np.argmin(np.abs(y - (np.min(y) + np.max(y)) / 2.0))
        guesses = {"ymin": float(np.min(y)), "ymax": float(np.max(y))}
        for parameter in self.parameters:
            if parameter.concentration and not parameter.fixed:
                guesses[parameter.name] = float(x[half])
        return guesses

    def parameter(self, name: str) -> Parameter:
        """Return the parameter called ``name``."""
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"Model {self.name!r} has no parameter {name!r}.")

    @abstractmethod
    def _fraction(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Return the modeled fraction for validated inputs."""

    def _checked(
        self, x: np.ndarray, params: dict[str, float]
    ) -> tuple[np.ndarray, dict[str, float]]:
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x < 0.0):
            raise ValueError("Concentrations must be finite and non-negative.")
        expected = [parameter.name for parameter in self.parameters]
        if set(params) != set(expected):
            raise TypeError(
                f"Model {self.name!r} takes {expected}; got {sorted(params)}."
            )
        checked = {}
        for parameter in self.parameters:
            value = float(params[parameter.name])
            if not np.isfinite(value) or not parameter.min <= value <= parameter.max:
                raise ValueError(
                    f"{parameter.name} = {value} is outside "
                    f"[{parameter.min}, {parameter.max}]."
                )
            if parameter.concentration and value <= 0.0:
                raise ValueError(f"{parameter.name} must be positive.")
            checked[parameter.name] = value
        return x, checked


class BindingModel(Model):
    """Equilibrium-binding model whose fraction is the tracer-bound fraction."""

    def species(self, x: np.ndarray, **params: float) -> dict[str, np.ndarray]:
        """Return equilibrium species concentrations and the bound fraction ``Fbs``."""
        x, params = self._checked(x, params)
        return self._species(x, **params)

    def _fraction(self, x: np.ndarray, **params: float) -> np.ndarray:
        return self._species(x, **params)["Fbs"]

    @abstractmethod
    def _species(self, x: np.ndarray, **params: float) -> dict[str, np.ndarray]:
        """Return species for validated inputs."""
