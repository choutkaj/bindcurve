from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a fitted model parameter."""

    name: str
    initial: float | None = None
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True
    description: str | None = None


@dataclass(frozen=True)
class ConcentrationParameterSpec:
    """Specification for one concentration-like quantity in a model."""

    parameter: str
    fitted_parameter: str
    fitted_scale: Literal["linear", "log10"] = "linear"
    reportable: bool = True
    log_parameter: str | None = None

    def __post_init__(self) -> None:
        if self.fitted_scale not in {"linear", "log10"}:
            raise ValueError("fitted_scale must be 'linear' or 'log10'.")

    @property
    def resolved_log_parameter(self) -> str:
        """Return the canonical log10 face name of the quantity."""
        if self.log_parameter is not None:
            return self.log_parameter
        return f"log{self.parameter}"
