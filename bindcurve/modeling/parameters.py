from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

STRICTLY_POSITIVE_PARAMETER_MIN = np.finfo(float).tiny


@dataclass(frozen=True)
class ParameterSpec:
    """Complete specification for one public model parameter.

    ``scale`` controls only the optimizer coordinate. Model evaluation and all
    public results always use the parameter's physical, linear-scale value.
    """

    name: str
    initial: float | None = None
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True
    kind: Literal["native", "concentration"] = "native"
    scale: Literal["linear", "log10"] = "linear"
    reportable: bool = True
    log_name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"native", "concentration"}:
            raise ValueError("kind must be 'native' or 'concentration'.")
        if self.scale not in {"linear", "log10"}:
            raise ValueError("scale must be 'linear' or 'log10'.")
        if not np.isfinite(self.min) and self.min != -np.inf:
            raise ValueError(f"Minimum for {self.name!r} must not be NaN.")
        if not np.isfinite(self.max) and self.max != np.inf:
            raise ValueError(f"Maximum for {self.name!r} must not be NaN.")
        if self.min >= self.max:
            raise ValueError(f"Minimum for {self.name!r} must be below maximum.")
        if self.scale == "log10":
            if self.kind != "concentration":
                raise ValueError("Only concentration parameters may use log10 scale.")
            if self.min <= 0.0:
                raise ValueError(
                    f"Log-scaled parameter {self.name!r} requires a positive minimum."
                )
            if self.initial is not None and self.initial <= 0.0:
                raise ValueError(
                    f"Log-scaled parameter {self.name!r} requires a positive "
                    "initial value."
                )

    @property
    def required_fixed(self) -> bool:
        """Whether the caller must provide this non-varying parameter."""
        return not self.vary and self.initial is None

    @property
    def resolved_log_name(self) -> str:
        """Return the canonical name for the log10 representation."""
        if self.log_name is not None:
            return self.log_name
        return f"log{self.name}"
