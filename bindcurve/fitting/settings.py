from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WeightingMode = Literal["none"]
ErrorMode = Literal["raise", "collect"]


@dataclass(frozen=True)
class FitSettings:
    """Settings controlling dose-response fitting."""

    weighting: WeightingMode = "none"
    lmfit_method: str = "leastsq"
    errors: ErrorMode = "raise"
    max_nfev: int | None = None

    def __post_init__(self) -> None:
        if self.weighting != "none":
            raise NotImplementedError(
                "Only weighting='none' is implemented in this skeleton."
            )
        if self.errors not in {"raise", "collect"}:
            raise ValueError("errors must be 'raise' or 'collect'.")
