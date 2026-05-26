from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorMode = Literal["raise", "collect"]


@dataclass(frozen=True)
class FitSettings:
    """Settings controlling dose-response fitting."""

    lmfit_method: str = "leastsq"
    errors: ErrorMode = "raise"
    max_nfev: int | None = None

    def __post_init__(self) -> None:
        if self.errors not in {"raise", "collect"}:
            raise ValueError("errors must be 'raise' or 'collect'.")
