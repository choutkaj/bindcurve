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
        if not isinstance(self.lmfit_method, str) or not self.lmfit_method.strip():
            raise ValueError("lmfit_method must be a non-empty string.")
        if self.max_nfev is not None and (
            not isinstance(self.max_nfev, int)
            or isinstance(self.max_nfev, bool)
            or self.max_nfev <= 0
        ):
            raise ValueError("max_nfev must be a positive integer or None.")
