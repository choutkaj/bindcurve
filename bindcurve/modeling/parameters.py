from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a fitted model parameter."""

    name: str
    initial: float | None = None
    min: float = -np.inf
    max: float = np.inf
    vary: bool = True
    unit_kind: str | None = None
    description: str | None = None
