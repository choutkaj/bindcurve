from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FitStrategy = Literal["per_experiment", "per_compound_summary", "pooled"]
AggregationMethod = Literal["mean", "median"]
WeightingMode = Literal["none"]
ErrorMode = Literal["raise", "collect"]


@dataclass(frozen=True)
class FitSettings:
    """Settings controlling dose-response fitting."""

    strategy: FitStrategy = "per_experiment"
    replicate_aggregation: AggregationMethod = "mean"
    weighting: WeightingMode = "none"
    lmfit_method: str = "leastsq"
    errors: ErrorMode = "raise"
    max_nfev: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {"per_experiment", "per_compound_summary", "pooled"}:
            raise ValueError(
                "strategy must be 'per_experiment', "
                "'per_compound_summary', or 'pooled'."
            )
        if self.replicate_aggregation not in {"mean", "median"}:
            raise ValueError("replicate_aggregation must be 'mean' or 'median'.")
        if self.weighting != "none":
            raise NotImplementedError(
                "Only weighting='none' is implemented in this skeleton."
            )
        if self.errors not in {"raise", "collect"}:
            raise ValueError("errors must be 'raise' or 'collect'.")
