from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

QualitySeverity = Literal["orange", "red"]
QualityStatus = Literal["green", "orange", "red"]


@dataclass(frozen=True)
class DataQualityThresholds:
    """Heuristic thresholds for data-level dose-response QC."""

    min_experiments_green: int = 3
    max_intra_noise_median_frac_range_orange: float = 0.05
    max_intra_noise_median_frac_range_red: float = 0.10
    max_intra_noise_p90_frac_range_orange: float = 0.10
    max_intra_noise_p90_frac_range_red: float = 0.20

    def __post_init__(self) -> None:
        _validate_nonnegative_int(
            "min_experiments_green",
            self.min_experiments_green,
            minimum=2,
        )
        _validate_threshold_pair(
            "max_intra_noise_median_frac_range",
            orange=self.max_intra_noise_median_frac_range_orange,
            red=self.max_intra_noise_median_frac_range_red,
        )
        _validate_threshold_pair(
            "max_intra_noise_p90_frac_range",
            orange=self.max_intra_noise_p90_frac_range_orange,
            red=self.max_intra_noise_p90_frac_range_red,
        )


@dataclass(frozen=True)
class ResultQualityThresholds:
    """Heuristic thresholds for fit- and summary-level QC."""

    min_experiments_green: int = 3
    max_inter_ci95_fold_orange: float = 3.0
    max_inter_ci95_fold_red: float = 10.0
    bound_tolerance_rel: float = 1.0e-6
    bound_tolerance_abs: float = 1.0e-12

    def __post_init__(self) -> None:
        _validate_nonnegative_int(
            "min_experiments_green",
            self.min_experiments_green,
            minimum=2,
        )
        _validate_threshold_pair(
            "max_inter_ci95_fold",
            orange=self.max_inter_ci95_fold_orange,
            red=self.max_inter_ci95_fold_red,
            minimum=1.0,
        )
        _validate_nonnegative_float("bound_tolerance_rel", self.bound_tolerance_rel)
        _validate_nonnegative_float("bound_tolerance_abs", self.bound_tolerance_abs)


def resolve_requested_compounds(
    available: Iterable[str],
    compounds: str | Iterable[str] | None,
) -> list[str]:
    """Return requested compound identifiers while preserving user order."""
    available_list = [str(compound_id) for compound_id in available]
    if compounds is None:
        return available_list

    if isinstance(compounds, str):
        requested = [str(compounds)]
    else:
        requested = [str(value) for value in compounds]

    missing = [
        compound_id for compound_id in requested if compound_id not in available_list
    ]
    if missing:
        raise KeyError(f"Unknown compound(s): {missing}")
    return requested


def summarize_quality_flags(
    flags: list[tuple[QualitySeverity, str]],
) -> tuple[QualityStatus, int, int, str]:
    """Collapse flag tuples into status counters and a readable message string."""
    orange_flags = [message for severity, message in flags if severity == "orange"]
    red_flags = [message for severity, message in flags if severity == "red"]
    if red_flags:
        status: QualityStatus = "red"
    elif orange_flags:
        status = "orange"
    else:
        status = "green"
    messages = [message for _, message in flags]
    return status, len(orange_flags), len(red_flags), "; ".join(messages)


def _validate_threshold_pair(
    name: str,
    *,
    orange: float,
    red: float,
    minimum: float = 0.0,
) -> None:
    _validate_nonnegative_float(f"{name}_orange", orange, minimum=minimum)
    _validate_nonnegative_float(f"{name}_red", red, minimum=minimum)
    if red < orange:
        raise ValueError(f"{name}_red must be greater than or equal to {name}_orange.")


def _validate_nonnegative_float(
    name: str,
    value: float,
    *,
    minimum: float = 0.0,
) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")


def _validate_nonnegative_int(
    name: str,
    value: int,
    *,
    minimum: int = 0,
) -> None:
    if not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
