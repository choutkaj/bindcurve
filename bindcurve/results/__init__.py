"""Structured fit result containers."""

from bindcurve.quality import ResultQualityThresholds
from bindcurve.results.core import FitResults
from bindcurve.results.types import (
    ConcentrationSummary,
    FitMetrics,
    FitResult,
    ParameterEstimate,
    ParameterSummary,
)

__all__ = [
    "ConcentrationSummary",
    "FitMetrics",
    "FitResult",
    "FitResults",
    "ParameterEstimate",
    "ParameterSummary",
    "ResultQualityThresholds",
]
