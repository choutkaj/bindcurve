"""Object-oriented dose-response fitting with an lmfit backend."""

from importlib.metadata import PackageNotFoundError, version

from bindcurve.datasets import CompoundData, DoseResponseData
from bindcurve.fitting import FitCalculator, FitSettings, fit
from bindcurve.modeling import (
    BaseDoseResponseModel,
    DirectSimpleKdModel,
    DirectSpecificKdModel,
    DirectTotalKdModel,
    EC50Model,
    IC50Model,
    LogIC50Model,
    ParameterSpec,
    get_model,
)
from bindcurve.results import FitMetrics, FitResult, FitResults, ParameterEstimate

try:
    __version__ = version("bindcurve")
except PackageNotFoundError:  # pragma: no cover - useful in editable source trees
    __version__ = "0+unknown"

__all__ = [
    "BaseDoseResponseModel",
    "CompoundData",
    "DirectSimpleKdModel",
    "DirectSpecificKdModel",
    "DirectTotalKdModel",
    "DoseResponseData",
    "EC50Model",
    "FitCalculator",
    "FitMetrics",
    "FitResult",
    "FitResults",
    "FitSettings",
    "IC50Model",
    "LogIC50Model",
    "ParameterEstimate",
    "ParameterSpec",
    "__version__",
    "fit",
    "get_model",
]
