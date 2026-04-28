"""Object-oriented dose-response fitting with an lmfit backend."""

from importlib.metadata import PackageNotFoundError, version

from bindcurve.conversion import (
    IC50ConversionResult,
    cheng_prusoff,
    cheng_prusoff_corrected,
    coleska,
    convert_ic50_to_kd,
)
from bindcurve.datasets import CompoundData, DoseResponseData
from bindcurve.fitting import FitCalculator, FitSettings, fit
from bindcurve.modeling import (
    BaseDoseResponseModel,
    CompetitiveFourStateSpecificKdModel,
    CompetitiveFourStateTotalKdModel,
    CompetitiveThreeStateSpecificKdModel,
    CompetitiveThreeStateTotalKdModel,
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
    "CompetitiveFourStateSpecificKdModel",
    "CompetitiveFourStateTotalKdModel",
    "CompetitiveThreeStateSpecificKdModel",
    "CompetitiveThreeStateTotalKdModel",
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
    "IC50ConversionResult",
    "IC50Model",
    "LogIC50Model",
    "ParameterEstimate",
    "ParameterSpec",
    "__version__",
    "cheng_prusoff",
    "cheng_prusoff_corrected",
    "coleska",
    "convert_ic50_to_kd",
    "fit",
    "get_model",
]
