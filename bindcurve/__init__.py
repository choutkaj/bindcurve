"""Fitting and plotting of dose-response and equilibrium-binding curves."""

from importlib.metadata import PackageNotFoundError, version

from bindcurve.conversion import cheng_prusoff, coleska, munson_rodbard
from bindcurve.data import DoseResponseData
from bindcurve.fitting import fit
from bindcurve.models import BindingModel, Model, Parameter, get_model
from bindcurve.plotting import plot_compounds, plot_fits, plot_residuals
from bindcurve.results import FitResult, FitResults

try:
    __version__ = version("bindcurve")
except PackageNotFoundError:  # pragma: no cover - source tree without metadata
    __version__ = "0+unknown"

__all__ = [
    "BindingModel",
    "DoseResponseData",
    "FitResult",
    "FitResults",
    "Model",
    "Parameter",
    "__version__",
    "cheng_prusoff",
    "coleska",
    "fit",
    "get_model",
    "munson_rodbard",
    "plot_compounds",
    "plot_fits",
    "plot_residuals",
]
