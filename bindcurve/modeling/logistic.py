from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.parameters import ParameterSpec


def _aggregate_for_guess(compound: CompoundData) -> tuple[np.ndarray, np.ndarray]:
    table = compound.aggregate_replicates(method="mean")
    concentration = table["concentration"].to_numpy(dtype=float)
    response = table["response"].to_numpy(dtype=float)
    return concentration, response


def _asymptote_and_midpoint_guess(response: np.ndarray) -> tuple[float, float, float]:
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    return ymin, ymax, midpoint


def _hill_slope_guess(response: np.ndarray) -> float:
    return 1.0 if response[-1] > response[0] else -1.0


class IC50Model(BaseDoseResponseModel):
    """Four-parameter IC50 / Hill dose-response model.

    The model is written as::

        y = ymin + (ymax - ymin) / (1 + (IC50 / x) ** hill_slope)

    A negative ``hill_slope`` describes a decreasing inhibition curve, while a
    positive ``hill_slope`` describes an increasing response curve.
    """

    name = "ic50"
    concentration_parameters = frozenset({"IC50"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("IC50", min=0.0, unit_kind="concentration"),
        ParameterSpec("hill_slope", initial=-1.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        IC50: float,
        hill_slope: float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return ymin + (ymax - ymin) / (1.0 + (IC50 / x) ** hill_slope)

    def guess(self, compound: CompoundData) -> dict[str, float]:
        concentration, response = _aggregate_for_guess(compound)
        ymin, ymax, midpoint = _asymptote_and_midpoint_guess(response)
        midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
        ic50 = float(concentration[midpoint_index])

        return {
            "ymin": ymin,
            "ymax": ymax,
            "IC50": ic50,
            "hill_slope": _hill_slope_guess(response),
        }


class EC50Model(BaseDoseResponseModel):
    """Four-parameter EC50 / Hill dose-response model.

    This model is mathematically identical to :class:`IC50Model`, but the fitted
    midpoint parameter is named ``EC50`` for activation or response-increase
    experiments.
    """

    name = "ec50"
    concentration_parameters = frozenset({"EC50"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("EC50", min=0.0, unit_kind="concentration"),
        ParameterSpec("hill_slope", initial=1.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        EC50: float,
        hill_slope: float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return ymin + (ymax - ymin) / (1.0 + (EC50 / x) ** hill_slope)

    def guess(self, compound: CompoundData) -> dict[str, float]:
        concentration, response = _aggregate_for_guess(compound)
        ymin, ymax, midpoint = _asymptote_and_midpoint_guess(response)
        midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
        ec50 = float(concentration[midpoint_index])

        return {
            "ymin": ymin,
            "ymax": ymax,
            "EC50": ec50,
            "hill_slope": _hill_slope_guess(response),
        }


class LogIC50Model(BaseDoseResponseModel):
    """Four-parameter logIC50 model fitted on log10 concentration.

    The model transforms concentration to ``log10(concentration)`` before
    fitting, then evaluates::

        y = ymin + (ymax - ymin) / (1 + 10 ** ((logIC50 - x) * hill_slope))

    where ``x`` is log10 concentration.
    """

    name = "logic50"
    log_concentration_parameters = frozenset({"logIC50"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("logIC50", unit_kind="log_concentration"),
        ParameterSpec("hill_slope", initial=-1.0),
    )

    def transform_x(self, concentration: np.ndarray) -> np.ndarray:
        concentration = np.asarray(concentration, dtype=float)
        return np.log10(concentration)

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        logIC50: float,
        hill_slope: float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return ymin + (ymax - ymin) / (1.0 + 10 ** ((logIC50 - x) * hill_slope))

    def guess(self, compound: CompoundData) -> dict[str, float]:
        concentration, response = _aggregate_for_guess(compound)
        ymin, ymax, midpoint = _asymptote_and_midpoint_guess(response)
        midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
        logic50 = float(np.log10(concentration[midpoint_index]))

        return {
            "ymin": ymin,
            "ymax": ymax,
            "logIC50": logic50,
            "hill_slope": _hill_slope_guess(response),
        }
