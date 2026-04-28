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


def _basic_guess(compound: CompoundData) -> dict[str, float]:
    concentration, response = _aggregate_for_guess(compound)
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
    kds_guess = float(concentration[midpoint_index])
    return {"ymin": ymin, "ymax": ymax, "Kds": kds_guess}


class DirectSimpleKdModel(BaseDoseResponseModel):
    """Simple direct-binding saturation model.

    The model assumes the x-axis is free receptor concentration and evaluates::

        fraction_bound = R / (Kds + R)
        y = ymin + (ymax - ymin) * fraction_bound
    """

    name = "dir_simple"
    concentration_parameters = frozenset({"Kds"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("Kds", min=0.0, unit_kind="concentration"),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        Kds: float,
    ) -> np.ndarray:
        receptor = np.asarray(x, dtype=float)
        fraction_bound = receptor / (Kds + receptor)
        return ymin + (ymax - ymin) * fraction_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class DirectSpecificKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion for specific binding."""

    name = "dir_specific"
    required_fixed_parameters = frozenset({"LsT"})
    concentration_parameters = frozenset({"Kds", "LsT"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("LsT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kds", min=0.0, unit_kind="concentration"),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        LsT: float,
        Kds: float,
    ) -> np.ndarray:
        receptor_total = np.asarray(x, dtype=float)
        a = Kds + LsT - receptor_total
        b = -Kds * receptor_total
        receptor_free = (-a + np.sqrt(a**2 - 4.0 * b)) / 2.0
        fraction_bound = receptor_free / (Kds + receptor_free)
        return ymin + (ymax - ymin) * fraction_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class DirectTotalKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion and nonspecific binding."""

    name = "dir_total"
    required_fixed_parameters = frozenset({"LsT", "Ns"})
    concentration_parameters = frozenset({"Kds", "LsT"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("LsT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Ns", min=0.0, vary=False),
        ParameterSpec("Kds", min=0.0, unit_kind="concentration"),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        LsT: float,
        Ns: float,
        Kds: float,
    ) -> np.ndarray:
        receptor_total = np.asarray(x, dtype=float)
        a = (1.0 + Ns) * Kds + LsT - receptor_total
        b = -Kds * receptor_total * (1.0 + Ns)
        receptor_free = (-a + np.sqrt(a**2 - 4.0 * b)) / 2.0
        fraction_bound = receptor_free / (Kds + receptor_free)
        return ymin + (ymax - ymin) * fraction_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)
