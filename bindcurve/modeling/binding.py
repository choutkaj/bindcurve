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


def _competition_guess(compound: CompoundData) -> dict[str, float]:
    concentration, response = _aggregate_for_guess(compound)
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
    kd_guess = float(concentration[midpoint_index])
    return {"ymin": ymin, "ymax": ymax, "Kd": kd_guess}


def _competitive_three_state_receptor_free(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    nonspecific_factor: float = 1.0,
) -> np.ndarray:
    ligand_total = np.asarray(ligand_total, dtype=float)
    scaled_kd = nonspecific_factor * Kd

    a = Kds + scaled_kd + LsT + ligand_total - RT
    b = Kds * (ligand_total - RT) + scaled_kd * (LsT - RT) + Kds * scaled_kd
    c = -Kds * scaled_kd * RT

    discriminant = np.maximum(a**2 - 3.0 * b, 0.0)
    denominator = 2.0 * np.sqrt(discriminant**3)
    numerator = -2.0 * a**3 + 9.0 * a * b - 27.0 * c
    argument = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )
    theta = np.arccos(np.clip(argument, -1.0, 1.0))
    return -(a / 3.0) + (2.0 / 3.0) * np.sqrt(discriminant) * np.cos(theta / 3.0)


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


class CompetitiveThreeStateSpecificKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model for specific binding."""

    name = "comp_3st_specific"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("RT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("LsT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kds", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kd", min=0.0, unit_kind="concentration"),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        RT: float,
        LsT: float,
        Kds: float,
        Kd: float,
    ) -> np.ndarray:
        receptor_free = _competitive_three_state_receptor_free(
            x,
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
        )
        fraction_tracer_bound = receptor_free / (Kds + receptor_free)
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)


class CompetitiveThreeStateTotalKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model with nonspecific binding."""

    name = "comp_3st_total"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "N"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("RT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("LsT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kds", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("N", min=0.0, vary=False),
        ParameterSpec("Kd", min=0.0, unit_kind="concentration"),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        RT: float,
        LsT: float,
        Kds: float,
        N: float,
        Kd: float,
    ) -> np.ndarray:
        receptor_free = _competitive_three_state_receptor_free(
            x,
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
            nonspecific_factor=1.0 + N,
        )
        fraction_tracer_bound = receptor_free / (Kds + receptor_free)
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)
