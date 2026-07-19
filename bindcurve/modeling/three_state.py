from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.guesses import midpoint_guess
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ParameterSpec,
)


def _competitive_three_state_receptor_free(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    nonspecific_factor: float = 1.0,
) -> np.ndarray:
    LT = np.asarray(LT, dtype=float)
    if RT == 0.0:
        return np.zeros_like(LT, dtype=float)

    scaled_kd = nonspecific_factor * Kd
    concentration_scale = float(RT)
    normalized_LsT = LsT / concentration_scale
    normalized_Kds = Kds / concentration_scale
    normalized_Kd = scaled_kd / concentration_scale

    R_values = []
    for concentration in LT.ravel():
        normalized_LT = float(concentration) / concentration_scale

        def mass_balance(
            normalized_R: float,
            normalized_LT: float = normalized_LT,
        ) -> float:
            RLs = (
                normalized_LsT
                * normalized_R
                / (normalized_Kds + normalized_R)
            )
            RL = (
                normalized_LT
                * normalized_R
                / (normalized_Kd + normalized_R)
            )
            return normalized_R + RLs + RL - 1.0

        normalized_R = brentq(
            mass_balance,
            0.0,
            1.0,
            xtol=np.finfo(float).tiny,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )
        R_values.append(normalized_R * concentration_scale)

    return np.asarray(R_values, dtype=float).reshape(LT.shape)


def _competitive_three_state_specific_component_arrays(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
) -> dict[str, np.ndarray]:
    LT = np.asarray(LT, dtype=float)
    R = _competitive_three_state_receptor_free(
        LT,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
    )
    RLs = np.divide(
        LsT * R,
        Kds + R,
        out=np.zeros_like(R, dtype=float),
        where=(Kds + R) != 0.0,
    )
    Ls = np.clip(LsT - RLs, 0.0, None)
    RL = np.divide(
        LT * R,
        Kd + R,
        out=np.zeros_like(R, dtype=float),
        where=(Kd + R) != 0.0,
    )
    L = np.clip(LT - RL, 0.0, None)
    Fbs = np.divide(
        RLs,
        LsT,
        out=np.zeros_like(RLs, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "LT": LT,
        "RT": np.full_like(LT, RT, dtype=float),
        "R": R,
        "LsT": np.full_like(LT, LsT, dtype=float),
        "Ls": Ls,
        "L": L,
        "RLs": RLs,
        "RL": RL,
        "Fbs": Fbs,
    }


def _competitive_three_state_total_component_arrays(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    N: float,
    Kd: float,
) -> dict[str, np.ndarray]:
    LT = np.asarray(LT, dtype=float)
    effective_factor = 1.0 + N
    R = _competitive_three_state_receptor_free(
        LT,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        nonspecific_factor=effective_factor,
    )
    RLs = np.divide(
        LsT * R,
        Kds + R,
        out=np.zeros_like(R, dtype=float),
        where=(Kds + R) != 0.0,
    )
    Ls = np.clip(LsT - RLs, 0.0, None)
    RL = np.divide(
        LT * R,
        effective_factor * Kd + R,
        out=np.zeros_like(R, dtype=float),
        where=(effective_factor * Kd + R) != 0.0,
    )
    L = np.divide(
        LT * Kd,
        effective_factor * Kd + R,
        out=np.zeros_like(R, dtype=float),
        where=(effective_factor * Kd + R) != 0.0,
    )
    L_nonspecific_bound = N * L
    L_bound_total = RL + L_nonspecific_bound
    Fbs = np.divide(
        RLs,
        LsT,
        out=np.zeros_like(RLs, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "LT": LT,
        "RT": np.full_like(LT, RT, dtype=float),
        "R": R,
        "LsT": np.full_like(LT, LsT, dtype=float),
        "Ls": Ls,
        "L": L,
        "RLs": RLs,
        "RL": RL,
        "L_nonspecific_bound": L_nonspecific_bound,
        "L_bound_total": L_bound_total,
        "Fbs": Fbs,
    }


def _fixed_concentration(name: str) -> ParameterSpec:
    return ParameterSpec(
        name,
        min=STRICTLY_POSITIVE_PARAMETER_MIN,
        vary=False,
        kind="concentration",
        scale="log10",
        reportable=False,
    )


class CompetitiveThreeStateSpecificKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model for specific binding."""

    name = "comp_3st_specific"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        *(_fixed_concentration(name) for name in ("RT", "LsT", "Kds")),
        ParameterSpec(
            "Kd",
            min=STRICTLY_POSITIVE_PARAMETER_MIN,
            kind="concentration",
            scale="log10",
        ),
    )

    def _component_arrays(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _competitive_three_state_specific_component_arrays(
            concentration,
            RT=float(params["RT"]),
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
            Kd=float(params["Kd"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return midpoint_guess(compound, concentration_parameter="Kd")


class CompetitiveThreeStateTotalKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model with nonspecific binding."""

    name = "comp_3st_total"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        *(_fixed_concentration(name) for name in ("RT", "LsT", "Kds")),
        ParameterSpec("N", min=0.0, vary=False),
        ParameterSpec(
            "Kd",
            min=STRICTLY_POSITIVE_PARAMETER_MIN,
            kind="concentration",
            scale="log10",
        ),
    )

    def _component_arrays(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _competitive_three_state_total_component_arrays(
            concentration,
            RT=float(params["RT"]),
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
            N=float(params["N"]),
            Kd=float(params["Kd"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return midpoint_guess(compound, concentration_parameter="Kd")
