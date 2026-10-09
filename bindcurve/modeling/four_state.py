from __future__ import annotations

from functools import partial

import numpy as np
from scipy.optimize import brentq

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.guesses import midpoint_guess
from bindcurve.modeling.parameters import ParameterSpec, concentration_spec


def _competition_guess(compound: CompoundData) -> dict[str, float]:
    return midpoint_guess(compound, concentration_parameter="Kd")


def _competitive_four_state_receptor_free(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    """Return free receptor by solving the receptor mass balance on [0, RT].

    For a given free receptor, both free ligands follow from a quadratic. A
    thermodynamically consistent four-state system has exactly one positive
    equilibrium, and the receptor balance changes sign on ``0 <= R <= RT``, so
    bracketing always finds the physical root.
    """
    LT = np.asarray(LT, dtype=float)
    if RT == 0.0:
        return np.zeros_like(LT, dtype=float)

    # Normalize every concentration by RT. This keeps the root interval at
    # [0, 1] and makes the solution invariant to concentration units.
    concentration_scale = float(RT)
    R_values = [
        concentration_scale
        * _solve_four_state_receptor_mass_balance(
            float(concentration) / concentration_scale,
            LsT=LsT / concentration_scale,
            Kds=Kds / concentration_scale,
            Kd=Kd / concentration_scale,
            Kd3=Kd3 / concentration_scale,
        )
        for concentration in LT.ravel()
    ]
    return np.asarray(R_values, dtype=float).reshape(LT.shape)


def _competitive_four_state_ligand_free(
    R: np.ndarray,
    LT: np.ndarray,
    *,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    """Return free competitor concentration for the specific four-state model."""
    R = np.asarray(R, dtype=float)
    LT = np.asarray(LT, dtype=float)

    a = 1.0 + R / Kds
    b = 1.0 + R / Kd
    c = R / (Kd * Kd3)

    quadratic_a = b * c
    quadratic_b = a * b + c * LsT - c * LT
    quadratic_c = -a * LT

    discriminant = np.maximum(
        quadratic_b**2 - 4.0 * quadratic_a * quadratic_c,
        0.0,
    )
    square_root = np.sqrt(discriminant)
    fallback = np.divide(
        -quadratic_c,
        quadratic_b,
        out=np.zeros_like(LT, dtype=float),
        where=quadratic_b != 0.0,
    )
    L = np.array(fallback, copy=True, dtype=float)

    positive_b = (quadratic_a > np.finfo(float).tiny) & (quadratic_b >= 0.0)
    stable_denominator = quadratic_b + square_root
    np.divide(
        -2.0 * quadratic_c,
        stable_denominator,
        out=L,
        where=positive_b & (stable_denominator > 0.0),
    )

    negative_b = (quadratic_a > np.finfo(float).tiny) & (quadratic_b < 0.0)
    np.divide(
        -quadratic_b + square_root,
        2.0 * quadratic_a,
        out=L,
        where=negative_b,
    )
    return np.maximum(L, 0.0)


def _competitive_four_state_mass_balance_residual(
    R: float,
    LT: float,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> float:
    """Return the receptor mass-balance residual for one candidate root."""
    R_array = np.asarray(R, dtype=float)
    LT_array = np.asarray(LT, dtype=float)
    L = _competitive_four_state_ligand_free(
        R_array,
        LT_array,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    ).item()
    Ls = LsT / (1.0 + R / Kds + R * L / (Kd * Kd3))
    RLs = R * Ls / Kds
    RL = R * L / Kd
    RLLs = R * L * Ls / (Kd * Kd3)
    return float(R + RLs + RL + RLLs - RT)


def _solve_four_state_receptor_mass_balance(
    LT: float,
    *,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> float:
    """Return free receptor normalized by total receptor (``RT = 1``)."""
    residual = partial(
        _competitive_four_state_mass_balance_residual,
        LT=LT,
        RT=1.0,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    # residual(0) = -1 and residual(1) = bound receptor >= 0.
    return float(
        brentq(
            residual,
            0.0,
            1.0,
            # The normalized physical root may be far below 1e-14 when tracer
            # is present in extreme excess, so an ordinary absolute tolerance
            # can collapse a valid positive root to zero.
            xtol=np.finfo(float).tiny,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )
    )


def _competitive_four_state_specific_component_arrays(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> dict[str, np.ndarray]:
    LT = np.asarray(LT, dtype=float)
    R = _competitive_four_state_receptor_free(
        LT,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    L = _competitive_four_state_ligand_free(
        R,
        LT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    Ls = np.divide(
        LsT,
        1.0 + R / Kds + R * L / (Kd * Kd3),
        out=np.zeros_like(R, dtype=float),
        where=(1.0 + R / Kds + R * L / (Kd * Kd3)) != 0.0,
    )
    RLs = np.divide(
        R * Ls,
        Kds,
        out=np.zeros_like(R, dtype=float),
        where=Kds != 0.0,
    )
    RL = np.divide(
        R * L,
        Kd,
        out=np.zeros_like(R, dtype=float),
        where=Kd != 0.0,
    )
    RLLs = np.divide(
        R * L * Ls,
        Kd * Kd3,
        out=np.zeros_like(R, dtype=float),
        where=(Kd * Kd3) != 0.0,
    )
    Fbs = np.divide(
        RLs + RLLs,
        LsT,
        out=np.zeros_like(R, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "LT": LT,
        "RT": np.full_like(LT, RT, dtype=float),
        "R": R,
        "L": L,
        "LsT": np.full_like(LT, LsT, dtype=float),
        "Ls": Ls,
        "RLs": RLs,
        "RL": RL,
        "RLLs": RLLs,
        "Fbs": Fbs,
    }


def _competitive_four_state_total_component_arrays(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
    N: float,
) -> dict[str, np.ndarray]:
    LT = np.asarray(LT, dtype=float)
    effective_kd = (1.0 + N) * Kd
    effective_components = _competitive_four_state_specific_component_arrays(
        LT,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=effective_kd,
        Kd3=Kd3,
    )
    # Under Roehrl et al. eq 29, LT = L + RL + RLLs + N*L. Replacing
    # Kd by (1 + N)*Kd solves the same observable equilibrium, but the
    # specific solver's apparent free ligand is (1 + N)*L.
    L = np.divide(
        effective_components["L"],
        1.0 + N,
        out=np.zeros_like(LT, dtype=float),
        where=(1.0 + N) != 0.0,
    )
    L_bound_specific = effective_components["RL"] + effective_components["RLLs"]
    L_nonspecific_bound = N * L
    L_bound_total = L_bound_specific + L_nonspecific_bound
    RLs_plus_RLLs = effective_components["RLs"] + effective_components["RLLs"]
    return {
        "LT": LT,
        "RT": np.full_like(LT, RT, dtype=float),
        "R": effective_components["R"],
        "L": L,
        "LsT": np.full_like(LT, LsT, dtype=float),
        "Ls": effective_components["Ls"],
        "RLs": effective_components["RLs"],
        "RL": effective_components["RL"],
        "RLLs": effective_components["RLLs"],
        "RLs_plus_RLLs": RLs_plus_RLLs,
        "L_bound_total": L_bound_total,
        "L_bound_specific": L_bound_specific,
        "L_nonspecific_bound": L_nonspecific_bound,
        "Fbs": effective_components["Fbs"],
    }


class CompetitiveFourStateSpecificKdModel(BaseDoseResponseModel):
    """Four-state competitive-binding model for specific binding."""

    name = "comp_4st_specific"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        *(
            concentration_spec(name, vary=False, reportable=False)
            for name in ("RT", "LsT", "Kds", "Kd3")
        ),
        concentration_spec("Kd"),
    )

    def _component_arrays(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _competitive_four_state_specific_component_arrays(
            concentration,
            RT=float(params["RT"]),
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
            Kd=float(params["Kd"]),
            Kd3=float(params["Kd3"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)


class CompetitiveFourStateTotalKdModel(BaseDoseResponseModel):
    """Four-state competitive-binding model with nonspecific binding."""

    name = "comp_4st_total"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        *(
            concentration_spec(name, vary=False, reportable=False)
            for name in ("RT", "LsT", "Kds", "Kd3")
        ),
        ParameterSpec("N", min=0.0, vary=False),
        concentration_spec("Kd"),
    )

    def _component_arrays(
        self,
        concentration: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _competitive_four_state_total_component_arrays(
            concentration,
            RT=float(params["RT"]),
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
            Kd=float(params["Kd"]),
            Kd3=float(params["Kd3"]),
            N=float(params["N"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)
