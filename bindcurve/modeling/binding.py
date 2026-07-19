from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ConcentrationParameterSpec,
    ParameterSpec,
)


def _aggregate_for_guess(compound: CompoundData) -> tuple[np.ndarray, np.ndarray]:
    table = compound.aggregate_replicates()
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


def _direct_specific_receptor_free(
    RT: np.ndarray,
    *,
    LsT: float,
    Kds: float,
) -> np.ndarray:
    RT = np.asarray(RT, dtype=float)
    a = Kds + LsT - RT
    b = -Kds * RT
    square_root = np.hypot(a, 2.0 * np.sqrt(-b))
    direct_root = (-a + square_root) / 2.0
    stable_root = np.divide(
        -2.0 * b,
        a + square_root,
        out=np.zeros_like(RT, dtype=float),
        where=(a + square_root) != 0.0,
    )
    return np.where(a >= 0.0, stable_root, direct_root)


def _direct_total_receptor_free(
    RT: np.ndarray,
    *,
    LsT: float,
    Ns: float,
    Kds: float,
) -> np.ndarray:
    RT = np.asarray(RT, dtype=float)
    a = (1.0 + Ns) * Kds + LsT - RT
    b = -Kds * RT * (1.0 + Ns)
    square_root = np.hypot(a, 2.0 * np.sqrt(-b))
    direct_root = (-a + square_root) / 2.0
    stable_root = np.divide(
        -2.0 * b,
        a + square_root,
        out=np.zeros_like(RT, dtype=float),
        where=(a + square_root) != 0.0,
    )
    return np.where(a >= 0.0, stable_root, direct_root)


def _direct_specific_component_arrays(
    RT: np.ndarray,
    *,
    LsT: float,
    Kds: float,
) -> dict[str, np.ndarray]:
    RT = np.asarray(RT, dtype=float)
    R = _direct_specific_receptor_free(
        RT,
        LsT=LsT,
        Kds=Kds,
    )
    RLs = np.clip(RT - R, 0.0, None)
    Ls = np.clip(LsT - RLs, 0.0, None)
    Fbs = np.divide(
        RLs,
        LsT,
        out=np.zeros_like(RLs, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "RT": RT,
        "R": R,
        "LsT": np.full_like(RT, LsT, dtype=float),
        "Ls": Ls,
        "RLs": RLs,
        "Fbs": Fbs,
    }


def _direct_total_component_arrays(
    RT: np.ndarray,
    *,
    LsT: float,
    Ns: float,
    Kds: float,
) -> dict[str, np.ndarray]:
    """Return the direct-binding state with linear tracer immobilization.

    This follows Roehrl, Wang & Wagner, Biochemistry 2004, 43,
    16056-16066, equations 4 and 7 (doi:10.1021/bi048233g).
    Nonspecific binding is nonsaturable and proportional to free tracer:
    ``Ls_nonspecific_bound = Ns * Ls``.
    """
    RT = np.asarray(RT, dtype=float)
    R = _direct_total_receptor_free(
        RT,
        LsT=LsT,
        Ns=Ns,
        Kds=Kds,
    )
    Ls = np.divide(
        LsT,
        1.0 + Ns + R / Kds,
        out=np.zeros_like(R, dtype=float),
        where=(1.0 + Ns + R / Kds) != 0.0,
    )
    RLs = R * Ls / Kds
    Ls_nonspecific_bound = Ns * Ls
    Ls_bound_total = RLs + Ls_nonspecific_bound
    Fbs_specific = np.divide(
        RLs,
        LsT,
        out=np.zeros_like(RLs, dtype=float),
        where=LsT != 0.0,
    )
    Fbs_total = np.divide(
        Ls_bound_total,
        LsT,
        out=np.zeros_like(Ls_bound_total, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "RT": RT,
        "R": R,
        "LsT": np.full_like(RT, LsT, dtype=float),
        "Ls": Ls,
        "RLs": RLs,
        "Ls_nonspecific_bound": Ls_nonspecific_bound,
        "Ls_bound_total": Ls_bound_total,
        "Fbs_specific": Fbs_specific,
        "Fbs_total": Fbs_total,
        # With fitted lower and upper asymptotes, baseline-normalizing Roehrl's
        # total-bound fraction reduces exactly to the specific-bound fraction.
        "Fbs": Fbs_specific,
    }


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


class DirectSimpleKdModel(BaseDoseResponseModel):
    """Simple direct-binding saturation model.

    The model assumes the x-axis is free receptor concentration and evaluates::

        Fbs = R / (Kds + R)
        y = ymin + (ymax - ymin) * Fbs
    """

    name = "dir_simple"
    concentration_parameters = frozenset({"Kds"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("Kds", min=STRICTLY_POSITIVE_PARAMETER_MIN),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        Kds: float,
    ) -> np.ndarray:
        R = np.asarray(x, dtype=float)
        Fbs = R / (Kds + R)
        return ymin + (ymax - ymin) * Fbs

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        R = np.asarray(concentration, dtype=float)
        kds = float(params["Kds"])
        Fbs = np.divide(
            R,
            kds + R,
            out=np.zeros_like(R, dtype=float),
            where=(kds + R) != 0.0,
        )
        return {
            "R": R,
            "Fbs": Fbs,
        }

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class DirectSpecificKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion for specific binding."""

    name = "dir_specific"
    required_fixed_parameters = frozenset({"LsT"})
    concentration_parameters = frozenset({"Kds", "LsT"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("LsT", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("Kds", min=STRICTLY_POSITIVE_PARAMETER_MIN),
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
        RT = np.asarray(x, dtype=float)
        components = _direct_specific_component_arrays(
            RT,
            LsT=LsT,
            Kds=Kds,
        )
        Fbs = components["Fbs"]
        return ymin + (ymax - ymin) * Fbs

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _direct_specific_component_arrays(
            concentration,
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class DirectTotalKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion and nonspecific binding."""

    name = "dir_total"
    required_fixed_parameters = frozenset({"LsT", "Ns"})
    concentration_parameters = frozenset({"Kds", "LsT"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("LsT", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("Ns", min=0.0, vary=False),
        ParameterSpec("Kds", min=STRICTLY_POSITIVE_PARAMETER_MIN),
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
        RT = np.asarray(x, dtype=float)
        components = _direct_total_component_arrays(
            RT,
            LsT=LsT,
            Ns=Ns,
            Kds=Kds,
        )
        Fbs = components["Fbs"]
        return ymin + (ymax - ymin) * Fbs

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _direct_total_component_arrays(
            concentration,
            LsT=float(params["LsT"]),
            Ns=float(params["Ns"]),
            Kds=float(params["Kds"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class CompetitiveThreeStateSpecificKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model for specific binding."""

    name = "comp_3st_specific"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="RT",
            fitted_parameter="RT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kd",
            fitted_parameter="Kd",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("RT", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("LsT", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("Kds", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("Kd", min=STRICTLY_POSITIVE_PARAMETER_MIN),
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
        components = _competitive_three_state_specific_component_arrays(
            np.asarray(x, dtype=float),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
        )
        Fbs = components["Fbs"]
        return ymin + (ymax - ymin) * Fbs

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
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
        return _competition_guess(compound)


class CompetitiveThreeStateTotalKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model with nonspecific binding."""

    name = "comp_3st_total"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "N"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="RT",
            fitted_parameter="RT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kd",
            fitted_parameter="Kd",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("RT", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("LsT", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("Kds", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
        ParameterSpec("N", min=0.0, vary=False),
        ParameterSpec("Kd", min=STRICTLY_POSITIVE_PARAMETER_MIN),
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
        components = _competitive_three_state_total_component_arrays(
            np.asarray(x, dtype=float),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            N=N,
            Kd=Kd,
        )
        Fbs = components["Fbs"]
        return ymin + (ymax - ymin) * Fbs

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
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
        return _competition_guess(compound)
