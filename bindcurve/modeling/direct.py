from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.guesses import midpoint_guess
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ParameterSpec,
)


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
    R = _direct_specific_receptor_free(RT, LsT=LsT, Kds=Kds)
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
    R = _direct_total_receptor_free(RT, LsT=LsT, Ns=Ns, Kds=Kds)
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
        # Fitted lower and upper asymptotes baseline-normalize Roehrl's
        # total-bound fraction to the specific-bound fraction.
        "Fbs": Fbs_specific,
    }


class DirectSimpleKdModel(BaseDoseResponseModel):
    """Simple direct-binding saturation model using free receptor on the x-axis."""

    name = "dir_simple"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec(
            "Kds",
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
        R = np.asarray(concentration, dtype=float)
        Kds = float(params["Kds"])
        Fbs = np.divide(
            R,
            Kds + R,
            out=np.zeros_like(R, dtype=float),
            where=(Kds + R) != 0.0,
        )
        return {"R": R, "Fbs": Fbs}

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return midpoint_guess(compound, concentration_parameter="Kds")


class DirectSpecificKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion for specific binding."""

    name = "dir_specific"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec(
            "LsT",
            min=STRICTLY_POSITIVE_PARAMETER_MIN,
            vary=False,
            kind="concentration",
            scale="log10",
            reportable=False,
        ),
        ParameterSpec(
            "Kds",
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
        return _direct_specific_component_arrays(
            concentration,
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return midpoint_guess(compound, concentration_parameter="Kds")


class DirectTotalKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion and nonspecific binding."""

    name = "dir_total"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec(
            "LsT",
            min=STRICTLY_POSITIVE_PARAMETER_MIN,
            vary=False,
            kind="concentration",
            scale="log10",
            reportable=False,
        ),
        ParameterSpec("Ns", min=0.0, vary=False),
        ParameterSpec(
            "Kds",
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
        return _direct_total_component_arrays(
            concentration,
            LsT=float(params["LsT"]),
            Ns=float(params["Ns"]),
            Kds=float(params["Kds"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return midpoint_guess(compound, concentration_parameter="Kds")
