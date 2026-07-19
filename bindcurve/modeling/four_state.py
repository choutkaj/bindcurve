from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.optimize import brentq

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.guesses import midpoint_guess
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ParameterSpec,
)


def _competition_guess(compound: CompoundData) -> dict[str, float]:
    return midpoint_guess(compound, concentration_parameter="Kd")


def _competitive_four_state_coefficients(
    LT: float,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    """Return quintic coefficients for true free receptor in the four-state model.

    The polynomial variable is literal free receptor concentration ``R``:

    ``a*R**5 + b*R**4 + c*R**3 + d*R**2 + e*R + f = 0``.

    After the physical root is selected in ``0 <= R <= RT``, the observable
    tracer-bound fraction is computed from the actual four-state species
    ``RLs + RLLs`` rather than from a transformed receptor-like coordinate.

    The total/nonspecific model should call this with an effective ``Kd`` of
    ``(1 + N) * Kd`` rather than maintaining a duplicated coefficient
    expression.
    """
    LT = float(LT)

    a = Kds - Kd3
    b = (
        -Kd3 * Kd
        - Kd3 * Kds
        - Kd3 * LT
        + Kd3 * RT
        - Kd3 * LsT
        + Kd * Kds
        + Kds**2
        + Kds * LT
        - 2.0 * Kds * RT
        + Kds * LsT
    )
    c = (
        Kd3 * Kd * RT
        - Kd3 * Kd * LsT
        - Kd3 * Kds * LT
        + Kd3 * Kds * RT
        + Kd * Kds**2
        - 2.0 * Kd * Kds * RT
        + 2.0 * Kd * Kds * LsT
        + 2.0 * Kds**2 * LT
        - 2.0 * Kds**2 * RT
        - Kds * LT * RT
        + Kds * LT * LsT
        + Kds * RT**2
        - Kds * RT * LsT
    )
    d = (
        Kd3 * Kd**2 * Kds
        + Kd3 * Kd * Kds**2
        + Kd3 * Kd * Kds * LT
        + Kd3 * Kd * Kds * LsT
        + Kd * Kds**2 * LT
        - 2.0 * Kd * Kds**2 * RT
        + Kd * Kds**2 * LsT
        + Kd * Kds * RT**2
        - 2.0 * Kd * Kds * RT * LsT
        + Kd * Kds * LsT**2
        + Kds**2 * LT**2
        - 2.0 * Kds**2 * LT * RT
        + Kds**2 * RT**2
    )
    e = (
        Kd3 * Kd**2 * Kds**2
        - Kd3 * Kd**2 * Kds * RT
        + Kd3 * Kd**2 * Kds * LsT
        + Kd3 * Kd * Kds**2 * LT
        - Kd3 * Kd * Kds**2 * RT
        - Kd * Kds**2 * LT * RT
        + Kd * Kds**2 * LT * LsT
        + Kd * Kds**2 * RT**2
        - Kd * Kds**2 * RT * LsT
    )
    f = -Kd3 * Kd**2 * Kds**2 * RT

    return np.array([a, b, c, d, e, f], dtype=float)


def _trim_leading_near_zero(
    coefficients: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Remove leading coefficients that are zero at working precision."""
    coefficients = np.asarray(coefficients, dtype=float)
    scale = float(np.max(np.abs(coefficients))) if coefficients.size else 0.0
    if scale == 0.0:
        return np.array([0.0], dtype=float)

    threshold = relative_tolerance * scale
    for index, coefficient in enumerate(coefficients):
        if abs(float(coefficient)) > threshold:
            return coefficients[index:]

    return np.array([0.0], dtype=float)


def _scaled_polynomial_residual(coefficients: np.ndarray, root: float) -> float:
    coefficients = _trim_leading_near_zero(coefficients)
    degree = len(coefficients) - 1
    root_scale = max(1.0, abs(root))
    denominator = sum(
        abs(coefficient) * root_scale ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )
    if denominator == 0.0:
        denominator = 1.0
    return abs(float(np.polyval(coefficients, root))) / denominator


def _select_physical_root(
    coefficients: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    candidate_score: Callable[[float], float] | None = None,
    score_tolerance: float = 1.0e-10,
    imaginary_tolerance: float = 1.0e-7,
    interval_tolerance: float = 1.0e-8,
) -> float:
    """Select the physical four-state free-receptor root.

    The physical root must be effectively real and lie in the feasible interval
    for literal free receptor concentration. For the four-state receptor
    polynomial this interval is ``0 <= R <= RT``. If ``candidate_score`` is
    provided, candidates must also reconstruct a physically consistent state;
    otherwise, the scaled polynomial residual is used as the selector.
    """
    coefficients = _trim_leading_near_zero(coefficients)
    roots = np.roots(coefficients)
    interval_scale = max(1.0, abs(lower_bound), abs(upper_bound))
    lower = lower_bound - interval_tolerance * interval_scale
    upper = upper_bound + interval_tolerance * interval_scale

    candidates: list[float] = []
    for root in roots:
        real_part = float(np.real(root))
        imaginary_part = float(abs(np.imag(root)))
        if imaginary_part > imaginary_tolerance * max(1.0, abs(real_part)):
            continue
        if lower <= real_part <= upper:
            candidates.append(float(np.clip(real_part, lower_bound, upper_bound)))

    if not candidates:
        raise ValueError(
            "No physical four-state root found in the feasible free-receptor "
            f"interval {lower_bound} <= R <= {upper_bound}. Roots were: "
            f"{roots!r}"
        )

    if candidate_score is None:
        return min(
            candidates,
            key=lambda root: _scaled_polynomial_residual(coefficients, root),
        )

    scored_candidates = [
        (
            float(candidate_score(root)),
            _scaled_polynomial_residual(coefficients, root),
            root,
        )
        for root in candidates
    ]
    score, _, selected = min(scored_candidates)
    if not np.isfinite(score) or score > score_tolerance:
        raise ValueError(
            "No four-state polynomial root satisfied the physical mass balances. "
            f"Candidate scores were: {scored_candidates!r}"
        )
    return selected


def _competitive_four_state_receptor_free(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    LT = np.asarray(LT, dtype=float)
    if RT == 0.0:
        return np.zeros_like(LT, dtype=float)

    # Normalize every concentration by RT before constructing the polynomial.
    # This keeps the root interval at [0, 1] and makes root selection invariant
    # to a consistent change of concentration units.
    concentration_scale = float(RT)
    normalized_LsT = LsT / concentration_scale
    normalized_Kds = Kds / concentration_scale
    normalized_Kd = Kd / concentration_scale
    normalized_Kd3 = Kd3 / concentration_scale

    flat_LT = LT.ravel()
    R_values = []

    for concentration in flat_LT:
        normalized_LT = float(concentration) / concentration_scale
        coefficients = _competitive_four_state_coefficients(
            normalized_LT,
            RT=1.0,
            LsT=normalized_LsT,
            Kds=normalized_Kds,
            Kd=normalized_Kd,
            Kd3=normalized_Kd3,
        )
        score_candidate = partial(
            _competitive_four_state_mass_balance_score,
            LT=normalized_LT,
            RT=1.0,
            LsT=normalized_LsT,
            Kds=normalized_Kds,
            Kd=normalized_Kd,
            Kd3=normalized_Kd3,
        )

        try:
            normalized_R = _select_physical_root(
                coefficients,
                lower_bound=0.0,
                upper_bound=1.0,
                candidate_score=score_candidate,
            )
        except ValueError:
            normalized_R = _solve_four_state_receptor_mass_balance(
                normalized_LT,
                RT=1.0,
                LsT=normalized_LsT,
                Kds=normalized_Kds,
                Kd=normalized_Kd,
                Kd3=normalized_Kd3,
            )
        R_values.append(normalized_R * concentration_scale)

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
    Ls = LsT / (
        1.0 + R / Kds + R * L / (Kd * Kd3)
    )
    RLs = R * Ls / Kds
    RL = R * L / Kd
    RLLs = R * L * Ls / (Kd * Kd3)
    return float(R + RLs + RL + RLLs - RT)


def _competitive_four_state_mass_balance_score(
    R: float,
    LT: float,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> float:
    """Score a candidate using physical bounds and normalized mass balance."""
    R_array = np.asarray(R, dtype=float)
    L = _competitive_four_state_ligand_free(
        R_array,
        np.asarray(LT, dtype=float),
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    ).item()
    Ls = LsT / (
        1.0 + R / Kds + R * L / (Kd * Kd3)
    )
    tolerance = 1.0e-8
    if (
        not np.isfinite(L)
        or not np.isfinite(Ls)
        or L < -tolerance * max(1.0, LT)
        or L > LT + tolerance * max(1.0, LT)
        or Ls < -tolerance * max(1.0, LsT)
        or Ls > LsT + tolerance * max(1.0, LsT)
    ):
        return np.inf
    residual = _competitive_four_state_mass_balance_residual(
        R,
        LT,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    return abs(residual) / max(abs(RT), np.finfo(float).tiny)


def _solve_four_state_receptor_mass_balance(
    LT: float,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> float:
    """Solve the physical receptor balance directly as a robust fallback."""
    residual = partial(
        _competitive_four_state_mass_balance_residual,
        LT=LT,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    lower_residual = residual(0.0)
    upper_residual = residual(RT)
    tolerance = 1.0e-12 * max(1.0, abs(RT))
    if abs(lower_residual) <= tolerance:
        return 0.0
    if abs(upper_residual) <= tolerance:
        return float(RT)
    if lower_residual >= 0.0 or upper_residual <= 0.0:
        raise ValueError(
            "Could not bracket the physical four-state receptor mass balance: "
            f"f(0)={lower_residual}, f(RT)={upper_residual}."
        )
    return float(
        brentq(
            residual,
            0.0,
            RT,
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
        where=(
            1.0 + R / Kds + R * L / (Kd * Kd3)
        )
        != 0.0,
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
            ParameterSpec(
                name,
                min=STRICTLY_POSITIVE_PARAMETER_MIN,
                vary=False,
                kind="concentration",
                scale="log10",
                reportable=False,
            )
            for name in ("RT", "LsT", "Kds", "Kd3")
        ),
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
            ParameterSpec(
                name,
                min=STRICTLY_POSITIVE_PARAMETER_MIN,
                vary=False,
                kind="concentration",
                scale="log10",
                reportable=False,
            )
            for name in ("RT", "LsT", "Kds", "Kd3")
        ),
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
