from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ConcentrationParameterSpec,
    ParameterSpec,
)


def _competition_guess(compound: CompoundData) -> dict[str, float]:
    table = compound.aggregate_replicates()
    concentration = table["concentration"].to_numpy(dtype=float)
    response = table["response"].to_numpy(dtype=float)
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
    kd_guess = float(concentration[midpoint_index])
    return {"ymin": ymin, "ymax": ymax, "Kd": kd_guess}


def _competitive_four_state_coefficients(
    ligand_total: float,
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
    ``RS + RLS`` rather than from a transformed receptor-like coordinate.

    The total/nonspecific model should call this with an effective ``Kd`` of
    ``(1 + N) * Kd`` rather than maintaining a duplicated coefficient
    expression.
    """
    LT = float(ligand_total)

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
    imaginary_tolerance: float = 1.0e-7,
    interval_tolerance: float = 1.0e-8,
) -> float:
    """Select the physical four-state free-receptor root.

    The physical root must be effectively real and lie in the feasible interval
    for literal free receptor concentration. For the four-state receptor
    polynomial this interval is ``0 <= R <= RT``. Among feasible candidates, the
    root with the smallest scaled polynomial residual is selected.
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

    return min(
        candidates,
        key=lambda root: _scaled_polynomial_residual(coefficients, root),
    )


def _competitive_four_state_receptor_free(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    ligand_total = np.asarray(ligand_total, dtype=float)
    if RT == 0.0:
        return np.zeros_like(ligand_total, dtype=float)

    flat_ligand_total = ligand_total.ravel()
    receptor_free = []

    for concentration in flat_ligand_total:
        coefficients = _competitive_four_state_coefficients(
            float(concentration),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
            Kd3=Kd3,
        )
        receptor_free.append(
            _select_physical_root(
                coefficients,
                lower_bound=0.0,
                upper_bound=RT,
            )
        )

    return np.asarray(receptor_free, dtype=float).reshape(ligand_total.shape)


def _competitive_four_state_fraction_tracer_bound(
    receptor_free: np.ndarray,
    ligand_total: np.ndarray,
    *,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    """Return ``(RS + RLS) / LsT`` from true free receptor concentration.

    For fixed free receptor ``R``, the tracer and competitor mass balances can
    be reduced to a quadratic equation in free competitor concentration ``L``.
    The tracer-bound fraction is then obtained from the actual four-state
    species without explicitly constructing free tracer concentration.
    """
    receptor_free = np.asarray(receptor_free, dtype=float)
    ligand_total = np.asarray(ligand_total, dtype=float)

    a = 1.0 + receptor_free / Kds
    b = 1.0 + receptor_free / Kd
    c = receptor_free / (Kd * Kd3)

    quadratic_a = b * c
    quadratic_b = a * b + c * LsT - c * ligand_total
    quadratic_c = -a * ligand_total

    discriminant = np.maximum(
        quadratic_b**2 - 4.0 * quadratic_a * quadratic_c,
        0.0,
    )
    denominator = 2.0 * quadratic_a
    fallback = np.divide(
        ligand_total,
        b,
        out=np.zeros_like(ligand_total, dtype=float),
        where=b != 0.0,
    )
    ligand_free = np.array(fallback, copy=True, dtype=float)
    np.divide(
        -quadratic_b + np.sqrt(discriminant),
        denominator,
        out=ligand_free,
        where=np.abs(denominator) > np.finfo(float).tiny,
    )

    bound_tracer_ratio = receptor_free / Kds + c * ligand_free
    return bound_tracer_ratio / (1.0 + bound_tracer_ratio)


def _competitive_four_state_ligand_free(
    receptor_free: np.ndarray,
    ligand_total: np.ndarray,
    *,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    """Return free competitor concentration for the specific four-state model."""
    receptor_free = np.asarray(receptor_free, dtype=float)
    ligand_total = np.asarray(ligand_total, dtype=float)

    a = 1.0 + receptor_free / Kds
    b = 1.0 + receptor_free / Kd
    c = receptor_free / (Kd * Kd3)

    quadratic_a = b * c
    quadratic_b = a * b + c * LsT - c * ligand_total
    quadratic_c = -a * ligand_total

    discriminant = np.maximum(
        quadratic_b**2 - 4.0 * quadratic_a * quadratic_c,
        0.0,
    )
    denominator = 2.0 * quadratic_a
    fallback = np.divide(
        ligand_total,
        b,
        out=np.zeros_like(ligand_total, dtype=float),
        where=b != 0.0,
    )
    ligand_free = np.array(fallback, copy=True, dtype=float)
    np.divide(
        -quadratic_b + np.sqrt(discriminant),
        denominator,
        out=ligand_free,
        where=np.abs(denominator) > np.finfo(float).tiny,
    )
    return ligand_free


def _competitive_four_state_specific_component_arrays(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> dict[str, np.ndarray]:
    ligand_total = np.asarray(ligand_total, dtype=float)
    receptor_free = _competitive_four_state_receptor_free(
        ligand_total,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    ligand_free = _competitive_four_state_ligand_free(
        receptor_free,
        ligand_total,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    tracer_free = np.divide(
        LsT,
        1.0 + receptor_free / Kds + receptor_free * ligand_free / (Kd * Kd3),
        out=np.zeros_like(receptor_free, dtype=float),
        where=(
            1.0 + receptor_free / Kds + receptor_free * ligand_free / (Kd * Kd3)
        )
        != 0.0,
    )
    rs = np.divide(
        receptor_free * tracer_free,
        Kds,
        out=np.zeros_like(receptor_free, dtype=float),
        where=Kds != 0.0,
    )
    rl = np.divide(
        receptor_free * ligand_free,
        Kd,
        out=np.zeros_like(receptor_free, dtype=float),
        where=Kd != 0.0,
    )
    rls = np.divide(
        receptor_free * ligand_free * tracer_free,
        Kd * Kd3,
        out=np.zeros_like(receptor_free, dtype=float),
        where=(Kd * Kd3) != 0.0,
    )
    fraction_tracer_bound = np.divide(
        rs + rls,
        LsT,
        out=np.zeros_like(receptor_free, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "L_total": ligand_total,
        "R_total": np.full_like(ligand_total, RT, dtype=float),
        "R_free": receptor_free,
        "L_free": ligand_free,
        "Lstar_total": np.full_like(ligand_total, LsT, dtype=float),
        "Lstar_free": tracer_free,
        "RS": rs,
        "RL": rl,
        "RLS": rls,
        "fraction_tracer_bound": fraction_tracer_bound,
    }


def _competitive_four_state_total_component_arrays(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
    N: float,
) -> dict[str, np.ndarray]:
    ligand_total = np.asarray(ligand_total, dtype=float)
    effective_kd = (1.0 + N) * Kd
    effective_components = _competitive_four_state_specific_component_arrays(
        ligand_total,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=effective_kd,
        Kd3=Kd3,
    )
    competitor_bound_total = np.clip(
        ligand_total - effective_components["L_free"],
        0.0,
        None,
    )
    competitor_bound_specific_equivalent = np.divide(
        competitor_bound_total,
        1.0 + N,
        out=np.zeros_like(competitor_bound_total, dtype=float),
        where=(1.0 + N) != 0.0,
    )
    competitor_bound_nonspecific = (
        competitor_bound_total - competitor_bound_specific_equivalent
    )
    tracer_bound_total = (
        effective_components["RS"] + effective_components["RLS"]
    )
    return {
        "L_total": ligand_total,
        "R_total": np.full_like(ligand_total, RT, dtype=float),
        "R_free": effective_components["R_free"],
        "L_free": effective_components["L_free"],
        "Lstar_total": np.full_like(ligand_total, LsT, dtype=float),
        "Lstar_free": effective_components["Lstar_free"],
        "RS_plus_RLS": tracer_bound_total,
        "L_bound_total": competitor_bound_total,
        "L_bound_specific_equivalent": competitor_bound_specific_equivalent,
        "L_nonspecific_bound": competitor_bound_nonspecific,
        "fraction_tracer_bound": effective_components["fraction_tracer_bound"],
    }


class CompetitiveFourStateSpecificKdModel(BaseDoseResponseModel):
    """Four-state competitive-binding model for specific binding."""

    name = "comp_4st_specific"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "Kd3"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd", "Kd3"})
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
            parameter="Kd3",
            fitted_parameter="Kd3",
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
        ParameterSpec("Kd3", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
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
        Kd3: float,
        Kd: float,
    ) -> np.ndarray:
        components = _competitive_four_state_specific_component_arrays(
            np.asarray(x, dtype=float),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
            Kd3=Kd3,
        )
        fraction_tracer_bound = components["fraction_tracer_bound"]
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
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
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "Kd3", "N"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd", "Kd3"})
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
            parameter="Kd3",
            fitted_parameter="Kd3",
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
        ParameterSpec("Kd3", min=STRICTLY_POSITIVE_PARAMETER_MIN, vary=False),
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
        Kd3: float,
        N: float,
        Kd: float,
    ) -> np.ndarray:
        components = _competitive_four_state_total_component_arrays(
            np.asarray(x, dtype=float),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
            Kd3=Kd3,
            N=N,
        )
        fraction_tracer_bound = components["fraction_tracer_bound"]
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
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
