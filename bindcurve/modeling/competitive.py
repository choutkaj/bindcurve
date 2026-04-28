from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.parameters import ParameterSpec


def _competition_guess(compound: CompoundData) -> dict[str, float]:
    table = compound.aggregate_replicates(method="mean")
    concentration = table["concentration"].to_numpy(dtype=float)
    response = table["response"].to_numpy(dtype=float)
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
    kd_guess = float(concentration[midpoint_index])
    return {"ymin": ymin, "ymax": ymax, "Kd": kd_guess}


def _direct_bound_fraction(*, Kd: float, RT: float, LsT: float) -> float:
    """Return the direct-binding bound tracer fraction for finite totals."""
    discriminant = (Kd + LsT + RT) ** 2 - 4.0 * LsT * RT
    return float((Kd + LsT + RT - np.sqrt(discriminant)) / (2.0 * LsT))


def _equivalent_receptor_from_bound_fraction(
    fraction_bound: float,
    *,
    Kds: float,
) -> float:
    """Transform FSB to the legacy four-state polynomial coordinate."""
    return float(Kds * fraction_bound / (1.0 - fraction_bound))


def _equivalent_receptor_bounds(
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd3: float,
) -> tuple[float, float]:
    """Return the physically feasible range for the transformed coordinate.

    The four-state polynomial is expressed in the coordinate
    ``r_equiv = Kds * FSB / (1 - FSB)``. This is equal to free receptor only in
    three-state/complete competition. In incomplete competition, the physical
    interval is determined by the initial direct-binding fraction, governed by
    ``Kds``, and the asymptotic bound fraction, governed by ``Kd3``.
    """
    initial_fraction = _direct_bound_fraction(Kd=Kds, RT=RT, LsT=LsT)
    asymptotic_fraction = _direct_bound_fraction(Kd=Kd3, RT=RT, LsT=LsT)
    initial_coordinate = _equivalent_receptor_from_bound_fraction(
        initial_fraction,
        Kds=Kds,
    )
    asymptotic_coordinate = _equivalent_receptor_from_bound_fraction(
        asymptotic_fraction,
        Kds=Kds,
    )
    return (
        min(initial_coordinate, asymptotic_coordinate),
        max(initial_coordinate, asymptotic_coordinate),
    )


def _competitive_four_state_coefficients(
    ligand_total: float,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    """Return quintic coefficients for the four-state competitive model.

    The polynomial variable is not literal free receptor in the four-state
    model. It is the transformed coordinate ``r_equiv = Kds * FSB / (1 - FSB)``.
    This coordinate lets the model recover ``FSB`` as
    ``FSB = r_equiv / (Kds + r_equiv)`` after solving the quintic.

    The total/nonspecific model should call this with an effective ``Kd`` of
    ``(1 + N) * Kd`` rather than maintaining a duplicated coefficient
    expression.
    """
    LT = float(ligand_total)

    a = -Kds**2 * Kd3**2
    b = Kds**2 * Kd3 * (Kds * Kd - 3 * Kds * Kd3 + Kds * LT - 2 * Kds * LsT + Kds * RT - Kd * Kd3 - Kd3 * LT - Kd3 * LsT + Kd3 * RT)  # noqa: E501
    c = Kds**2 * (3 * Kds**2 * Kd * Kd3 - 3 * Kds**2 * Kd3**2 + 3 * Kds**2 * Kd3 * LT - 4 * Kds**2 * Kd3 * LsT + 3 * Kds**2 * Kd3 * RT + Kds**2 * LT * LsT - Kds**2 * LT * RT - Kds**2 * LsT**2 + Kds**2 * LsT * RT - 3 * Kds * Kd * Kd3**2 + Kds * Kd * Kd3 * LsT - Kds * Kd * Kd3 * RT - 3 * Kds * Kd3**2 * LT - 2 * Kds * Kd3**2 * LsT + 3 * Kds * Kd3**2 * RT - Kds * Kd3 * LT * LsT + Kds * Kd3 * LT * RT - 2 * Kds * Kd3 * LsT**2 + 3 * Kds * Kd3 * LsT * RT - Kds * Kd3 * RT**2 - Kd * Kd3**2 * LsT + Kd * Kd3**2 * RT)  # noqa: E501
    d = Kds**3 * (3 * Kds**2 * Kd * Kd3 - Kds**2 * Kd3**2 + 3 * Kds**2 * Kd3 * LT - 2 * Kds**2 * Kd3 * LsT + 3 * Kds**2 * Kd3 * RT + 2 * Kds**2 * LT * LsT - 3 * Kds**2 * LT * RT - Kds**2 * LsT**2 + 2 * Kds**2 * LsT * RT - 3 * Kds * Kd * Kd3**2 + 2 * Kds * Kd * Kd3 * LsT - 3 * Kds * Kd * Kd3 * RT - 3 * Kds * Kd3**2 * LT - Kds * Kd3**2 * LsT + 3 * Kds * Kd3**2 * RT - 2 * Kds * Kd3 * LT * LsT + 3 * Kds * Kd3 * LT * RT - 2 * Kds * Kd3 * LsT**2 + 6 * Kds * Kd3 * LsT * RT - 3 * Kds * Kd3 * RT**2 - Kds * LsT**3 + 2 * Kds * LsT**2 * RT - Kds * LsT * RT**2 - 2 * Kd * Kd3**2 * LsT + 3 * Kd * Kd3**2 * RT)  # noqa: E501
    e = Kds**4 * (Kds**2 * Kd * Kd3 + Kds**2 * Kd3 * LT + Kds**2 * Kd3 * RT + Kds**2 * LT * LsT - 3 * Kds**2 * LT * RT + Kds**2 * LsT * RT - Kds * Kd * Kd3**2 + Kds * Kd * Kd3 * LsT - 3 * Kds * Kd * Kd3 * RT - Kds * Kd3**2 * LT + Kds * Kd3**2 * RT - Kds * Kd3 * LT * LsT + 3 * Kds * Kd3 * LT * RT + 3 * Kds * Kd3 * LsT * RT - 3 * Kds * Kd3 * RT**2 + 2 * Kds * LsT**2 * RT - 2 * Kds * LsT * RT**2 - Kd * Kd3**2 * LsT + 3 * Kd * Kd3**2 * RT)  # noqa: E501
    f = Kds**5 * RT * (-Kds**2 * LT - Kds * Kd * Kd3 + Kds * Kd3 * LT - Kds * Kd3 * RT - Kds * LsT * RT + Kd * Kd3**2)  # noqa: E501

    return np.array([a, b, c, d, e, f], dtype=float)


def _scaled_polynomial_residual(coefficients: np.ndarray, root: float) -> float:
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
    """Select the physical four-state transformed-coordinate root.

    The physical root must be effectively real and lie in the feasible interval
    for ``r_equiv = Kds * FSB / (1 - FSB)``. This interval is not generally
    ``0 <= r_equiv <= RT``. Among feasible candidates, the root with the
    smallest scaled polynomial residual is selected.
    """
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
            "No physical four-state root found in the feasible transformed "
            f"coordinate interval {lower_bound} <= r_equiv <= {upper_bound}. "
            f"Roots were: {roots!r}"
        )

    return min(
        candidates,
        key=lambda root: _scaled_polynomial_residual(coefficients, root),
    )


def _competitive_four_state_equivalent_receptor(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    ligand_total = np.asarray(ligand_total, dtype=float)
    flat_ligand_total = ligand_total.ravel()
    lower_bound, upper_bound = _equivalent_receptor_bounds(
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd3=Kd3,
    )
    equivalent_receptor = []

    for concentration in flat_ligand_total:
        coefficients = _competitive_four_state_coefficients(
            float(concentration),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
            Kd3=Kd3,
        )
        equivalent_receptor.append(
            _select_physical_root(
                coefficients,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )

    return np.asarray(equivalent_receptor, dtype=float).reshape(ligand_total.shape)


class CompetitiveFourStateSpecificKdModel(BaseDoseResponseModel):
    """Four-state competitive-binding model for specific binding."""

    name = "comp_4st_specific"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "Kd3"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd", "Kd3"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("RT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("LsT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kds", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kd3", min=0.0, vary=False, unit_kind="concentration"),
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
        Kd3: float,
        Kd: float,
    ) -> np.ndarray:
        equivalent_receptor = _competitive_four_state_equivalent_receptor(
            x,
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
            Kd3=Kd3,
        )
        fraction_tracer_bound = equivalent_receptor / (Kds + equivalent_receptor)
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)


class CompetitiveFourStateTotalKdModel(BaseDoseResponseModel):
    """Four-state competitive-binding model with nonspecific binding."""

    name = "comp_4st_total"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "Kd3", "N"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd", "Kd3"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("RT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("LsT", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kds", min=0.0, vary=False, unit_kind="concentration"),
        ParameterSpec("Kd3", min=0.0, vary=False, unit_kind="concentration"),
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
        Kd3: float,
        N: float,
        Kd: float,
    ) -> np.ndarray:
        equivalent_receptor = _competitive_four_state_equivalent_receptor(
            x,
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=(1.0 + N) * Kd,
            Kd3=Kd3,
        )
        fraction_tracer_bound = equivalent_receptor / (Kds + equivalent_receptor)
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)
