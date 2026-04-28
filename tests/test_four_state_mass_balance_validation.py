from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import bindcurve as bc


def solve_four_state_mass_balance(
    ligand_total: float,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> float:
    """Numerically solve the full four-state equilibrium mass balance.

    Species model:

    - R: free receptor
    - S: free labeled ligand/tracer
    - L: free unlabeled competitor
    - RS = R*S/Kds
    - RL = R*L/Kd
    - RLS = R*L*S/(Kd*Kd3)

    The returned value is FSB, the fraction of total labeled ligand bound in
    RS + RLS. This is an independent numerical validation target for the
    polynomial implementation.
    """
    scales = np.array([RT, LsT, ligand_total], dtype=float)
    lower = np.zeros(3, dtype=float)
    upper = np.maximum(scales, 1.0e-30)
    initial = np.maximum(scales * 0.5, 1.0e-30)

    def residuals(free_concentrations: np.ndarray) -> np.ndarray:
        receptor_free, tracer_free, competitor_free = free_concentrations
        receptor_tracer = receptor_free * tracer_free / Kds
        receptor_competitor = receptor_free * competitor_free / Kd
        ternary = receptor_free * competitor_free * tracer_free / (Kd * Kd3)

        raw = np.array(
            [
                receptor_free + receptor_tracer + receptor_competitor + ternary - RT,
                tracer_free + receptor_tracer + ternary - LsT,
                competitor_free + receptor_competitor + ternary - ligand_total,
            ],
            dtype=float,
        )
        return raw / np.maximum(scales, 1.0e-30)

    result = least_squares(
        residuals,
        x0=initial,
        bounds=(lower, upper),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=10_000,
    )

    assert result.success, result.message
    receptor_free, tracer_free, competitor_free = result.x
    receptor_tracer = receptor_free * tracer_free / Kds
    ternary = receptor_free * competitor_free * tracer_free / (Kd * Kd3)
    return float((receptor_tracer + ternary) / LsT)


def polynomial_four_state_fsb(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    response = bc.CompetitiveFourStateSpecificKdModel().evaluate(
        ligand_total,
        ymin=0.0,
        ymax=1.0,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    return np.asarray(response, dtype=float)


def test_specific_four_state_matches_numerical_mass_balance_typical_case():
    params = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 1.6,
        "Kd3": 0.5,
    }
    ligand_total = np.logspace(-3, 2, 12)

    polynomial = polynomial_four_state_fsb(ligand_total, **params)
    numerical = np.array(
        [solve_four_state_mass_balance(float(x), **params) for x in ligand_total]
    )

    assert np.allclose(polynomial, numerical, rtol=1.0e-7, atol=1.0e-9)


def test_specific_four_state_matches_numerical_mass_balance_cooperative_case():
    params = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 0.8,
        "Kd3": 0.001,
    }
    ligand_total = np.logspace(-4, 2, 12)

    polynomial = polynomial_four_state_fsb(ligand_total, **params)
    numerical = np.array(
        [solve_four_state_mass_balance(float(x), **params) for x in ligand_total]
    )

    assert np.allclose(polynomial, numerical, rtol=1.0e-7, atol=1.0e-9)


def test_specific_four_state_matches_numerical_mass_balance_anti_cooperative_case():
    params = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 0.8,
        "Kd3": 5.0,
    }
    ligand_total = np.logspace(-4, 2, 12)

    polynomial = polynomial_four_state_fsb(ligand_total, **params)
    numerical = np.array(
        [solve_four_state_mass_balance(float(x), **params) for x in ligand_total]
    )

    assert np.allclose(polynomial, numerical, rtol=1.0e-7, atol=1.0e-9)


def test_total_four_state_matches_numerical_mass_balance_with_effective_kd():
    params = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 2.4,
        "Kd3": 0.8,
        "N": 0.35,
    }
    ligand_total = np.logspace(-3, 2, 12)

    polynomial = bc.CompetitiveFourStateTotalKdModel().evaluate(
        ligand_total,
        ymin=0.0,
        ymax=1.0,
        **params,
    )
    numerical = np.array(
        [
            solve_four_state_mass_balance(
                float(x),
                RT=params["RT"],
                LsT=params["LsT"],
                Kds=params["Kds"],
                Kd=(1.0 + params["N"]) * params["Kd"],
                Kd3=params["Kd3"],
            )
            for x in ligand_total
        ]
    )

    assert np.allclose(polynomial, numerical, rtol=1.0e-7, atol=1.0e-9)
