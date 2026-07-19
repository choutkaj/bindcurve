from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

import bindcurve as bc


def solve_four_state_mass_balance(
    LT: float,
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
    - Ls: free labeled ligand/tracer
    - L: free unlabeled competitor
    - RLs = R*Ls/Kds
    - RL = R*L/Kd
    - RLLs = R*L*Ls/(Kd*Kd3)

    The returned value is Fbs, the fraction of total labeled ligand bound in
    RLs + RLLs. This is an independent numerical validation target for the
    polynomial implementation.
    """
    scales = np.array([RT, LsT, LT], dtype=float)
    lower = np.zeros(3, dtype=float)
    upper = np.maximum(scales, 1.0e-30)
    initial = np.maximum(scales * 0.5, 1.0e-30)

    def residuals(free_concentrations: np.ndarray) -> np.ndarray:
        R, Ls, L = free_concentrations
        RLs = R * Ls / Kds
        RL = R * L / Kd
        RLLs = R * L * Ls / (Kd * Kd3)

        raw = np.array(
            [
                R + RLs + RL + RLLs - RT,
                Ls + RLs + RLLs - LsT,
                L + RL + RLLs - LT,
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
    R, Ls, L = result.x
    RLs = R * Ls / Kds
    RLLs = R * L * Ls / (Kd * Kd3)
    return float((RLs + RLLs) / LsT)


def polynomial_four_state_fbs(
    LT: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    Kd3: float,
) -> np.ndarray:
    response = bc.CompetitiveFourStateSpecificKdModel().evaluate(
        LT,
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
    LT = np.logspace(-3, 2, 12)

    polynomial = polynomial_four_state_fbs(LT, **params)
    numerical = np.array(
        [solve_four_state_mass_balance(float(x), **params) for x in LT]
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
    LT = np.logspace(-4, 2, 12)

    polynomial = polynomial_four_state_fbs(LT, **params)
    numerical = np.array(
        [solve_four_state_mass_balance(float(x), **params) for x in LT]
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
    LT = np.logspace(-4, 2, 12)

    polynomial = polynomial_four_state_fbs(LT, **params)
    numerical = np.array(
        [solve_four_state_mass_balance(float(x), **params) for x in LT]
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
    LT = np.logspace(-3, 2, 12)

    polynomial = bc.CompetitiveFourStateTotalKdModel().evaluate(
        LT,
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
            for x in LT
        ]
    )

    assert np.allclose(polynomial, numerical, rtol=1.0e-7, atol=1.0e-9)


def test_specific_four_state_rejects_extraneous_interval_root_by_mass_balance():
    params = {
        "RT": 1.1230881504118417,
        "LsT": 0.16772085032232287,
        "Kds": 0.0959937419194935,
        "Kd": 0.013895014894051528,
        "Kd3": 0.5947156861737775,
    }
    LT = 0.2819073703806677

    polynomial = polynomial_four_state_fbs(np.array([LT]), **params)[0]
    numerical = solve_four_state_mass_balance(LT, **params)

    assert polynomial == pytest.approx(numerical, rel=1.0e-7, abs=1.0e-9)


def test_specific_four_state_is_invariant_to_concentration_units():
    params = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 1.6,
        "Kd3": 0.5,
    }
    LT = np.logspace(-3, 2, 12)
    reference = polynomial_four_state_fbs(LT, **params)

    scale = 1.0e9
    scaled_params = {name: value * scale for name, value in params.items()}
    scaled = polynomial_four_state_fbs(
        LT * scale,
        **scaled_params,
    )

    assert np.allclose(scaled, reference, rtol=1.0e-10, atol=1.0e-12)


def test_specific_four_state_preserves_tiny_physical_root_in_tracer_excess():
    model = bc.CompetitiveFourStateSpecificKdModel()
    components = model.component_arrays(
        np.array([1.0]),
        np.array([1.0]),
        RT=1.0,
        LsT=1.0e15,
        Kds=1.0,
        Kd=1.0,
        Kd3=1.0,
    )

    assert components["R"][0] > 0.0
    RT_reconstructed = sum(
        components[name][0] for name in ("R", "RLs", "RL", "RLLs")
    )
    assert RT_reconstructed == pytest.approx(1.0, rel=1.0e-12)
