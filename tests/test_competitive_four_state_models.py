from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

import bindcurve as bc
from bindcurve.modeling import (
    CompetitiveFourStateSpecificKdModel,
    CompetitiveFourStateTotalKdModel,
)
from bindcurve.modeling.four_state import (
    _competitive_four_state_coefficients,
    _select_physical_root,
)


def numerical_four_state_fbs(
    x,
    *,
    RT,
    LsT,
    Kds,
    Kd,
    Kd3,
    N=0.0,
):
    LT_values = np.asarray(x, dtype=float)
    Fbs_values = []
    for LT in np.atleast_1d(LT_values):
        scales = np.array([RT, LsT, LT], dtype=float)

        def residuals(
            free_concentrations,
            LT=LT,
            scales=scales,
        ):
            R, Ls, L = free_concentrations
            RLs = R * Ls / Kds
            RL = R * L / Kd
            RLLs = R * L * Ls / (Kd * Kd3)
            return np.array(
                [
                    R + RLs + RL + RLLs - RT,
                    Ls + RLs + RLLs - LsT,
                    (1.0 + N) * L + RL + RLLs - LT,
                ]
            ) / scales

        solution = least_squares(
            residuals,
            x0=scales / 2.0,
            bounds=(np.zeros(3), scales),
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=10_000,
        )
        assert solution.success
        assert np.max(np.abs(residuals(solution.x))) < 1.0e-8
        R, Ls, L = solution.x
        RLs = R * Ls / Kds
        RLLs = R * L * Ls / (Kd * Kd3)
        Fbs_values.append((RLs + RLLs) / LsT)
    Fbs_values = np.asarray(Fbs_values)
    return Fbs_values.item() if LT_values.ndim == 0 else Fbs_values


def comp_4st_specific_curve(
    x,
    *,
    ymin=5.0,
    ymax=95.0,
    RT=0.05,
    LsT=0.005,
    Kds=0.02,
    Kd=1.6,
    Kd3=0.5,
):
    Fbs = numerical_four_state_fbs(
        x,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )
    return ymin + (ymax - ymin) * Fbs


def comp_4st_total_curve(
    x,
    *,
    ymin=4.0,
    ymax=90.0,
    RT=0.05,
    LsT=0.005,
    Kds=0.02,
    Kd=2.4,
    Kd3=0.8,
    N=0.35,
):
    Fbs = numerical_four_state_fbs(
        x,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
        N=N,
    )
    return ymin + (ymax - ymin) * Fbs


def make_competition_data(curve, *, compound_id="cmpd_a") -> bc.DoseResponseData:
    concentrations = np.logspace(-3, 2, 22)
    rows = []
    multipliers = {"exp1": 0.95, "exp2": 1.00, "exp3": 1.05}
    for experiment_id, multiplier in multipliers.items():
        for concentration in concentrations:
            response = curve(concentration * multiplier)
            for replicate_id, noise in enumerate([-0.08, 0.0, 0.08], start=1):
                rows.append(
                    {
                        "compound_id": compound_id,
                        "experiment_id": experiment_id,
                        "concentration": concentration,
                        "replicate_id": f"rep{replicate_id}",
                        "response": response + noise,
                    }
                )
    return bc.DoseResponseData.from_dataframe(
        pd.DataFrame(rows),
    )


def test_registry_contains_competitive_four_state_models():
    assert isinstance(
        bc.get_model("comp_4st_specific"),
        CompetitiveFourStateSpecificKdModel,
    )
    assert isinstance(
        bc.get_model("comp_4st_total"),
        CompetitiveFourStateTotalKdModel,
    )


def test_four_state_root_selector_returns_physical_free_receptor_root():
    kwargs = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 1.6,
        "Kd3": 0.5,
    }
    coefficients = _competitive_four_state_coefficients(0.1, **kwargs)
    root = _select_physical_root(
        coefficients,
        lower_bound=0.0,
        upper_bound=kwargs["RT"],
    )

    assert 0.0 <= root <= kwargs["RT"]
    assert np.isclose(np.polyval(coefficients, root), 0.0, atol=1.0e-10)


def test_four_state_root_selector_handles_degenerate_quartic_case():
    kwargs = {
        "RT": 0.05,
        "LsT": 0.005,
        "Kds": 0.02,
        "Kd": 0.8,
        "Kd3": 0.02,
    }
    coefficients = _competitive_four_state_coefficients(0.1, **kwargs)

    assert coefficients[0] == 0.0

    root = _select_physical_root(
        coefficients,
        lower_bound=0.0,
        upper_bound=kwargs["RT"],
    )

    assert 0.0 <= root <= kwargs["RT"]
    assert np.isclose(np.polyval(coefficients, root), 0.0, atol=1.0e-10)


def test_four_state_root_selector_rejects_polynomial_with_no_physical_root():
    coefficients = np.array([1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="No physical four-state root"):
        _select_physical_root(coefficients, lower_bound=0.0, upper_bound=0.05)


def test_comp_4st_specific_recovers_kd_from_synthetic_data():
    data = make_competition_data(comp_4st_specific_curve)
    results = bc.fit(
        data,
        model="comp_4st_specific",
        fixed={
            "ymin": 5.0,
            "ymax": 95.0,
            "RT": 0.05,
            "LsT": 0.005,
            "Kds": 0.02,
            "Kd3": 0.5,
        },
    )
    fits = results.fit_summary()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 1.6, rtol=0.18)


def test_comp_4st_total_recovers_kd_from_synthetic_data():
    data = make_competition_data(comp_4st_total_curve)
    results = bc.fit(
        data,
        model="comp_4st_total",
        fixed={
            "ymin": 4.0,
            "ymax": 90.0,
            "RT": 0.05,
            "LsT": 0.005,
            "Kds": 0.02,
            "Kd3": 0.8,
            "N": 0.35,
        },
    )
    fits = results.fit_summary()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 2.4, rtol=0.18)


def test_comp_4st_total_with_zero_n_matches_specific_model():
    concentration = np.logspace(-3, 2, 22)
    specific = CompetitiveFourStateSpecificKdModel().evaluate(
        concentration,
        ymin=5.0,
        ymax=95.0,
        RT=0.05,
        LsT=0.005,
        Kds=0.02,
        Kd=1.6,
        Kd3=0.5,
    )
    total = CompetitiveFourStateTotalKdModel().evaluate(
        concentration,
        ymin=5.0,
        ymax=95.0,
        RT=0.05,
        LsT=0.005,
        Kds=0.02,
        Kd=1.6,
        Kd3=0.5,
        N=0.0,
    )

    assert np.allclose(total, specific)


def test_comp_4st_specific_requires_kd3():
    data = make_competition_data(comp_4st_specific_curve)

    with pytest.raises(ValueError, match="Kd3"):
        bc.fit(
            data,
            model="comp_4st_specific",
            fixed={
                "ymin": 5.0,
                "ymax": 95.0,
                "RT": 0.05,
                "LsT": 0.005,
                "Kds": 0.02,
            },
        )


def test_comp_4st_total_requires_n():
    data = make_competition_data(comp_4st_total_curve)

    with pytest.raises(ValueError, match="N"):
        bc.fit(
            data,
            model="comp_4st_total",
            fixed={
                "ymin": 4.0,
                "ymax": 90.0,
                "RT": 0.05,
                "LsT": 0.005,
                "Kds": 0.02,
                "Kd3": 0.8,
            },
        )
