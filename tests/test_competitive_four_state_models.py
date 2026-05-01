from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc
from bindcurve.modeling.competitive import (
    _competitive_four_state_coefficients,
    _select_physical_root,
)


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
    model = bc.CompetitiveFourStateSpecificKdModel()
    return model.evaluate(
        np.asarray(x, dtype=float),
        ymin=ymin,
        ymax=ymax,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
    )


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
    model = bc.CompetitiveFourStateTotalKdModel()
    return model.evaluate(
        np.asarray(x, dtype=float),
        ymin=ymin,
        ymax=ymax,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
        N=N,
    )


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
        concentration_unit="uM",
        response_unit="percent",
    )


def test_registry_contains_competitive_four_state_models():
    assert isinstance(
        bc.get_model("comp_4st_specific"),
        bc.CompetitiveFourStateSpecificKdModel,
    )
    assert isinstance(
        bc.get_model("comp_4st_total"),
        bc.CompetitiveFourStateTotalKdModel,
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
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 1.6, rtol=0.18)
    assert set(fits["Kd_unit"]) == {"uM"}
    assert set(fits["Kd3_unit"]) == {"uM"}


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
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 2.4, rtol=0.18)
    assert set(fits["Kd_unit"]) == {"uM"}
    assert set(fits["N_unit"]) == {None}


def test_comp_4st_total_with_zero_n_matches_specific_model():
    concentration = np.logspace(-3, 2, 22)
    specific = bc.CompetitiveFourStateSpecificKdModel().evaluate(
        concentration,
        ymin=5.0,
        ymax=95.0,
        RT=0.05,
        LsT=0.005,
        Kds=0.02,
        Kd=1.6,
        Kd3=0.5,
    )
    total = bc.CompetitiveFourStateTotalKdModel().evaluate(
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
