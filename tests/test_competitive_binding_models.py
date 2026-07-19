from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

import bindcurve as bc
from bindcurve.modeling import (
    CompetitiveThreeStateSpecificKdModel,
    CompetitiveThreeStateTotalKdModel,
)


def receptor_free_three_state(LT, *, RT, LsT, Kds, Kd, factor=1.0):
    LT = np.asarray(LT, dtype=float)
    roots = []
    for total in np.atleast_1d(LT):
        roots.append(
            brentq(
                lambda free, total=total: free
                + LsT * free / (Kds + free)
                + float(total) * free / (factor * Kd + free)
                - RT,
                0.0,
                RT,
            )
        )
    roots = np.asarray(roots)
    return roots.item() if LT.ndim == 0 else roots


def comp_specific_curve(
    x,
    *,
    ymin=5.0,
    ymax=95.0,
    RT=0.05,
    LsT=0.005,
    Kds=0.02,
    Kd=1.6,
):
    R = receptor_free_three_state(x, RT=RT, LsT=LsT, Kds=Kds, Kd=Kd)
    Fbs = R / (Kds + R)
    return ymin + (ymax - ymin) * Fbs


def comp_total_curve(
    x,
    *,
    ymin=4.0,
    ymax=90.0,
    RT=0.05,
    LsT=0.005,
    Kds=0.02,
    Kd=2.4,
    N=0.35,
):
    R = receptor_free_three_state(
        x,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        factor=1.0 + N,
    )
    Fbs = R / (Kds + R)
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


def test_registry_contains_competitive_three_state_models():
    assert isinstance(
        bc.get_model("comp_3st_specific"),
        CompetitiveThreeStateSpecificKdModel,
    )
    assert isinstance(
        bc.get_model("comp_3st_total"),
        CompetitiveThreeStateTotalKdModel,
    )


def test_comp_3st_specific_recovers_kd_from_synthetic_data():
    data = make_competition_data(comp_specific_curve)
    results = bc.fit(
        data,
        model="comp_3st_specific",
        fixed={
            "ymin": 5.0,
            "ymax": 95.0,
            "RT": 0.05,
            "LsT": 0.005,
            "Kds": 0.02,
        },
    )
    fits = results.fit_summary()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 1.6, rtol=0.15)


def test_comp_3st_total_recovers_kd_from_synthetic_data():
    data = make_competition_data(comp_total_curve)
    results = bc.fit(
        data,
        model="comp_3st_total",
        fixed={
            "ymin": 4.0,
            "ymax": 90.0,
            "RT": 0.05,
            "LsT": 0.005,
            "Kds": 0.02,
            "N": 0.35,
        },
    )
    fits = results.fit_summary()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 2.4, rtol=0.15)


def test_comp_3st_specific_requires_constants():
    data = make_competition_data(comp_specific_curve)

    with pytest.raises(ValueError, match="RT"):
        bc.fit(
            data,
            model="comp_3st_specific",
            fixed={"ymin": 5.0, "ymax": 95.0, "LsT": 0.005, "Kds": 0.02},
        )


def test_comp_3st_total_requires_nonspecific_constant():
    data = make_competition_data(comp_total_curve)

    with pytest.raises(ValueError, match="N"):
        bc.fit(
            data,
            model="comp_3st_total",
            fixed={
                "ymin": 4.0,
                "ymax": 90.0,
                "RT": 0.05,
                "LsT": 0.005,
                "Kds": 0.02,
            },
        )
