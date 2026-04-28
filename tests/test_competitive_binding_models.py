from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def receptor_free_three_state(ligand_total, *, RT, LsT, Kds, Kd, factor=1.0):
    ligand_total = np.asarray(ligand_total, dtype=float)
    scaled_kd = factor * Kd
    a = Kds + scaled_kd + LsT + ligand_total - RT
    b = Kds * (ligand_total - RT) + scaled_kd * (LsT - RT) + Kds * scaled_kd
    c = -Kds * scaled_kd * RT
    discriminant = np.maximum(a**2 - 3.0 * b, 0.0)
    denominator = 2.0 * np.sqrt(discriminant**3)
    numerator = -2.0 * a**3 + 9.0 * a * b - 27.0 * c
    argument = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )
    theta = np.arccos(np.clip(argument, -1.0, 1.0))
    return -(a / 3.0) + (2.0 / 3.0) * np.sqrt(discriminant) * np.cos(theta / 3.0)


def comp_specific_curve(x, *, ymin=5.0, ymax=95.0, RT=0.05, LsT=0.005, Kds=0.02, Kd=1.6):
    receptor_free = receptor_free_three_state(x, RT=RT, LsT=LsT, Kds=Kds, Kd=Kd)
    fraction_tracer_bound = receptor_free / (Kds + receptor_free)
    return ymin + (ymax - ymin) * fraction_tracer_bound


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
    receptor_free = receptor_free_three_state(
        x,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        factor=1.0 + N,
    )
    fraction_tracer_bound = receptor_free / (Kds + receptor_free)
    return ymin + (ymax - ymin) * fraction_tracer_bound


def make_competition_data(curve, *, compound_id="cmpd_a") -> bc.DoseResponseData:
    concentrations = np.logspace(-3, 2, 22)
    rows = []
    for experiment_id, multiplier in {"exp1": 0.95, "exp2": 1.00, "exp3": 1.05}.items():
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


def test_registry_contains_competitive_three_state_models():
    assert isinstance(
        bc.get_model("comp_3st_specific"),
        bc.CompetitiveThreeStateSpecificKdModel,
    )
    assert isinstance(
        bc.get_model("comp_3st_total"),
        bc.CompetitiveThreeStateTotalKdModel,
    )


def test_comp_3st_specific_recovers_kd_from_synthetic_data():
    data = make_competition_data(comp_specific_curve)
    results = bc.fit(
        data,
        model="comp_3st_specific",
        fixed={"ymin": 5.0, "ymax": 95.0, "RT": 0.05, "LsT": 0.005, "Kds": 0.02},
    )
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 1.6, rtol=0.15)
    assert set(fits["Kd_unit"]) == {"uM"}
    assert set(fits["RT_unit"]) == {"uM"}
    assert set(fits["LsT_unit"]) == {"uM"}
    assert set(fits["Kds_unit"]) == {"uM"}


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
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kd"].mean(), 2.4, rtol=0.15)
    assert set(fits["Kd_unit"]) == {"uM"}
    assert set(fits["N_unit"]) == {None}


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
            fixed={"ymin": 4.0, "ymax": 90.0, "RT": 0.05, "LsT": 0.005, "Kds": 0.02},
        )
