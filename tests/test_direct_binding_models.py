from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def dir_simple_curve(x, *, ymin=2.0, ymax=88.0, Kds=1.4):
    receptor = np.asarray(x, dtype=float)
    fraction_bound = receptor / (Kds + receptor)
    return ymin + (ymax - ymin) * fraction_bound


def dir_specific_curve(x, *, ymin=3.0, ymax=91.0, LsT=0.35, Kds=1.8):
    receptor_total = np.asarray(x, dtype=float)
    a = Kds + LsT - receptor_total
    b = -Kds * receptor_total
    receptor_free = (-a + np.sqrt(a**2 - 4.0 * b)) / 2.0
    fraction_bound = receptor_free / (Kds + receptor_free)
    return ymin + (ymax - ymin) * fraction_bound


def dir_total_curve(x, *, ymin=4.0, ymax=86.0, LsT=0.4, Ns=0.25, Kds=2.2):
    receptor_total = np.asarray(x, dtype=float)
    a = (1.0 + Ns) * Kds + LsT - receptor_total
    b = -Kds * receptor_total * (1.0 + Ns)
    receptor_free = (-a + np.sqrt(a**2 - 4.0 * b)) / 2.0
    fraction_bound = receptor_free / (Kds + receptor_free)
    return ymin + (ymax - ymin) * fraction_bound


def make_saturation_data(curve, *, compound_id="cmpd_a") -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 18)
    rows = []
    for experiment_id, multiplier in {"exp1": 0.96, "exp2": 1.00, "exp3": 1.04}.items():
        for concentration in concentrations:
            response = curve(concentration * multiplier)
            for replicate_id, noise in enumerate([-0.15, 0.0, 0.15], start=1):
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


def test_registry_contains_direct_binding_models():
    assert isinstance(bc.get_model("dir_simple"), bc.DirectSimpleKdModel)
    assert isinstance(bc.get_model("dir_specific"), bc.DirectSpecificKdModel)
    assert isinstance(bc.get_model("dir_total"), bc.DirectTotalKdModel)


def test_dir_simple_recovers_kds_from_synthetic_data():
    data = make_saturation_data(dir_simple_curve)
    results = bc.fit(
        data,
        model="dir_simple",
        fixed={"ymin": 2.0, "ymax": 88.0},
    )
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kds"].mean(), 1.4, rtol=0.12)
    assert set(fits["Kds_unit"]) == {"uM"}


def test_dir_specific_recovers_kds_from_synthetic_data():
    data = make_saturation_data(dir_specific_curve)
    results = bc.fit(
        data,
        model="dir_specific",
        fixed={"ymin": 3.0, "ymax": 91.0, "LsT": 0.35},
    )
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kds"].mean(), 1.8, rtol=0.12)
    assert set(fits["Kds_unit"]) == {"uM"}
    assert set(fits["LsT_unit"]) == {"uM"}
    assert not fits["LsT"].isna().any()


def test_dir_total_recovers_kds_from_synthetic_data():
    data = make_saturation_data(dir_total_curve)
    results = bc.fit(
        data,
        model="dir_total",
        fixed={"ymin": 4.0, "ymax": 86.0, "LsT": 0.4, "Ns": 0.25},
    )
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert fits["success"].all()
    assert np.allclose(fits["Kds"].mean(), 2.2, rtol=0.12)
    assert set(fits["Kds_unit"]) == {"uM"}
    assert set(fits["LsT_unit"]) == {"uM"}
    assert set(fits["Ns_unit"]) == {None}


def test_missing_required_direct_binding_constant_raises():
    data = make_saturation_data(dir_specific_curve)

    with pytest.raises(ValueError, match="LsT"):
        bc.fit(
            data,
            model="dir_specific",
            fixed={"ymin": 3.0, "ymax": 91.0},
        )


def test_dir_total_requires_all_fixed_constants():
    data = make_saturation_data(dir_total_curve)

    with pytest.raises(ValueError, match="Ns"):
        bc.fit(
            data,
            model="dir_total",
            fixed={"ymin": 4.0, "ymax": 86.0, "LsT": 0.4},
        )
