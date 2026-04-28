from __future__ import annotations

import numpy as np
import pandas as pd

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.7, hill_slope=-1.2):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def ec50_curve(x, *, ymin=5.0, ymax=95.0, ec50=3.5, hill_slope=1.3):
    return ymin + (ymax - ymin) / (1.0 + (ec50 / x) ** hill_slope)


def logic50_curve(
    concentration,
    *,
    ymin=0.0,
    ymax=100.0,
    logic50=0.25,
    hill_slope=-1.15,
):
    log_concentration = np.log10(concentration)
    return ymin + (ymax - ymin) / (
        1.0 + 10 ** ((logic50 - log_concentration) * hill_slope)
    )


def make_data(curve, *, compound_id="cmpd_a") -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 16)
    rows = []
    for experiment_id, offset in {"exp1": -0.15, "exp2": 0.0, "exp3": 0.15}.items():
        for concentration in concentrations:
            response = curve(concentration) + offset
            for replicate_id, noise in enumerate([-0.2, 0.0, 0.2], start=1):
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


def test_model_registry_contains_logistic_family():
    assert isinstance(bc.get_model("ic50"), bc.IC50Model)
    assert isinstance(bc.get_model("ec50"), bc.EC50Model)
    assert isinstance(bc.get_model("logic50"), bc.LogIC50Model)


def test_ic50_model_fits_synthetic_inhibition_data():
    data = make_data(ic50_curve)
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert np.allclose(fits["IC50"].mean(), 1.7, rtol=0.10)
    assert np.allclose(fits["hill_slope"].mean(), -1.2, rtol=0.10)
    assert set(fits["IC50_unit"]) == {"uM"}


def test_ec50_model_fits_synthetic_activation_data():
    data = make_data(ec50_curve)
    results = bc.fit(data, model="ec50", fixed={"ymin": 5.0, "ymax": 95.0})
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert np.allclose(fits["EC50"].mean(), 3.5, rtol=0.10)
    assert np.allclose(fits["hill_slope"].mean(), 1.3, rtol=0.10)
    assert set(fits["EC50_unit"]) == {"uM"}


def test_logic50_model_fits_synthetic_log_concentration_data():
    data = make_data(logic50_curve)
    results = bc.fit(data, model="logic50", fixed={"ymin": 0.0, "ymax": 100.0})
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert np.allclose(fits["logIC50"].mean(), 0.25, atol=0.08)
    assert np.allclose(fits["hill_slope"].mean(), -1.15, rtol=0.12)
    assert set(fits["logIC50_unit"]) == {"log10(uM)"}


def test_logic50_summary_stays_on_linear_log_parameter_scale():
    data = make_data(logic50_curve)
    results = bc.fit(data, model="logic50", fixed={"ymin": 0.0, "ymax": 100.0})
    summary = results.summary_to_dataframe()
    logic50_summary = summary[summary["parameter"] == "logIC50"].iloc[0]

    assert logic50_summary["summary_scale"] == "linear"
    assert logic50_summary["geometric_mean"] is None
    assert np.isclose(logic50_summary["mean"], 0.25, atol=0.08)
