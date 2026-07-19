from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc
from bindcurve.modeling import IC50Model


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.7, hill_slope=1.2):
    return ymin + (ymax - ymin) / (1.0 + (x / ic50) ** hill_slope)


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
    )


def test_model_registry_contains_ic50_model():
    assert isinstance(bc.get_model("ic50"), IC50Model)


def test_ic50_model_fits_synthetic_inhibition_data():
    data = make_data(ic50_curve)
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "amplitude": 100.0})
    fits = results.fit_summary()

    assert len(fits) == 3
    assert np.allclose(fits["IC50"].mean(), 1.7, rtol=0.10)
    assert np.allclose(fits["hill_slope"].mean(), 1.2, rtol=0.10)


def test_ec50_model_is_not_registered():
    with np.testing.assert_raises_regex(KeyError, "Unknown model 'ec50'"):
        bc.get_model("ec50")


def test_ic50_parameterization_rejects_the_old_orientation_symmetry():
    model = IC50Model()
    concentration = np.logspace(-3, 3, 20)
    valid = {
        "ymin": 0.0,
        "amplitude": 100.0,
        "IC50": 1.0,
        "hill_slope": 1.0,
    }

    with pytest.raises(ValueError, match="amplitude"):
        model.evaluate(concentration, **{**valid, "amplitude": -100.0})
    with pytest.raises(ValueError, match="hill_slope"):
        model.evaluate(concentration, **{**valid, "hill_slope": -1.0})


def test_ic50_evaluation_is_finite_and_monotone_at_extreme_concentrations():
    model = IC50Model()
    concentration = np.asarray([0.0, 1e-300, 1.0, 1e300])

    response = model.evaluate(
        concentration,
        ymin=5.0,
        amplitude=90.0,
        IC50=1.0,
        hill_slope=2.0,
    )

    assert np.all(np.isfinite(response))
    assert np.all(np.diff(response) <= 0.0)
    assert response[0] == pytest.approx(95.0)
    assert response[-1] == pytest.approx(5.0)

def test_ic50_summary_exposes_derived_log_face_in_parameters():
    data = make_data(ic50_curve)
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "amplitude": 100.0})
    summary = results.summary()
    parameters = results.parameters()

    assert len(summary) == 1
    assert summary.loc[0, "N_exp"] == 3
    assert np.isclose(summary.loc[0, "IC50"], 1.7, rtol=0.10)
    assert "logIC50" not in summary.columns
    assert (
        summary.loc[0, "IC50_SD_lower"]
        < summary.loc[0, "IC50"]
        < summary.loc[0, "IC50_SD_upper"]
    )
    ic50_parameters = parameters[parameters["parameter"] == "IC50"].iloc[0]
    assert ic50_parameters["summary_type"] == "concentration"
    assert ic50_parameters["log_parameter"] == "logIC50"
    assert np.isclose(ic50_parameters["center"], 1.7, rtol=0.10)
    assert np.isclose(ic50_parameters["log10_mean"], np.log10(1.7), atol=0.06)
