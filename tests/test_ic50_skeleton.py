from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.5, hill_slope=-1.1):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_synthetic_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    offsets = {"exp1": 0.0, "exp2": 0.03, "exp3": -0.02}

    for compound_id, ic50 in [("cmpd_a", 1.5), ("cmpd_b", 4.0)]:
        for experiment_id, offset in offsets.items():
            effective_ic50 = ic50 * (10**offset)
            for concentration in concentrations:
                base_response = ic50_curve(concentration, ic50=effective_ic50)
                for replicate_id, noise in enumerate([-0.5, 0.0, 0.5], start=1):
                    rows.append(
                        {
                            "compound_id": compound_id,
                            "experiment_id": experiment_id,
                            "concentration": concentration,
                            "replicate_id": f"rep{replicate_id}",
                            "response": base_response + noise,
                        }
                    )

    return bc.DoseResponseData.from_dataframe(
        pd.DataFrame(rows),
        concentration_unit="uM",
        response_unit="percent",
    )


def test_dose_response_data_validates_positive_concentrations():
    df = pd.DataFrame(
        {
            "compound_id": ["cmpd_a"],
            "concentration": [0.0],
            "response": [50.0],
        }
    )

    with pytest.raises(ValueError, match="positive"):
        bc.DoseResponseData.from_dataframe(df)


def test_default_strategy_fits_one_curve_per_independent_experiment():
    data = make_synthetic_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    fits = results.fits_to_dataframe()

    assert len(fits) == 6
    assert set(fits["compound_id"]) == {"cmpd_a", "cmpd_b"}
    assert set(fits["experiment_id"]) == {"exp1", "exp2", "exp3"}
    assert fits["success"].all()

    cmpd_a = fits[fits["compound_id"] == "cmpd_a"]
    cmpd_b = fits[fits["compound_id"] == "cmpd_b"]

    assert np.allclose(cmpd_a["IC50"].mean(), 1.5, rtol=0.15)
    assert np.allclose(cmpd_b["IC50"].mean(), 4.0, rtol=0.15)
    assert set(cmpd_a["IC50_unit"]) == {"uM"}
    assert set(cmpd_a["ymin_unit"]) == {"percent"}


def test_summary_uses_log_scale_for_ic50():
    data = make_synthetic_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})
    summary = results.summary_to_dataframe()

    ic50_summary = summary[summary["parameter"] == "IC50"]

    assert len(ic50_summary) == 2
    assert set(ic50_summary["summary_scale"]) == {"log10"}
    assert ic50_summary["geometric_mean"].notna().all()


def test_collect_errors_returns_failed_result():
    data = make_synthetic_data()
    settings = bc.FitSettings(errors="collect")
    results = bc.fit(
        data,
        model="ic50",
        settings=settings,
        compounds=["missing_compound"],
    )

    assert len(results.failed()) == 1
    assert results.failed()[0].success is False
