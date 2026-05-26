from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.8, hill_slope=-1.0):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_multi_experiment_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    for experiment_id, offset in {"exp1": -0.03, "exp2": 0.0, "exp3": 0.04}.items():
        effective_ic50 = 1.8 * (10**offset)
        for concentration in concentrations:
            base_response = ic50_curve(concentration, ic50=effective_ic50)
            for replicate_id, noise in enumerate([-0.3, 0.0, 0.3], start=1):
                rows.append(
                    {
                        "compound_id": "cmpd_a",
                        "experiment_id": experiment_id,
                        "concentration": concentration,
                        "replicate_id": f"rep{replicate_id}",
                        "response": base_response + noise,
                    }
                )
    return bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))


def make_single_experiment_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    for concentration in concentrations:
        base_response = ic50_curve(concentration, ic50=1.8)
        for replicate_id, noise in enumerate([-0.3, 0.0, 0.3], start=1):
            rows.append(
                {
                    "compound_id": "cmpd_a",
                    "experiment_id": "exp1",
                    "concentration": concentration,
                    "replicate_id": f"rep{replicate_id}",
                    "response": base_response + noise,
                }
            )
    return bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))


def test_report_representation_both_labels_linear_and_log_faces():
    data = make_multi_experiment_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    report = results.report(
        representation="both",
        uncertainty="sd",
        rounding="decimals",
        places_mean=2,
        places_uncertainty=2,
        unit="uM",
    )

    assert len(report) == 1
    text = report.loc[0, "report"]
    assert "IC50:" in text
    assert "logIC50:" in text
    assert "uM" in text
    assert "±" in text
    assert "[" in text


def test_report_omits_missing_uncertainty_for_single_experiment():
    data = make_single_experiment_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    summary = results.summary()
    report = results.report(rounding="decimals", places_mean=2, places_uncertainty=2)

    assert np.isnan(summary.loc[0, "IC50_SD_lower"])
    assert np.isnan(summary.loc[0, "IC50_CI95_lower"])
    assert "[" not in report.loc[0, "report"]
    assert "±" not in report.loc[0, "report"]


def test_fit_summary_omits_message_column():
    data = make_single_experiment_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    assert "message" not in results.fit_summary().columns


def test_report_auto_raises_for_multiple_reportable_quantities():
    fit = bc.FitResult(compound_id="cmpd_a", model_name="mock", success=True)
    summaries = [
        bc.ConcentrationSummary(
            compound_id="cmpd_a",
            parameter="IC50",
            log_parameter="logIC50",
            N_exp=3,
            reportable=True,
            log10_mean=0.1,
            log10_sd=0.02,
            log10_sem=0.01,
            log10_ci95_lower=0.05,
            log10_ci95_upper=0.15,
        ),
        bc.ConcentrationSummary(
            compound_id="cmpd_a",
            parameter="Kd",
            log_parameter="logKd",
            N_exp=3,
            reportable=True,
            log10_mean=0.2,
            log10_sd=0.03,
            log10_sem=0.02,
            log10_ci95_lower=0.12,
            log10_ci95_upper=0.28,
        ),
    ]
    results = bc.FitResults(fit_results=[fit], summaries=summaries)

    with pytest.raises(
        ValueError,
        match="Multiple reportable concentration quantities",
    ):
        results.report()
