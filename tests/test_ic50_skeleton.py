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

    fits = results.fits()

    assert len(fits) == 6
    assert set(fits["compound_id"]) == {"cmpd_a", "cmpd_b"}
    assert set(fits["experiment_id"]) == {"exp1", "exp2", "exp3"}
    assert fits["success"].all()

    cmpd_a = fits[fits["compound_id"] == "cmpd_a"]
    cmpd_b = fits[fits["compound_id"] == "cmpd_b"]

    assert np.allclose(cmpd_a["IC50"].mean(), 1.5, rtol=0.15)
    assert np.allclose(cmpd_b["IC50"].mean(), 4.0, rtol=0.15)


def test_summary_reports_one_row_per_compound_with_ic50_triplets():
    data = make_synthetic_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})
    summary = results.summary()

    assert len(summary) == 2
    assert "compound_id" in summary.columns
    assert "N_exp" in summary.columns
    assert "N_obs" in summary.columns
    assert "n" not in summary.columns
    assert "IC50" in summary.columns
    assert "IC50_SD_lower" in summary.columns
    assert "IC50_SD_upper" in summary.columns
    assert "IC50_SEM_lower" in summary.columns
    assert "IC50_SEM_upper" in summary.columns
    assert "IC50_CI95_lower" in summary.columns
    assert "IC50_CI95_upper" in summary.columns
    assert "IC50_SD" not in summary.columns
    assert "IC50_SEM" not in summary.columns
    assert "R_squared" in summary.columns
    assert "Chi_squared" in summary.columns

    cmpd_a = summary[summary["compound_id"] == "cmpd_a"].iloc[0]
    cmpd_b = summary[summary["compound_id"] == "cmpd_b"].iloc[0]

    assert cmpd_a["N_exp"] == 3
    assert cmpd_b["N_exp"] == 3
    assert cmpd_a["N_obs"] == 36
    assert cmpd_b["N_obs"] == 36
    assert np.isclose(cmpd_a["IC50"], 1.5, rtol=0.15)
    assert np.isclose(cmpd_b["IC50"], 4.0, rtol=0.15)
    assert cmpd_a["IC50_SD_lower"] < cmpd_a["IC50"] < cmpd_a["IC50_SD_upper"]
    assert cmpd_a["IC50_SEM_lower"] < cmpd_a["IC50"] < cmpd_a["IC50_SEM_upper"]
    assert cmpd_a["IC50_CI95_lower"] < cmpd_a["IC50"] < cmpd_a["IC50_CI95_upper"]
    assert 0.0 <= cmpd_a["R_squared"] <= 1.0
    assert cmpd_a["Chi_squared"] >= 0.0


def test_parameters_keeps_detailed_parameter_rows():
    data = make_synthetic_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})
    parameters = results.parameters()

    assert len(parameters) == 8
    assert {"compound_id", "parameter", "N_exp", "summary_type"} <= set(
        parameters.columns
    )
    ic50_parameters = parameters[parameters["parameter"] == "IC50"]
    assert len(ic50_parameters) == 2
    assert ic50_parameters["N_exp"].eq(3).all()
    assert set(ic50_parameters["summary_type"]) == {"concentration"}
    assert ic50_parameters["center"].notna().all()
    assert ic50_parameters["log10_mean"].notna().all()
    assert ic50_parameters["log_parameter"].eq("logIC50").all()


def test_report_formats_geometric_center_and_intervals():
    data = make_synthetic_data()
    results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    report = results.report(
        uncertainty="sem",
        rounding="decimals",
        places_mean=2,
        places_uncertainty=3,
        unit="uM",
        include_n_exp=True,
    )

    assert list(report.columns) == ["compound_id", "report"]
    assert list(report["compound_id"]) == ["cmpd_a", "cmpd_b"]
    assert report.loc[0, "report"].endswith("uM, N_exp = 3")
    assert "[" in report.loc[0, "report"]
    assert "±" not in report.loc[0, "report"]


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


def test_collect_errors_continues_after_experiment_failure():
    class FailsForExp2Model(bc.IC50Model):
        name = "fails_for_exp2"

        def guess(self, compound: bc.CompoundData) -> dict[str, float]:
            experiment_id = str(compound.table["experiment_id"].iloc[0])
            if experiment_id == "exp2":
                raise RuntimeError("synthetic experiment failure")
            return super().guess(compound)

    data = make_synthetic_data().keep_only("cmpd_a")
    results = bc.fit(
        data,
        model=FailsForExp2Model(),
        settings=bc.FitSettings(errors="collect"),
        fixed={"ymin": 0.0, "ymax": 100.0},
    )

    successful_experiments = {fit.experiment_id for fit in results.successful()}
    failed_experiments = {fit.experiment_id for fit in results.failed()}

    assert successful_experiments == {"exp1", "exp3"}
    assert failed_experiments == {"exp2"}
    assert len(results.fit_results) == 3
