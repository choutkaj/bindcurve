from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.8, hill_slope=-1.0):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_quality_data(
    *,
    N_exp: int = 3,
    replicate_noise: float = 0.3,
    missing_cells: set[tuple[str, float]] | None = None,
    single_replicate_cells: set[tuple[str, float]] | None = None,
) -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 8)
    rows = []
    experiment_offsets = [-0.01, 0.0, 0.01, -0.005, 0.005]
    for index in range(N_exp):
        experiment_id = f"exp{index + 1}"
        effective_ic50 = 1.8 * (10 ** experiment_offsets[index])
        for concentration in concentrations:
            cell_key = (experiment_id, float(concentration))
            if missing_cells and cell_key in missing_cells:
                continue
            base_response = ic50_curve(concentration, ic50=effective_ic50)
            replicate_offsets = (
                [0.0]
                if single_replicate_cells and cell_key in single_replicate_cells
                else [-replicate_noise, 0.0, replicate_noise]
            )
            for replicate_id, noise in enumerate(replicate_offsets, start=1):
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


def make_clean_results() -> bc.FitResults:
    data = make_quality_data(replicate_noise=0.05)
    return bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})


def test_data_quality_report_green_for_balanced_low_noise_dataset():
    data = make_quality_data(replicate_noise=0.3)

    quality = data.quality_report()

    assert list(quality["compound_id"]) == ["cmpd_a"]
    assert quality.loc[0, "status"] == "green"
    assert quality.loc[0, "N_exp"] == 3
    assert quality.loc[0, "grid_coverage"] == pytest.approx(1.0)
    assert quality.loc[0, "single_replicate_fraction"] == pytest.approx(0.0)


def test_data_quality_report_red_for_single_experiment():
    data = make_quality_data(N_exp=1)

    quality = data.quality_report()

    assert quality.loc[0, "status"] == "red"
    assert "fewer than 2 independent experiments" in quality.loc[0, "flags"]


def test_data_quality_report_orange_for_two_experiments():
    data = make_quality_data(N_exp=2)

    quality = data.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert "only 2 independent experiments" in quality.loc[0, "flags"]


def test_data_quality_report_orange_for_incomplete_grid():
    concentrations = np.logspace(-2, 2, 8)
    data = make_quality_data(missing_cells={("exp2", float(concentrations[3]))})

    quality = data.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert quality.loc[0, "grid_coverage"] < 1.0
    assert "incomplete experiment-concentration grid" in quality.loc[0, "flags"]


def test_data_quality_report_orange_for_single_replicate_cells():
    concentrations = np.logspace(-2, 2, 8)
    data = make_quality_data(
        single_replicate_cells={("exp1", float(concentrations[0]))}
    )

    quality = data.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert quality.loc[0, "single_replicate_fraction"] > 0.0
    assert "single-replicate concentration cells present" in quality.loc[0, "flags"]


def test_data_quality_report_flags_high_intra_noise_orange_and_red():
    orange_data = make_quality_data(replicate_noise=6.0)
    red_data = make_quality_data(replicate_noise=12.0)

    orange_quality = orange_data.quality_report()
    red_quality = red_data.quality_report()

    assert orange_quality.loc[0, "status"] == "orange"
    assert "median intra-experiment noise" in orange_quality.loc[0, "flags"]
    assert red_quality.loc[0, "status"] == "red"
    assert "high median intra-experiment noise" in red_quality.loc[0, "flags"]


def test_data_quality_report_red_for_nonpositive_concentration():
    data = make_quality_data()
    data.table.loc[data.table.index[0], "concentration"] = 0.0

    quality = data.quality_report()

    assert quality.loc[0, "status"] == "red"
    assert quality.loc[0, "nonpositive_concentration_count"] == 1
    assert "nonpositive concentration values present" in quality.loc[0, "flags"]


def test_data_quality_report_preserves_order_and_threshold_override():
    first = make_quality_data(replicate_noise=6.0).table
    second = make_quality_data(replicate_noise=0.3).table.copy()
    second["compound_id"] = "cmpd_b"
    data = bc.DoseResponseData.from_dataframe(
        pd.concat([first, second], ignore_index=True)
    )

    quality = data.quality_report(compounds=["cmpd_b", "cmpd_a"])
    relaxed = data.quality_report(
        compounds="cmpd_a",
        thresholds=bc.DataQualityThresholds(
            max_intra_noise_median_frac_range_orange=0.20,
            max_intra_noise_median_frac_range_red=0.40,
            max_intra_noise_p90_frac_range_orange=0.30,
            max_intra_noise_p90_frac_range_red=0.50,
        ),
    )

    assert list(quality["compound_id"]) == ["cmpd_b", "cmpd_a"]
    assert quality.loc[1, "status"] == "orange"
    assert relaxed.loc[0, "status"] == "green"


def test_results_quality_report_green_for_clean_fits():
    results = make_clean_results()

    quality = results.quality_report()

    assert quality.loc[0, "status"] == "green"
    assert quality.loc[0, "parameter"] == "IC50"
    assert quality.loc[0, "N_fit_success"] == 3
    assert quality.loc[0, "N_fit_failed"] == 0


def test_results_quality_report_orange_for_partial_fit_failure():
    results = make_clean_results()
    results.fit_results.append(
        bc.FitResult.failed(
            compound_id="cmpd_a",
            model_name="ic50",
            experiment_id="exp4",
            message="failed",
        )
    )

    quality = results.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert quality.loc[0, "N_fit_failed"] == 1
    assert "fit failures present" in quality.loc[0, "flags"]


def test_results_quality_report_red_for_all_failed_fits():
    results = bc.FitResults(
        fit_results=[
            bc.FitResult.failed(
                compound_id="cmpd_a",
                model_name="ic50",
                experiment_id="exp1",
                message="failed",
            ),
            bc.FitResult.failed(
                compound_id="cmpd_a",
                model_name="ic50",
                experiment_id="exp2",
                message="failed",
            ),
        ],
        summaries=[],
    )

    quality = results.quality_report()

    assert quality.loc[0, "status"] == "red"
    assert quality.loc[0, "parameter"] == "IC50"
    assert quality.loc[0, "N_fit_success"] == 0
    assert "no successful fits" in quality.loc[0, "flags"]
    assert "selected concentration summary unavailable" in quality.loc[0, "flags"]


def test_results_quality_report_orange_for_missing_covariance():
    results = make_clean_results()
    results.fit_results[0].lmfit_result.covar = None

    quality = results.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert quality.loc[0, "covariance_missing_fraction"] > 0.0
    assert "missing covariance" in quality.loc[0, "flags"]


def test_results_quality_report_orange_for_missing_stderr():
    results = make_clean_results()
    estimate = results.fit_results[0].parameters["IC50"]
    results.fit_results[0].parameters["IC50"] = bc.ParameterEstimate(
        name=estimate.name,
        value=estimate.value,
        stderr=None,
        vary=estimate.vary,
    )

    quality = results.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert quality.loc[0, "stderr_missing_fraction"] > 0.0
    assert "missing parameter standard errors" in quality.loc[0, "flags"]


def test_results_quality_report_orange_for_parameter_at_bound():
    results = make_clean_results()
    parameter = results.fit_results[0].lmfit_result.params["IC50"]
    parameter.min = parameter.value

    quality = results.quality_report()

    assert quality.loc[0, "status"] == "orange"
    assert quality.loc[0, "parameter_at_bound_fraction"] > 0.0
    assert "parameter at a bound" in quality.loc[0, "flags"]


def test_results_quality_report_flags_wide_inter_ci95_fold_orange_and_red():
    base_results = make_clean_results()
    base_summary = next(
        summary
        for summary in base_results.summaries
        if isinstance(summary, bc.ConcentrationSummary)
    )

    orange_results = make_clean_results()
    orange_results.summaries = [
        replace(
            base_summary,
            log10_ci95_lower=0.0,
            log10_ci95_upper=0.6,
        )
    ]
    red_results = make_clean_results()
    red_results.summaries = [
        replace(
            base_summary,
            log10_ci95_lower=0.0,
            log10_ci95_upper=1.1,
        )
    ]

    orange_quality = orange_results.quality_report()
    red_quality = red_results.quality_report()

    assert orange_quality.loc[0, "status"] == "orange"
    assert "wide inter-experiment CI95 fold range" in orange_quality.loc[0, "flags"]
    assert red_quality.loc[0, "status"] == "red"
    assert "very wide inter-experiment CI95 fold range" in red_quality.loc[0, "flags"]


def test_results_quality_report_parameter_auto_raises_for_ambiguity():
    fit = bc.FitResult(compound_id="cmpd_a", model_name="ic50", success=True)
    results = bc.FitResults(
        fit_results=[fit],
        summaries=[
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
        ],
    )

    with pytest.raises(
        ValueError,
        match="Multiple reportable concentration quantities",
    ):
        results.quality_report()


def test_results_quality_report_preserves_compound_order_and_threshold_override():
    first = make_clean_results()
    second = make_clean_results()
    for fit in second.fit_results:
        fit.compound_id = "cmpd_b"
    second_summary = [
        replace(summary, compound_id="cmpd_b") for summary in second.summaries
    ]

    merged = bc.FitResults(
        fit_results=first.fit_results + second.fit_results,
        summaries=first.summaries + second_summary,
    )
    merged.summaries = [
        replace(summary, log10_ci95_lower=0.0, log10_ci95_upper=0.6)
        if (
            isinstance(summary, bc.ConcentrationSummary)
            and summary.compound_id == "cmpd_a"
        )
        else summary
        for summary in merged.summaries
    ]

    quality = merged.quality_report(compounds=["cmpd_b", "cmpd_a"])
    relaxed = merged.quality_report(
        compounds="cmpd_a",
        thresholds=bc.ResultQualityThresholds(
            max_inter_ci95_fold_orange=5.0,
            max_inter_ci95_fold_red=20.0,
        ),
    )

    assert list(quality["compound_id"]) == ["cmpd_b", "cmpd_a"]
    assert quality.loc[1, "status"] == "orange"
    assert relaxed.loc[0, "status"] == "green"
