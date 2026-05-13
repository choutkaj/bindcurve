from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.8, hill_slope=-1.0):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_quality_data(
    *,
    compounds: tuple[str, ...] = ("cmpd_a",),
) -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 8)
    rows = []
    for compound_index, compound_id in enumerate(compounds):
        offset = 0.02 * compound_index
        for experiment_id, experiment_offset in {
            "exp1": -0.01,
            "exp2": 0.0,
            "exp3": 0.01,
        }.items():
            effective_ic50 = 1.8 * (10 ** (offset + experiment_offset))
            for concentration in concentrations:
                base_response = ic50_curve(concentration, ic50=effective_ic50)
                for replicate_id, noise in enumerate([-0.2, 0.0, 0.2], start=1):
                    rows.append(
                        {
                            "compound_id": compound_id,
                            "experiment_id": experiment_id,
                            "concentration": concentration,
                            "replicate_id": f"rep{replicate_id}",
                            "response": base_response + noise,
                        }
                    )
    return bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))


def make_results() -> bc.FitResults:
    data = make_quality_data()
    return bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})


def test_data_quality_dashboard_returns_three_panel_figure_for_one_compound():
    data = make_quality_data()

    figure = data.quality_dashboard()

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert "data QC" in figure.axes[0].get_title(loc="left")
    assert figure.axes[1].get_title() == "Replicate Count"
    assert figure.axes[2].get_title() == "Replicate SD / Response Range"
    plt.close(figure)


def test_data_quality_dashboard_respects_compound_subset_order():
    data = make_quality_data(compounds=("cmpd_a", "cmpd_b"))

    figure = data.quality_dashboard(compounds=["cmpd_b", "cmpd_a"])

    assert len(figure.axes) == 6
    titles = [axis.get_title(loc="left") for axis in figure.axes[::3]]
    assert titles[0].startswith("cmpd_b")
    assert titles[1].startswith("cmpd_a")
    plt.close(figure)


def test_results_quality_dashboard_returns_three_panel_figure():
    results = make_results()

    figure = results.quality_dashboard()

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert "results QC" in figure.axes[0].get_title(loc="left")
    assert "logIC50" in figure.axes[1].get_title()
    assert figure.axes[2].get_title() == "Per-fit diagnostics"
    plt.close(figure)


def test_results_quality_dashboard_handles_failed_fit_and_preserves_order():
    first = make_results()
    second = make_results()
    for fit in second.fit_results:
        fit.compound_id = "cmpd_b"
    second.summaries = [
        (
            summary
            if summary.compound_id != "cmpd_a"
            else _replace_compound(summary, "cmpd_b")
        )
        for summary in second.summaries
    ]
    first.fit_results.append(
        bc.FitResult.failed(
            compound_id="cmpd_a",
            model_name="ic50",
            experiment_id="exp4",
            message="failed",
        )
    )
    merged = bc.FitResults(
        fit_results=first.fit_results + second.fit_results,
        summaries=first.summaries + second.summaries,
    )

    figure = merged.quality_dashboard(compounds=["cmpd_b", "cmpd_a"])

    assert len(figure.axes) == 6
    titles = [axis.get_title(loc="left") for axis in figure.axes[::3]]
    assert titles[0].startswith("cmpd_b")
    assert titles[1].startswith("cmpd_a")
    plt.close(figure)


def test_results_quality_dashboard_raises_for_ambiguous_auto_parameter():
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
        results.quality_dashboard()


def _replace_compound(summary: object, compound_id: str) -> object:
    if isinstance(summary, bc.ConcentrationSummary):
        return bc.ConcentrationSummary(
            compound_id=compound_id,
            parameter=summary.parameter,
            log_parameter=summary.log_parameter,
            N_exp=summary.N_exp,
            reportable=summary.reportable,
            log10_mean=summary.log10_mean,
            log10_sd=summary.log10_sd,
            log10_sem=summary.log10_sem,
            log10_ci95_lower=summary.log10_ci95_lower,
            log10_ci95_upper=summary.log10_ci95_upper,
        )
    return bc.ParameterSummary(
        compound_id=compound_id,
        parameter=summary.parameter,
        N_exp=summary.N_exp,
        mean=summary.mean,
        sd=summary.sd,
        sem=summary.sem,
        ci95_lower=summary.ci95_lower,
        ci95_upper=summary.ci95_upper,
    )
