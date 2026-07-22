from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import bindcurve as bc
from bindcurve.modeling import IC50Model, ParameterSpec


class AmbiguousConcentrationModel(IC50Model):
    name = "ambiguous_concentrations"
    parameter_specs = IC50Model.parameter_specs + (
        ParameterSpec(
            "Kd",
            min=np.finfo(float).tiny,
            kind="concentration",
            scale="log10",
            log_name="logKd",
        ),
    )


def make_ambiguous_results() -> bc.FitResults:
    model = AmbiguousConcentrationModel()
    fits = tuple(
        bc.FitResult(
            model=model,
            compound_id="cmpd_a",
            experiment_id=f"exp{index}",
            parameters={
                "ymin": bc.ParameterEstimate("ymin", 0.0, vary=False),
                "ymax": bc.ParameterEstimate(
                    "ymax", 100.0, vary=False
                ),
                "IC50": bc.ParameterEstimate("IC50", 1.0 + index, stderr=0.1),
                "hill_slope": bc.ParameterEstimate(
                    "hill_slope", 1.0, vary=False
                ),
                "Kd": bc.ParameterEstimate("Kd", 2.0 + index, stderr=0.1),
            },
        )
        for index in range(1, 4)
    )
    return bc.FitResults(model=model, fit_results=fits)


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.8, hill_slope=1.0):
    return ymin + (ymax - ymin) / (1.0 + (x / ic50) ** hill_slope)


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
    second_fits = tuple(
        replace(fit, compound_id="cmpd_b") for fit in make_results().fit_results
    )
    failure = bc.FitResult.failed(
        model=first.model,
        compound_id="cmpd_a",
        experiment_id="exp4",
        stage="test",
        error=RuntimeError("synthetic failure"),
    )
    merged = bc.FitResults(
        model=first.model,
        fit_results=first.fit_results + (failure,) + second_fits,
    )

    figure = merged.quality_dashboard(compounds=["cmpd_b", "cmpd_a"])

    assert len(figure.axes) == 6
    titles = [axis.get_title(loc="left") for axis in figure.axes[::3]]
    assert titles[0].startswith("cmpd_b")
    assert titles[1].startswith("cmpd_a")
    plt.close(figure)


def test_results_quality_dashboard_raises_for_ambiguous_auto_parameter():
    results = make_ambiguous_results()

    with pytest.raises(
        ValueError,
        match="Multiple reportable concentration quantities",
    ):
        results.quality_dashboard()
