from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=1.5, hill_slope=-1.1):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_data() -> bc.DoseResponseData:
    concentrations = np.logspace(-2, 2, 12)
    rows = []
    for experiment_id, multiplier in {"exp1": 0.95, "exp2": 1.05}.items():
        for concentration in concentrations:
            response = ic50_curve(concentration, ic50=1.5 * multiplier)
            for replicate_id, noise in enumerate([-0.25, 0.25], start=1):
                rows.append(
                    {
                        "compound_id": "cmpd_a",
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


def make_results(data: bc.DoseResponseData) -> bc.FitResults:
    return bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})


def test_plot_observations_draws_on_existing_axes():
    data = make_data()
    fig, ax = plt.subplots()

    returned_ax = bc.plot_observations(data, ax=ax)

    assert returned_ax is ax
    assert len(ax.collections) > 0
    assert ax.get_xlabel() == "concentration (uM)"
    assert ax.get_ylabel() == "response (percent)"
    plt.close(fig)


def test_plot_fits_draws_fit_lines_on_existing_axes():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_fits(data, results, ax=ax, n_points=50)

    assert returned_ax is ax
    assert len(ax.lines) == 2
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_curves_draws_observations_and_fits():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    returned_ax = bc.plot_curves(
        data,
        results,
        ax=ax,
        n_points=50,
        observation_kwargs={"markersize": 4},
        fit_kwargs={"linewidth": 1.5},
    )

    assert returned_ax is ax
    assert len(ax.collections) > 0
    assert len(ax.lines) >= 2
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_curves_requires_compound_id_for_multi_compound_data():
    data = make_data()
    extra = data.table.copy()
    extra["compound_id"] = "cmpd_b"
    multi = bc.DoseResponseData.from_dataframe(pd.concat([data.table, extra]))
    results = bc.fit(multi, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="compound_id"):
        bc.plot_curves(multi, results, ax=ax)
    plt.close(fig)


def test_plot_curves_can_select_experiment_subset():
    data = make_data()
    results = make_results(data)
    fig, ax = plt.subplots()

    bc.plot_curves(data, results, ax=ax, experiments=["exp1"], n_points=50)

    assert len(ax.lines) >= 1
    plt.close(fig)
