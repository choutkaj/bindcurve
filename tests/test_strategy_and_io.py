from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=2.0, hill_slope=-1.0):
    return ymin + (ymax - ymin) / (1.0 + (ic50 / x) ** hill_slope)


def make_multi_experiment_data() -> bc.DoseResponseData:
    rows = []
    concentrations = np.logspace(-2, 2, 12)
    for experiment_id, multiplier in {"exp1": 1.0, "exp2": 1.2, "exp3": 0.8}.items():
        for concentration in concentrations:
            response = ic50_curve(concentration, ic50=2.0 * multiplier)
            for replicate_id, offset in enumerate([-0.25, 0.25], start=1):
                rows.append(
                    {
                        "compound_id": "cmpd_a",
                        "experiment_id": experiment_id,
                        "concentration": concentration,
                        "replicate_id": f"rep{replicate_id}",
                        "response": response + offset,
                    }
                )
    return bc.DoseResponseData.from_dataframe(pd.DataFrame(rows))


def test_per_experiment_strategy_is_default():
    data = make_multi_experiment_data()
    results = bc.fit(data, fixed={"ymin": 0.0, "ymax": 100.0})
    fits = results.fits_to_dataframe()

    assert len(fits) == 3
    assert set(fits["experiment_id"]) == {"exp1", "exp2", "exp3"}
    assert fits["n_data"].eq(12).all()


def test_pooled_strategy_fits_all_raw_observations_once():
    data = make_multi_experiment_data()
    settings = bc.FitSettings(strategy="pooled")
    results = bc.fit(data, settings=settings, fixed={"ymin": 0.0, "ymax": 100.0})
    fits = results.fits_to_dataframe()

    assert len(fits) == 1
    assert fits.loc[0, "experiment_id"] is None
    assert fits.loc[0, "n_data"] == 72
    assert fits.loc[0, "success"]


def test_per_compound_summary_strategy_fits_one_aggregated_curve():
    data = make_multi_experiment_data()
    settings = bc.FitSettings(strategy="per_compound_summary")
    results = bc.fit(data, settings=settings, fixed={"ymin": 0.0, "ymax": 100.0})
    fits = results.fits_to_dataframe()

    assert len(fits) == 1
    assert fits.loc[0, "experiment_id"] == "compound_summary"
    assert fits.loc[0, "n_data"] == 12
    assert fits.loc[0, "success"]


def test_from_wide_dataframe_normalizes_to_long_form():
    wide = pd.DataFrame(
        {
            "compound": ["cmpd_a", "cmpd_a", "cmpd_b", "cmpd_b"],
            "experiment": ["exp1", "exp1", "exp1", "exp1"],
            "dose": [0.1, 1.0, 0.1, 1.0],
            "rep_1": [95.0, 50.0, 90.0, 40.0],
            "rep_2": [94.0, 51.0, 91.0, 39.0],
        }
    )

    data = bc.DoseResponseData.from_wide_dataframe(
        wide,
        compound_col="compound",
        concentration_col="dose",
        experiment_col="experiment",
        replicate_cols=["rep_1", "rep_2"],
        concentration_unit="uM",
        response_unit="percent",
    )

    assert set(data.table.columns) >= {
        "compound_id",
        "experiment_id",
        "concentration",
        "replicate_id",
        "response",
    }
    assert len(data.table) == 8
    assert data.compounds == ["cmpd_a", "cmpd_b"]
    assert data.concentration_unit == "uM"
    assert data.response_unit == "percent"


def test_missing_optional_ids_are_filled():
    df = pd.DataFrame(
        {
            "compound_id": ["cmpd_a", "cmpd_a"],
            "concentration": [0.1, 1.0],
            "response": [90.0, 50.0],
        }
    )

    data = bc.DoseResponseData.from_dataframe(df)

    assert set(data.table["experiment_id"]) == {"experiment_1"}
    assert list(data.table["replicate_id"]) == ["replicate_1", "replicate_1"]


def test_errors_raise_is_default_for_unknown_compound():
    data = make_multi_experiment_data()

    with pytest.raises(KeyError, match="missing"):
        bc.fit(data, compounds=["missing"])


def test_fixed_parameters_and_bounds_are_respected():
    data = make_multi_experiment_data()
    results = bc.fit(
        data,
        fixed={"ymin": 0.0, "ymax": 100.0},
        bounds={"IC50": (0.1, 10.0), "hill_slope": (-3.0, -0.1)},
    )
    fits = results.fits_to_dataframe()

    assert fits["ymin"].eq(0.0).all()
    assert fits["ymax"].eq(100.0).all()
    assert fits["IC50"].between(0.1, 10.0).all()
    assert fits["hill_slope"].between(-3.0, -0.1).all()
