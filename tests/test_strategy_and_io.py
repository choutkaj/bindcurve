from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc
from bindcurve.modeling import (
    CompetitiveFourStateSpecificKdModel,
    CompetitiveFourStateTotalKdModel,
    CompetitiveThreeStateSpecificKdModel,
    CompetitiveThreeStateTotalKdModel,
    DirectSimpleKdModel,
    DirectSpecificKdModel,
    DirectTotalKdModel,
    IC50Model,
)


def ic50_curve(x, *, ymin=0.0, ymax=100.0, ic50=2.0, hill_slope=1.0):
    return ymin + (ymax - ymin) / (1.0 + (x / ic50) ** hill_slope)


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


def make_multi_compound_data() -> bc.DoseResponseData:
    rows = []
    for compound_id, base_response in [
        ("cmpd_b", 92.0),
        ("cmpd_a", 88.0),
        ("cmpd_c", 84.0),
    ]:
        for concentration, drop in [(0.1, 0.0), (1.0, 20.0)]:
            rows.append(
                {
                    "compound_id": compound_id,
                    "experiment_id": "exp1",
                    "concentration": concentration,
                    "replicate_id": "rep1",
                    "response": base_response - drop,
                }
            )
    return bc.DoseResponseData.from_dataframe(
        pd.DataFrame(rows),
        metadata={"source": "synthetic"},
    )


def test_fit_runs_one_curve_per_experiment():
    data = make_multi_experiment_data()
    results = bc.fit(data, fixed={"ymin": 0.0, "ymax": 100.0})
    fits = results.fit_summary()

    assert len(fits) == 3
    assert set(fits["experiment_id"]) == {"exp1", "exp2", "exp3"}
    assert fits["n_data"].eq(12).all()


def test_fitsettings_no_longer_accepts_strategy():
    with pytest.raises(TypeError, match="strategy"):
        bc.FitSettings(strategy="per_experiment")


def test_fitsettings_no_longer_accepts_weighting():
    with pytest.raises(TypeError, match="weighting"):
        bc.FitSettings(weighting="none")


def test_from_dataframe_accepts_wide_format():
    wide = pd.DataFrame(
        {
            "compound": ["cmpd_a", "cmpd_a", "cmpd_b", "cmpd_b"],
            "experiment": ["exp1", "exp1", "exp1", "exp1"],
            "dose": [0.1, 1.0, 0.1, 1.0],
            "rep_1": [95.0, 50.0, 90.0, 40.0],
            "rep_2": [94.0, 51.0, 91.0, 39.0],
        }
    )

    data = bc.DoseResponseData.from_dataframe(
        wide,
        format="wide",
        compound_col="compound",
        concentration_col="dose",
        experiment_col="experiment",
        replicate_cols=["rep_1", "rep_2"],
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


def test_summary_returns_one_row_per_compound():
    data = make_multi_experiment_data()
    extra = pd.DataFrame(
        [
            {
                "compound_id": "cmpd_b",
                "experiment_id": "exp1",
                "concentration": 0.1,
                "replicate_id": "rep1",
                "response": 88.0,
            },
            {
                "compound_id": "cmpd_b",
                "experiment_id": "exp1",
                "concentration": 1.0,
                "replicate_id": "rep1",
                "response": 44.0,
            },
        ]
    )
    data = bc.DoseResponseData.from_dataframe(pd.concat([data.table, extra]))

    summary = data.summary()

    assert list(summary["compound_id"]) == ["cmpd_a", "cmpd_b"]
    assert list(summary.columns) == [
        "compound_id",
        "N_exp",
        "N_obs",
        "N_conc_total",
        "concentration_min",
        "concentration_max",
        "response_min",
        "response_max",
    ]

    cmpd_a = summary.loc[summary["compound_id"] == "cmpd_a"].iloc[0]
    assert cmpd_a["N_exp"] == 3
    assert cmpd_a["N_obs"] == 72
    assert cmpd_a["N_conc_total"] == 12
    assert cmpd_a["concentration_min"] == pytest.approx(1e-2)
    assert cmpd_a["concentration_max"] == pytest.approx(1e2)

    cmpd_b = summary.loc[summary["compound_id"] == "cmpd_b"].iloc[0]
    assert cmpd_b["N_exp"] == 1
    assert cmpd_b["N_obs"] == 2
    assert cmpd_b["N_conc_total"] == 2
    assert cmpd_b["concentration_min"] == pytest.approx(0.1)
    assert cmpd_b["concentration_max"] == pytest.approx(1.0)


def test_keep_only_supports_name_index_mixed_and_negative_selectors():
    data = make_multi_compound_data()

    by_name = data.keep_only("cmpd_a")
    by_index = data.keep_only(1)
    mixed = data.keep_only(["cmpd_a", 1, -1])
    by_negative = data.keep_only(-1)

    assert by_name.compounds == ["cmpd_a"]
    assert by_index.compounds == ["cmpd_b"]
    assert mixed.compounds == ["cmpd_a", "cmpd_b", "cmpd_c"]
    assert by_negative.compounds == ["cmpd_c"]


def test_remove_supports_name_index_and_mixed_selectors():
    data = make_multi_compound_data()

    removed_name = data.remove("cmpd_a")
    removed_index = data.remove(0)
    removed_mixed = data.remove(["cmpd_a", -1])

    assert removed_name.compounds == ["cmpd_b", "cmpd_c"]
    assert removed_index.compounds == ["cmpd_b", "cmpd_c"]
    assert removed_mixed.compounds == ["cmpd_b"]


def test_compound_filtering_preserves_row_order_and_metadata():
    data = make_multi_compound_data()

    filtered = data.keep_only(["cmpd_a", "cmpd_c"])
    expected = data.table[
        data.table["compound_id"].isin(["cmpd_a", "cmpd_c"])
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(filtered.table.reset_index(drop=True), expected)
    assert filtered.metadata == {"source": "synthetic"}


def test_compound_filtering_deduplicates_repeated_selectors():
    data = make_multi_compound_data()

    filtered = data.keep_only(["cmpd_a", "cmpd_a", 1, 1])

    assert filtered.compounds == ["cmpd_a", "cmpd_b"]


def test_compound_filtering_raises_for_invalid_selectors_and_empty_results():
    data = make_multi_compound_data()

    with pytest.raises(KeyError, match="missing"):
        data.keep_only("missing")
    with pytest.raises(IndexError, match="out of range"):
        data.keep_only(10)
    with pytest.raises(TypeError, match="strings or integers"):
        data.keep_only(1.5)
    with pytest.raises(ValueError, match="removed all compounds"):
        data.keep_only([])
    with pytest.raises(ValueError, match="removed all compounds"):
        data.remove(["cmpd_a", "cmpd_b", "cmpd_c"])


def test_concatenate_merges_disjoint_compounds_and_preserves_metadata():
    first = make_multi_compound_data().keep_only("cmpd_a")
    second = make_multi_compound_data().keep_only("cmpd_b")

    combined = bc.DoseResponseData.concatenate(first, second)
    expected = pd.concat([first.table, second.table], ignore_index=True)

    pd.testing.assert_frame_equal(combined.table.reset_index(drop=True), expected)
    assert combined.compounds == ["cmpd_a", "cmpd_b"]
    assert combined.metadata == {"source": "synthetic"}


def test_concatenate_merges_same_compound_with_distinct_experiments():
    first = make_multi_compound_data().keep_only("cmpd_a")
    second_table = first.table.copy()
    second_table["experiment_id"] = "exp2"
    second = bc.DoseResponseData(second_table, metadata=first.metadata)

    combined = bc.DoseResponseData.concatenate(first, second)

    assert combined.compounds == ["cmpd_a"]
    assert set(combined.table["experiment_id"]) == {"exp1", "exp2"}


def test_concatenate_rejects_different_metadata():
    first = make_multi_compound_data().keep_only("cmpd_a")
    second_table = first.table.copy()
    second_table["experiment_id"] = "exp2"
    second = bc.DoseResponseData(second_table, metadata={"source": "other"})

    with pytest.raises(ValueError, match="different metadata"):
        bc.DoseResponseData.concatenate(first, second)


def test_concatenate_raises_for_overlapping_experiment_ids_and_invalid_inputs():
    first = make_multi_compound_data().keep_only("cmpd_a")
    second = make_multi_compound_data().keep_only("cmpd_a")

    with pytest.raises(ValueError, match="reuses experiment_id"):
        bc.DoseResponseData.concatenate(first, second)
    with pytest.raises(ValueError, match="at least 2 datasets"):
        bc.DoseResponseData.concatenate(first)
    with pytest.raises(TypeError, match="DoseResponseData"):
        bc.DoseResponseData.concatenate(first, "not-data")


def test_errors_raise_is_default_for_unknown_compound():
    data = make_multi_experiment_data()

    with pytest.raises(KeyError, match="missing"):
        bc.fit(data, compounds=["missing"])


def test_fixed_parameters_and_bounds_are_respected():
    data = make_multi_experiment_data()
    results = bc.fit(
        data,
        fixed={"ymin": 0.0, "ymax": 100.0},
        bounds={"IC50": (0.1, 10.0), "hill_slope": (0.1, 3.0)},
    )
    fits = results.fit_summary()

    assert fits["ymin"].eq(0.0).all()
    assert fits["ymax"].eq(100.0).all()
    assert fits["IC50"].between(0.1, 10.0).all()
    assert fits["hill_slope"].between(0.1, 3.0).all()


def test_concentration_parameter_specs_are_strictly_positive():
    models = [
        IC50Model(),
        DirectSimpleKdModel(),
        DirectSpecificKdModel(),
        DirectTotalKdModel(),
        CompetitiveThreeStateSpecificKdModel(),
        CompetitiveThreeStateTotalKdModel(),
        CompetitiveFourStateSpecificKdModel(),
        CompetitiveFourStateTotalKdModel(),
    ]

    for model in models:
        for spec in model.concentration_parameter_specs:
            assert spec.min > 0.0


def test_fixed_zero_concentration_parameter_is_rejected():
    data = make_multi_experiment_data()

    with pytest.raises(ValueError, match="IC50.*strictly positive"):
        bc.fit(
            data,
            fixed={"ymin": 0.0, "ymax": 100.0, "IC50": 0.0},
        )


def test_zero_lower_bound_for_concentration_parameter_is_rejected():
    data = make_multi_experiment_data()

    with pytest.raises(ValueError, match="Lower bound.*IC50.*strictly positive"):
        bc.fit(
            data,
            fixed={"ymin": 0.0, "ymax": 100.0},
            bounds={"IC50": (0.0, 10.0)},
        )


def test_zero_nonspecific_factor_is_allowed():
    model = DirectTotalKdModel()

    parameters = model.make_lmfit_parameters(
        {"Kds": 1.0},
        fixed={"ymin": 0.0, "ymax": 1.0, "LsT": 1.0, "Ns": 0.0},
    )

    assert parameters["Ns"].value == 0.0
