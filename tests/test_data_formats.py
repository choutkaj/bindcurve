from __future__ import annotations

from pathlib import Path

import pytest

import bindcurve as bc

ROOT = Path(__file__).resolve().parents[1]


def test_from_csv_loads_long_format_example():
    data = bc.DoseResponseData.from_csv(
        ROOT / "synthetic_direct_binding_long.csv",
        format="long",
        concentration_unit="uM",
        response_unit="percent",
    )

    assert len(data.table) == 36
    assert data.compounds == ["direct_cmpd_1", "direct_cmpd_2", "direct_cmpd_3"]
    assert set(data.table["experiment_id"]) == {"exp_1", "exp_2"}
    assert set(data.table["replicate_id"]) == {"rep_1", "rep_2"}
    assert data.concentration_unit == "uM"
    assert data.response_unit == "percent"


def test_from_csv_loads_replicate_wide_format_example():
    data = bc.DoseResponseData.from_csv(
        ROOT / "synthetic_direct_binding_replicate_wide.csv",
        format="replicate_wide",
    )

    assert len(data.table) == 36
    assert data.compounds == ["direct_cmpd_1", "direct_cmpd_2", "direct_cmpd_3"]
    assert set(data.table["experiment_id"]) == {"exp_1", "exp_2"}
    assert set(data.table["replicate_id"]) == {"response_1", "response_2"}


def test_long_and_replicate_wide_examples_contain_same_direct_binding_values():
    long = bc.DoseResponseData.from_csv(
        ROOT / "synthetic_direct_binding_long.csv",
        format="long",
    ).table
    wide = bc.DoseResponseData.from_csv(
        ROOT / "synthetic_direct_binding_replicate_wide.csv",
        format="replicate_wide",
    ).table

    long_normalized = long.copy()
    long_normalized["replicate_id"] = long_normalized["replicate_id"].str.replace(
        "rep_", "response_", regex=False
    )

    sort_cols = ["compound_id", "experiment_id", "concentration", "replicate_id"]
    long_normalized = long_normalized.sort_values(sort_cols).reset_index(drop=True)
    wide = wide.sort_values(sort_cols).reset_index(drop=True)

    assert long_normalized[sort_cols + ["response"]].equals(
        wide[sort_cols + ["response"]]
    )


def test_from_csv_loads_competitive_examples():
    long = bc.DoseResponseData.from_csv(
        ROOT / "synthetic_competitive_binding_long.csv",
        format="long",
    )
    wide = bc.DoseResponseData.from_csv(
        ROOT / "synthetic_competitive_binding_replicate_wide.csv",
        format="replicate_wide",
    )

    assert len(long.table) == 120
    assert len(wide.table) == 120
    assert len(long.compounds) == 10
    assert len(wide.compounds) == 10
    assert set(long.table["experiment_id"]) == {"exp_1", "exp_2"}
    assert set(wide.table["experiment_id"]) == {"exp_1", "exp_2"}


def test_from_csv_rejects_unknown_format(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("compound_id,concentration,response\ncmpd,1.0,50.0\n")

    with pytest.raises(ValueError, match="format must be"):
        bc.DoseResponseData.from_csv(path, format="fully_wide")


def test_from_csv_rejects_replicate_wide_without_replicate_columns(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("compound_id,experiment_id,concentration\ncmpd,exp_1,1.0\n")

    with pytest.raises(ValueError, match="No technical replicate columns"):
        bc.DoseResponseData.from_csv(path, format="replicate_wide")
