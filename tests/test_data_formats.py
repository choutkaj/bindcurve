from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import bindcurve as bc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tutorials" / "tutorial_data" / "synthetic"


def make_round_trip_data(*, metadata: dict | None = None) -> bc.DoseResponseData:
    return bc.DoseResponseData.from_dataframe(
        pd.DataFrame(
            [
                {
                    "compound_id": "cmpd_a",
                    "experiment_id": "exp_1",
                    "concentration": 0.1,
                    "replicate_id": "response_1",
                    "response": 95.0,
                },
                {
                    "compound_id": "cmpd_a",
                    "experiment_id": "exp_1",
                    "concentration": 0.1,
                    "replicate_id": "response_2",
                    "response": 94.0,
                },
                {
                    "compound_id": "cmpd_a",
                    "experiment_id": "exp_1",
                    "concentration": 1.0,
                    "replicate_id": "response_1",
                    "response": 50.0,
                },
                {
                    "compound_id": "cmpd_a",
                    "experiment_id": "exp_1",
                    "concentration": 1.0,
                    "replicate_id": "response_2",
                    "response": 51.0,
                },
            ]
        ),
        metadata=metadata,
    )


def assert_same_long_table(left: pd.DataFrame, right: pd.DataFrame) -> None:
    sort_cols = ["compound_id", "experiment_id", "concentration", "replicate_id"]
    left_sorted = left.sort_values(sort_cols).reset_index(drop=True)
    right_sorted = right.sort_values(sort_cols).reset_index(drop=True)
    assert left_sorted[sort_cols + ["response"]].equals(
        right_sorted[sort_cols + ["response"]]
    )


def test_from_csv_loads_long_format_example():
    data = bc.DoseResponseData.from_csv(
        DATA_DIR / "direct_binding_long.csv",
        format="long",
    )

    # 3 compounds * 3 experiments * 11 points * 5 replicates = 495 rows
    assert len(data.table) == 495
    assert data.compounds == ["cmpd_1", "cmpd_2", "cmpd_3"]
    assert set(data.table["experiment_id"]) == {"exp_1", "exp_2", "exp_3"}
    assert set(data.table["replicate_id"]) == {
        "rep_1",
        "rep_2",
        "rep_3",
        "rep_4",
        "rep_5",
    }


def test_from_csv_loads_wide_format_example():
    data = bc.DoseResponseData.from_csv(
        DATA_DIR / "direct_binding_wide.csv",
        format="wide",
    )

    # 3 compounds * 3 experiments * 11 points = 99 rows (in long format it's 495)
    # The DoseResponseData.from_csv might return the long format table? 
    # Let's check bc.DoseResponseData.from_csv implementation if needed.
    # But usually it's the number of unique points.
    assert len(data.table) == 495
    assert data.compounds == ["cmpd_1", "cmpd_2", "cmpd_3"]
    assert set(data.table["experiment_id"]) == {"exp_1", "exp_2", "exp_3"}
    assert set(data.table["replicate_id"]) == {
        "response_1", "response_2", "response_3", "response_4", "response_5"
    }


def test_long_and_wide_examples_contain_same_direct_binding_values():
    long = bc.DoseResponseData.from_csv(
        DATA_DIR / "direct_binding_long.csv",
        format="long",
    ).table
    wide = bc.DoseResponseData.from_csv(
        DATA_DIR / "direct_binding_wide.csv",
        format="wide",
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
        DATA_DIR / "competitive_binding_long.csv",
        format="long",
    )
    wide = bc.DoseResponseData.from_csv(
        DATA_DIR / "competitive_binding_wide.csv",
        format="wide",
    )

    # 3 compounds * 3 experiments * 11 points * 5 replicates = 495 rows
    assert len(long.table) == 495
    assert len(wide.table) == 495
    assert len(long.compounds) == 3
    assert len(wide.compounds) == 3
    assert set(long.table["experiment_id"]) == {"exp_1", "exp_2", "exp_3"}
    assert set(wide.table["experiment_id"]) == {"exp_1", "exp_2", "exp_3"}


def test_from_csv_rejects_unknown_format(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("compound_id,concentration,response\ncmpd,1.0,50.0\n")

    with pytest.raises(ValueError, match="format must be"):
        bc.DoseResponseData.from_csv(path, format="fully_wide")


def test_from_csv_rejects_wide_without_replicate_columns(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("compound_id,experiment_id,concentration\ncmpd,exp_1,1.0\n")

    with pytest.raises(ValueError, match="No technical replicate columns"):
        bc.DoseResponseData.from_csv(path, format="wide")


def test_to_dataframe_can_emit_wide_format():
    data = make_round_trip_data()

    wide = data.to_dataframe(format="wide")

    assert list(wide.columns) == [
        "compound_id",
        "experiment_id",
        "concentration",
        "response_1",
        "response_2",
    ]
    assert len(wide) == 2
    assert wide["response_1"].tolist() == [95.0, 50.0]
    assert wide["response_2"].tolist() == [94.0, 51.0]


def test_to_csv_and_from_csv_round_trip_wide_format(tmp_path):
    data = make_round_trip_data()
    path = tmp_path / "data.csv"

    data.to_csv(path, format="wide")
    loaded = bc.DoseResponseData.from_csv(path, format="wide")

    assert_same_long_table(data.table, loaded.table)


def test_to_json_and_from_json_round_trip_preserves_metadata():
    data = make_round_trip_data(metadata={"assay": "direct_binding"})

    payload = data.to_json(format="wide")
    loaded = bc.DoseResponseData.from_json(payload)

    parsed = json.loads(payload)
    assert parsed["format"] == "wide"
    assert parsed["metadata"] == {"assay": "direct_binding"}
    assert loaded.metadata == {"assay": "direct_binding"}
    assert_same_long_table(data.table, loaded.table)


def test_to_json_and_from_json_round_trip_via_file(tmp_path):
    data = make_round_trip_data(metadata={"plate": "A01"})
    path = tmp_path / "data.json"

    data.to_json(path, format="long", indent=2)
    loaded = bc.DoseResponseData.from_json(path)

    assert loaded.metadata == {"plate": "A01"}
    assert_same_long_table(data.table, loaded.table)


def test_from_json_rejects_conflicting_requested_format():
    data = make_round_trip_data()
    payload = data.to_json(format="wide")

    with pytest.raises(ValueError, match="does not match"):
        bc.DoseResponseData.from_json(payload, format="long")
