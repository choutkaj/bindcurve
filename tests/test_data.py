from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc
from bindcurve.data import replicate_means

TUTORIAL_DATA = Path(__file__).parents[1] / "docs" / "tutorials" / "data"


def table(**columns):
    base = {"compound_id": "a", "concentration": [1.0, 2.0], "response": [10.0, 5.0]}
    return pd.DataFrame({**base, **columns})


def test_defaults_and_identifier_types():
    data = bc.DoseResponseData(table(compound_id=[1, 1], extra=["x", "y"]))
    assert data.table["experiment_id"].tolist() == ["experiment_1"] * 2
    assert data.compounds == ["1"]
    assert data.table["extra"].tolist() == ["x", "y"]


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"concentration": [0.0, 1.0]}, "concentration must be positive"),
        ({"concentration": [np.nan, 1.0]}, "concentration must contain only finite"),
        ({"response": [np.inf, 1.0]}, "response must contain only finite"),
        ({"compound_id": ["a", None]}, "compound_id contains missing"),
    ],
)
def test_invalid_observations_are_rejected(columns, message):
    with pytest.raises(ValueError, match=message):
        bc.DoseResponseData(table(**columns))


def test_missing_columns_and_empty_tables_are_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        bc.DoseResponseData(pd.DataFrame({"compound_id": ["a"], "response": [1.0]}))
    with pytest.raises(ValueError, match="empty"):
        bc.DoseResponseData(table().iloc[:0])


def test_long_and_wide_tutorial_files_agree(tmp_path):
    wide = bc.DoseResponseData.from_csv(
        TUTORIAL_DATA / "direct-binding.csv", format="wide"
    )
    long_path = tmp_path / "long.csv"
    wide.table.to_csv(long_path, index=False)
    long = bc.DoseResponseData.from_csv(long_path)
    key = ["compound_id", "experiment_id", "concentration", "response"]
    pd.testing.assert_frame_equal(
        long.table[key].sort_values(key).reset_index(drop=True),
        wide.table[key].sort_values(key).reset_index(drop=True),
    )
    assert wide.compounds == ["simple", "specific", "total"]


def test_wide_csv_files_accept_a_replicate_prefix(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {"compound_id": "a", "concentration": [1.0, 2.0], "rep1": [3.0, 4.0]}
    ).to_csv(path, index=False)
    data = bc.DoseResponseData.from_csv(path, format="wide", prefix="rep")
    assert data.table["response"].tolist() == [3.0, 4.0]


def test_wide_tables_drop_missing_replicates_and_reject_other_columns():
    wide = pd.DataFrame(
        {
            "compound_id": "a",
            "concentration": [1.0, 2.0],
            "response_1": [1.0, 2.0],
            "response_2": [3.0, np.nan],
        }
    )
    assert len(bc.DoseResponseData.from_wide(wide).table) == 3
    with pytest.raises(ValueError, match="cannot contain other columns"):
        bc.DoseResponseData.from_wide(wide.assign(note="x"))
    with pytest.raises(ValueError, match="No response columns"):
        bc.DoseResponseData.from_wide(wide, prefix="signal_")
    with pytest.raises(ValueError, match="format"):
        bc.DoseResponseData.from_csv(
            TUTORIAL_DATA / "direct-binding.csv", format="tall"
        )


def test_select_and_summary():
    data = bc.DoseResponseData(
        pd.DataFrame(
            {
                "compound_id": ["a", "a", "a", "b"],
                "experiment_id": ["e1", "e1", "e2", "e1"],
                "concentration": [1.0, 2.0, 1.0, 3.0],
                "response": [1.0, 2.0, 3.0, 4.0],
            }
        )
    )
    assert data.select("b").compounds == ["b"]
    assert data.select(["a", "b"]).compounds == ["a", "b"]
    with pytest.raises(KeyError, match="'c'"):
        data.select(["a", "c"])
    summary = data.summary().set_index("compound_id")
    assert summary.loc["a", ["N_exp", "N_obs"]].tolist() == [2, 3]
    assert summary.loc["b", "concentration_max"] == 3.0


def test_replicate_means():
    observed = pd.DataFrame(
        {"concentration": [1.0, 1.0, 2.0], "response": [10.0, 14.0, 5.0]}
    )
    means = replicate_means(observed, ["concentration"])
    assert means["response"].tolist() == [12.0, 5.0]
    assert means["n"].tolist() == [2, 1]
    assert means["sd"].iloc[0] == pytest.approx(np.sqrt(8.0))
    assert means["sem"].iloc[0] == pytest.approx(2.0)
