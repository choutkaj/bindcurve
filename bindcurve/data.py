"""Validated dose-response observations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("compound_id", "concentration", "response")
DEFAULT_EXPERIMENT = "experiment_1"


class DoseResponseData:
    """Long-form dose-response observations, one row per measured response.

    Parameters
    ----------
    table
        Columns ``compound_id``, ``concentration`` and ``response`` are
        required. ``experiment_id`` identifies independent experiments and
        defaults to a single experiment. Other columns are kept.

    Notes
    -----
    Concentrations must be finite and positive and share one unit. Rows with
    the same compound, experiment and concentration are technical replicates.
    """

    def __init__(self, table: pd.DataFrame) -> None:
        self._table = _validated(table)

    @classmethod
    def from_wide(
        cls, table: pd.DataFrame, *, prefix: str = "response_"
    ) -> DoseResponseData:
        """Create data from one row per concentration with replicate columns.

        Replicate responses are the columns whose names start with ``prefix``.
        Missing replicate values are ignored.
        """
        responses = [
            column for column in table.columns if str(column).startswith(prefix)
        ]
        if not responses:
            raise ValueError(f"No response columns starting with {prefix!r}.")
        keys = [column for column in table.columns if column not in responses]
        unknown = set(keys) - {"compound_id", "experiment_id", "concentration"}
        if unknown:
            raise ValueError(
                f"Wide tables cannot contain other columns: {sorted(unknown)}."
            )
        long = table.melt(id_vars=keys, value_vars=responses, value_name="response")
        return cls(long.drop(columns="variable").dropna(subset=["response"]))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        format: Literal["long", "wide"] = "long",
        prefix: str = "response_",
        **read_csv_kwargs: object,
    ) -> DoseResponseData:
        """Read a long or wide CSV file; see the constructor and `from_wide`.

        ``prefix`` names the replicate response columns of wide files.
        """
        table = pd.read_csv(path, **read_csv_kwargs)
        if format == "long":
            return cls(table)
        if format == "wide":
            return cls.from_wide(table, prefix=prefix)
        raise ValueError("format must be 'long' or 'wide'.")

    @property
    def table(self) -> pd.DataFrame:
        """A copy of the validated observation table."""
        return self._table.copy()

    @property
    def compounds(self) -> list[str]:
        """Sorted compound identifiers."""
        return sorted(self._table["compound_id"].unique())

    def select(self, compounds: str | Iterable[str]) -> DoseResponseData:
        """Return the observations of the given compound or compounds."""
        selected = [compounds] if isinstance(compounds, str) else list(compounds)
        missing = sorted(set(selected) - set(self.compounds))
        if missing:
            raise KeyError(f"Unknown compound(s): {missing}")
        return DoseResponseData(self._table[self._table["compound_id"].isin(selected)])

    def summary(self) -> pd.DataFrame:
        """One row per compound with experiment, observation and range counts."""
        grouped = self._table.groupby("compound_id")
        return pd.DataFrame(
            {
                "N_exp": grouped["experiment_id"].nunique(),
                "N_obs": grouped.size(),
                "concentration_min": grouped["concentration"].min(),
                "concentration_max": grouped["concentration"].max(),
            }
        ).reset_index()


def replicate_means(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean, sample SD, SEM and count of ``response`` per group."""
    groups = table.groupby(by, sort=True)
    means = groups["response"].agg(response="mean", sd="std", n="count")
    means["sem"] = means["sd"] / np.sqrt(means["n"])
    return means.reset_index()


def _validated(table: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if table.empty:
        raise ValueError("The observation table is empty.")

    table = table.copy()
    if "experiment_id" not in table.columns:
        table["experiment_id"] = DEFAULT_EXPERIMENT
    for column in ("compound_id", "experiment_id"):
        if table[column].isna().any():
            raise ValueError(f"{column} contains missing values.")
        table[column] = table[column].astype(str)

    for column in ("concentration", "response"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
        if not np.isfinite(table[column]).all():
            raise ValueError(f"{column} must contain only finite values.")
    if (table["concentration"] <= 0.0).any():
        raise ValueError("concentration must be positive.")
    return table.reset_index(drop=True)
