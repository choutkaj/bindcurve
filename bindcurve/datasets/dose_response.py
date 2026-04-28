from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

AggregationMethod = Literal["mean", "median"]


@dataclass
class CompoundData:
    """View of dose-response observations for one compound."""

    compound_id: str
    table: pd.DataFrame
    concentration_unit: str | None = None
    response_unit: str | None = None

    @property
    def experiments(self) -> list[str]:
        """Return sorted experiment identifiers for this compound."""
        return sorted(self.table["experiment_id"].astype(str).unique())

    @property
    def concentrations(self) -> np.ndarray:
        """Return sorted unique concentrations for this compound."""
        return np.sort(self.table["concentration"].unique())

    def select_experiment(self, experiment_id: str) -> "CompoundData":
        """Return a view containing only one independent experiment."""
        selected = self.table[self.table["experiment_id"].astype(str) == str(experiment_id)]
        if selected.empty:
            raise KeyError(
                f"Experiment {experiment_id!r} not found for compound "
                f"{self.compound_id!r}."
            )
        return CompoundData(
            compound_id=self.compound_id,
            table=selected.copy(),
            concentration_unit=self.concentration_unit,
            response_unit=self.response_unit,
        )

    def aggregate_replicates(
        self,
        *,
        method: AggregationMethod = "mean",
    ) -> pd.DataFrame:
        """Aggregate technical replicates by concentration."""
        if method not in {"mean", "median"}:
            raise ValueError("method must be 'mean' or 'median'.")

        grouped = self.table.groupby("concentration", as_index=False)["response"]
        if method == "mean":
            response = grouped.mean().rename(columns={"response": "response"})
        else:
            response = grouped.median().rename(columns={"response": "response"})

        diagnostics = grouped.agg(response_sd="std", n_replicates="count")
        diagnostics["response_sem"] = diagnostics["response_sd"] / np.sqrt(
            diagnostics["n_replicates"]
        )

        return response.merge(diagnostics, on="concentration", how="left").sort_values(
            "concentration"
        )

    def as_xy(
        self,
        *,
        aggregate_replicates: bool = True,
        aggregation: AggregationMethod = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return concentration and response arrays for fitting."""
        if aggregate_replicates:
            table = self.aggregate_replicates(method=aggregation)
        else:
            table = self.table.sort_values("concentration")
        return (
            table["concentration"].to_numpy(dtype=float),
            table["response"].to_numpy(dtype=float),
        )


@dataclass
class DoseResponseData:
    """Validated long-form dose-response observations."""

    table: pd.DataFrame
    concentration_unit: str | None = None
    response_unit: str | None = None
    metadata: dict = field(default_factory=dict)

    REQUIRED_COLUMNS = {"compound_id", "concentration", "response"}

    def __post_init__(self) -> None:
        self.table = self.table.copy()
        self._normalize_columns()
        self.validate()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        concentration_unit: str | None = None,
        response_unit: str | None = None,
        metadata: dict | None = None,
    ) -> "DoseResponseData":
        """Create a validated data object from a long-form DataFrame."""
        return cls(
            table=df,
            concentration_unit=concentration_unit,
            response_unit=response_unit,
            metadata=metadata or {},
        )

    @classmethod
    def from_wide_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        compound_col: str,
        concentration_col: str,
        replicate_cols: list[str],
        experiment_col: str | None = None,
        concentration_unit: str | None = None,
        response_unit: str | None = None,
        metadata: dict | None = None,
    ) -> "DoseResponseData":
        """Create data from a wide table with one column per replicate."""
        id_vars = [compound_col, concentration_col]
        if experiment_col is not None:
            id_vars.append(experiment_col)

        long = df.melt(
            id_vars=id_vars,
            value_vars=replicate_cols,
            var_name="replicate_id",
            value_name="response",
        ).dropna(subset=["response"])

        long = long.rename(
            columns={
                compound_col: "compound_id",
                concentration_col: "concentration",
            }
        )
        if experiment_col is not None:
            long = long.rename(columns={experiment_col: "experiment_id"})

        return cls(
            table=long,
            concentration_unit=concentration_unit,
            response_unit=response_unit,
            metadata=metadata or {},
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        concentration_unit: str | None = None,
        response_unit: str | None = None,
        metadata: dict | None = None,
        **read_csv_kwargs,
    ) -> "DoseResponseData":
        """Create data from a long-form CSV file."""
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_dataframe(
            df,
            concentration_unit=concentration_unit,
            response_unit=response_unit,
            metadata=metadata,
        )

    @property
    def compounds(self) -> list[str]:
        """Return sorted compound identifiers."""
        return sorted(self.table["compound_id"].astype(str).unique())

    def select_compound(self, compound_id: str) -> CompoundData:
        """Return a view containing observations for one compound."""
        selected = self.table[self.table["compound_id"].astype(str) == str(compound_id)]
        if selected.empty:
            raise KeyError(f"Compound {compound_id!r} not found.")
        return CompoundData(
            compound_id=str(compound_id),
            table=selected.copy(),
            concentration_unit=self.concentration_unit,
            response_unit=self.response_unit,
        )

    def _normalize_columns(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self.table.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if "experiment_id" not in self.table.columns:
            self.table["experiment_id"] = "experiment_1"
        if "replicate_id" not in self.table.columns:
            self.table["replicate_id"] = (
                self.table.groupby(["compound_id", "experiment_id", "concentration"])
                .cumcount()
                .add(1)
                .astype(str)
                .radd("replicate_")
            )

        self.table["compound_id"] = self.table["compound_id"].astype(str)
        self.table["experiment_id"] = self.table["experiment_id"].astype(str)
        self.table["replicate_id"] = self.table["replicate_id"].astype(str)
        self.table["concentration"] = pd.to_numeric(
            self.table["concentration"], errors="raise"
        )
        self.table["response"] = pd.to_numeric(self.table["response"], errors="raise")

    def validate(self) -> None:
        """Validate the dose-response data schema and basic numerical assumptions."""
        if self.table.empty:
            raise ValueError("Dose-response table is empty.")
        if self.table["compound_id"].isna().any():
            raise ValueError("compound_id contains missing values.")
        if self.table["experiment_id"].isna().any():
            raise ValueError("experiment_id contains missing values.")
        if self.table["concentration"].isna().any():
            raise ValueError("concentration contains missing values.")
        if self.table["response"].isna().any():
            raise ValueError("response contains missing values.")
        if (self.table["concentration"] <= 0).any():
            raise ValueError("All concentrations must be positive.")
