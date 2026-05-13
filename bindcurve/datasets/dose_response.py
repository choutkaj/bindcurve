from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DataFormat = Literal["long", "wide"]


@dataclass
class CompoundData:
    """View of dose-response observations for one compound."""

    compound_id: str
    table: pd.DataFrame

    @property
    def experiments(self) -> list[str]:
        """Return sorted experiment identifiers for this compound."""
        return sorted(self.table["experiment_id"].astype(str).unique())

    @property
    def concentrations(self) -> np.ndarray:
        """Return sorted unique concentrations for this compound."""
        return np.sort(self.table["concentration"].unique())

    def select_experiment(self, experiment_id: str) -> CompoundData:
        """Return a view containing only one independent experiment."""
        selected = self.table[
            self.table["experiment_id"].astype(str) == str(experiment_id)
        ]
        if selected.empty:
            raise KeyError(
                f"Experiment {experiment_id!r} not found for compound "
                f"{self.compound_id!r}."
            )
        return CompoundData(
            compound_id=self.compound_id,
            table=selected.copy(),
        )

    def aggregate_replicates(
        self,
    ) -> pd.DataFrame:
        """Aggregate technical replicates by concentration using the arithmetic mean."""
        grouped = self.table.groupby("concentration", as_index=False)["response"]
        response = grouped.mean().rename(columns={"response": "response"})

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return concentration and response arrays for fitting."""
        if aggregate_replicates:
            table = self.aggregate_replicates()
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
        format: DataFormat = "long",
        compound_col: str = "compound_id",
        concentration_col: str = "concentration",
        response_col: str = "response",
        experiment_col: str = "experiment_id",
        replicate_col: str = "replicate_id",
        replicate_cols: list[str] | None = None,
        replicate_prefix: str = "response_",
        metadata: dict | None = None,
    ) -> DoseResponseData:
        """Create a validated data object from a long- or wide-form DataFrame."""
        normalized_format = cls._normalize_format(format)

        if normalized_format == "long":
            table = cls._standardize_long_dataframe_columns(
                df,
                compound_col=compound_col,
                concentration_col=concentration_col,
                response_col=response_col,
                experiment_col=experiment_col,
                replicate_col=replicate_col,
            )
            return cls(
                table=table,
                metadata=metadata or {},
            )

        table = cls._wide_to_long_dataframe(
            df,
            compound_col=compound_col,
            concentration_col=concentration_col,
            experiment_col=experiment_col if experiment_col in df.columns else None,
            replicate_cols=replicate_cols,
            replicate_prefix=replicate_prefix,
        )
        return cls(
            table=table,
            metadata=metadata or {},
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        format: DataFormat = "long",
        compound_col: str = "compound_id",
        concentration_col: str = "concentration",
        response_col: str = "response",
        experiment_col: str = "experiment_id",
        replicate_col: str = "replicate_id",
        replicate_prefix: str = "response_",
        metadata: dict | None = None,
        **read_csv_kwargs,
    ) -> DoseResponseData:
        """Create data from a CSV file.

        Supported formats are ``"long"`` and ``"wide"``.
        """
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_dataframe(
            df,
            format=format,
            compound_col=compound_col,
            concentration_col=concentration_col,
            response_col=response_col,
            experiment_col=experiment_col,
            replicate_col=replicate_col,
            replicate_prefix=replicate_prefix,
            metadata=metadata,
        )

    @classmethod
    def from_json(
        cls,
        source: str | Path,
        *,
        format: DataFormat | None = None,
        compound_col: str = "compound_id",
        concentration_col: str = "concentration",
        response_col: str = "response",
        experiment_col: str = "experiment_id",
        replicate_col: str = "replicate_id",
        replicate_cols: list[str] | None = None,
        replicate_prefix: str = "response_",
        metadata: dict | None = None,
    ) -> DoseResponseData:
        """Create data from a JSON string or JSON file."""
        text = cls._read_json_source(source)
        payload = json.loads(text)
        payload_metadata: dict = {}
        payload_format: str | None = None

        if isinstance(payload, dict) and "table" in payload:
            table_payload = payload["table"]
            payload_format = payload.get("format")
            payload_metadata = payload.get("metadata") or {}
            if not isinstance(payload_metadata, dict):
                raise ValueError("JSON metadata must be an object/dictionary.")
        else:
            table_payload = payload

        resolved_format = cls._resolve_json_format(
            requested_format=format,
            payload_format=payload_format,
        )
        table = pd.DataFrame(table_payload)
        resolved_metadata = payload_metadata.copy()
        if metadata:
            resolved_metadata.update(metadata)
        return cls.from_dataframe(
            table,
            format=resolved_format,
            compound_col=compound_col,
            concentration_col=concentration_col,
            response_col=response_col,
            experiment_col=experiment_col,
            replicate_col=replicate_col,
            replicate_cols=replicate_cols,
            replicate_prefix=replicate_prefix,
            metadata=resolved_metadata,
        )

    def to_dataframe(
        self,
        *,
        format: DataFormat = "long",
        compound_col: str = "compound_id",
        concentration_col: str = "concentration",
        response_col: str = "response",
        experiment_col: str = "experiment_id",
        replicate_col: str = "replicate_id",
        replicate_prefix: str = "response_",
    ) -> pd.DataFrame:
        """Serialize the data object to a long- or wide-form DataFrame."""
        normalized_format = self._normalize_format(format)

        if normalized_format == "long":
            rename_map = {
                "compound_id": compound_col,
                "concentration": concentration_col,
                "response": response_col,
                "experiment_id": experiment_col,
                "replicate_id": replicate_col,
            }
            return self.table.rename(
                columns={
                    source: target
                    for source, target in rename_map.items()
                    if source != target
                }
            ).copy()

        table = self.table[
            [
                "compound_id",
                "experiment_id",
                "concentration",
                "response",
            ]
        ].copy()
        group_cols = ["compound_id", "experiment_id", "concentration"]
        table["__replicate_position"] = table.groupby(group_cols).cumcount().add(1)
        wide = table.pivot(
            index=group_cols,
            columns="__replicate_position",
            values="response",
        ).reset_index()
        wide = wide.rename(
            columns={
                column: f"{replicate_prefix}{int(column)}"
                for column in wide.columns
                if column not in group_cols
            }
        )
        wide.columns.name = None
        return wide.rename(
            columns={
                "compound_id": compound_col,
                "experiment_id": experiment_col,
                "concentration": concentration_col,
            }
        )

    def to_csv(
        self,
        path: str | Path | None = None,
        *,
        format: DataFormat = "long",
        compound_col: str = "compound_id",
        concentration_col: str = "concentration",
        response_col: str = "response",
        experiment_col: str = "experiment_id",
        replicate_col: str = "replicate_id",
        replicate_prefix: str = "response_",
        index: bool = False,
        **to_csv_kwargs,
    ) -> str | None:
        """Serialize the data object to CSV."""
        table = self.to_dataframe(
            format=format,
            compound_col=compound_col,
            concentration_col=concentration_col,
            response_col=response_col,
            experiment_col=experiment_col,
            replicate_col=replicate_col,
            replicate_prefix=replicate_prefix,
        )
        return table.to_csv(path_or_buf=path, index=index, **to_csv_kwargs)

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        format: DataFormat = "long",
        compound_col: str = "compound_id",
        concentration_col: str = "concentration",
        response_col: str = "response",
        experiment_col: str = "experiment_id",
        replicate_col: str = "replicate_id",
        replicate_prefix: str = "response_",
        **json_kwargs,
    ) -> str | None:
        """Serialize the data object to JSON."""
        normalized_format = self._normalize_format(format)
        payload = {
            "format": normalized_format,
            "metadata": self.metadata,
            "table": self.to_dataframe(
                format=normalized_format,
                compound_col=compound_col,
                concentration_col=concentration_col,
                response_col=response_col,
                experiment_col=experiment_col,
                replicate_col=replicate_col,
                replicate_prefix=replicate_prefix,
            ).to_dict(orient="records"),
        }
        json_text = json.dumps(payload, **json_kwargs)

        if path is None:
            return json_text

        Path(path).write_text(json_text, encoding="utf-8")
        return None

    @classmethod
    def _standardize_long_dataframe_columns(
        cls,
        df: pd.DataFrame,
        *,
        compound_col: str,
        concentration_col: str,
        response_col: str,
        experiment_col: str,
        replicate_col: str,
    ) -> pd.DataFrame:
        """Rename user-provided long-form columns to the canonical schema."""
        cls._require_columns(
            df,
            columns=[compound_col, concentration_col, response_col],
            format_name="long",
        )
        rename_map = {
            compound_col: "compound_id",
            concentration_col: "concentration",
            response_col: "response",
        }
        if experiment_col in df.columns:
            rename_map[experiment_col] = "experiment_id"
        if replicate_col in df.columns:
            rename_map[replicate_col] = "replicate_id"
        return df.rename(
            columns={
                source: target
                for source, target in rename_map.items()
                if source != target
            }
        )

    @classmethod
    def _wide_to_long_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        compound_col: str,
        concentration_col: str,
        experiment_col: str | None,
        replicate_cols: list[str] | None,
        replicate_prefix: str,
    ) -> pd.DataFrame:
        """Normalize a wide response table to the canonical long schema."""
        cls._require_columns(
            df,
            columns=[compound_col, concentration_col],
            format_name="wide",
        )
        selected_replicate_cols = cls._resolve_wide_replicate_columns(
            df,
            replicate_cols=replicate_cols,
            replicate_prefix=replicate_prefix,
        )

        id_vars = [compound_col, concentration_col]
        if experiment_col is not None:
            id_vars.append(experiment_col)

        long = df.melt(
            id_vars=id_vars,
            value_vars=selected_replicate_cols,
            var_name="replicate_id",
            value_name="response",
        ).dropna(subset=["response"])

        rename_map = {
            compound_col: "compound_id",
            concentration_col: "concentration",
        }
        if experiment_col is not None:
            rename_map[experiment_col] = "experiment_id"
        return long.rename(columns=rename_map)

    @classmethod
    def _resolve_wide_replicate_columns(
        cls,
        df: pd.DataFrame,
        *,
        replicate_cols: list[str] | None,
        replicate_prefix: str,
    ) -> list[str]:
        """Resolve or discover the response columns of a wide input table."""
        if replicate_cols is not None:
            cls._require_columns(
                df,
                columns=replicate_cols,
                format_name="wide",
            )
            return list(replicate_cols)

        discovered = [
            column for column in df.columns if str(column).startswith(replicate_prefix)
        ]
        if not discovered:
            raise ValueError(
                "No technical replicate columns found for wide "
                f"format. Expected at least one column starting with "
                f"{replicate_prefix!r}."
            )
        return discovered

    @classmethod
    def _normalize_format(cls, format: str) -> str:
        """Normalize a user-provided serialization format name."""
        normalized_format = format.replace("-", "_").lower()
        if normalized_format not in {"long", "wide"}:
            raise ValueError("format must be 'long' or 'wide'.")
        return normalized_format

    @classmethod
    def _resolve_json_format(
        cls,
        *,
        requested_format: str | None,
        payload_format: str | None,
    ) -> str:
        """Resolve the effective format for JSON deserialization."""
        normalized_requested = (
            cls._normalize_format(requested_format)
            if requested_format is not None
            else None
        )
        normalized_payload = (
            cls._normalize_format(payload_format)
            if payload_format is not None
            else None
        )

        if normalized_requested and normalized_payload:
            if normalized_requested != normalized_payload:
                raise ValueError(
                    "Requested format does not match the JSON payload format."
                )
            return normalized_requested
        if normalized_payload is not None:
            return normalized_payload
        if normalized_requested is not None:
            return normalized_requested
        return "long"

    @staticmethod
    def _read_json_source(source: str | Path) -> str:
        """Return JSON text from a string payload or JSON file."""
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8")

        stripped = source.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return source

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {source}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        *,
        columns: list[str],
        format_name: str,
    ) -> None:
        """Raise a clear error if required input columns are missing."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {format_name} format: {missing}"
            )

    @property
    def compounds(self) -> list[str]:
        """Return sorted compound identifiers."""
        return sorted(self.table["compound_id"].astype(str).unique())

    def select_compound(self, compound_id: str) -> CompoundData:
        """Return a view containing observations for one compound."""
        selected = self.table[
            self.table["compound_id"].astype(str) == str(compound_id)
        ]
        if selected.empty:
            raise KeyError(f"Compound {compound_id!r} not found.")
        return CompoundData(
            compound_id=str(compound_id),
            table=selected.copy(),
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
