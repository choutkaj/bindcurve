from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from bindcurve.quality import (
    DataQualityThresholds,
    resolve_requested_compounds,
    summarize_quality_flags,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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

    def summary(self) -> pd.DataFrame:
        """Represent compound-level dataset summaries as a DataFrame."""
        rows: list[dict[str, object]] = []

        for compound_id in self.compounds:
            compound_table = self.select_compound(compound_id).table

            rows.append(
                {
                    "compound_id": compound_id,
                    "N_exp": int(compound_table["experiment_id"].nunique()),
                    "N_obs": int(len(compound_table)),
                    "N_conc_total": int(compound_table["concentration"].nunique()),
                    "concentration_min": float(compound_table["concentration"].min()),
                    "concentration_max": float(compound_table["concentration"].max()),
                    "response_min": float(compound_table["response"].min()),
                    "response_max": float(compound_table["response"].max()),
                }
            )

        return pd.DataFrame(rows)

    def keep_only(
        self,
        selectors: str | int | Iterable[str | int],
    ) -> DoseResponseData:
        """Return a new dataset containing only the selected compounds."""
        selected_compounds = self._resolve_compound_selectors(selectors)
        filtered = self.table[
            self.table["compound_id"].astype(str).isin(selected_compounds)
        ].copy()
        if filtered.empty:
            raise ValueError("Filtering removed all compounds from the dataset.")
        return type(self)(
            table=filtered,
            metadata=self.metadata.copy(),
        )

    def remove(
        self,
        selectors: str | int | Iterable[str | int],
    ) -> DoseResponseData:
        """Return a new dataset with the selected compounds removed."""
        removed_compounds = set(self._resolve_compound_selectors(selectors))
        filtered = self.table[
            ~self.table["compound_id"].astype(str).isin(removed_compounds)
        ].copy()
        if filtered.empty:
            raise ValueError("Filtering removed all compounds from the dataset.")
        return type(self)(
            table=filtered,
            metadata=self.metadata.copy(),
        )

    @classmethod
    def concatenate(
        cls,
        *datasets: DoseResponseData,
    ) -> DoseResponseData:
        """Return a new dataset created by concatenating multiple datasets."""
        if len(datasets) < 2:
            raise ValueError("concatenate() requires at least 2 datasets.")
        for dataset in datasets:
            if not isinstance(dataset, DoseResponseData):
                raise TypeError(
                    "concatenate() accepts only DoseResponseData objects."
                )

        _validate_concatenation_inputs(datasets)
        combined = pd.concat(
            [dataset.table for dataset in datasets],
            ignore_index=True,
        )
        return cls(
            table=combined,
            metadata={},
        )

    def quality_report(
        self,
        *,
        compounds: str | Iterable[str] | None = None,
        thresholds: DataQualityThresholds | None = None,
    ) -> pd.DataFrame:
        """Return compound-level quality-control metrics for the dataset."""
        resolved_thresholds = thresholds or DataQualityThresholds()
        selected_compounds = resolve_requested_compounds(self.compounds, compounds)
        if not selected_compounds:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for compound_id in selected_compounds:
            compound_table = self.select_compound(compound_id).table
            rows.append(
                _compound_data_quality_row(
                    compound_table,
                    thresholds=resolved_thresholds,
                )
            )

        columns = [
            "compound_id",
            "status",
            "N_flag_orange",
            "N_flag_red",
            "flags",
            "N_exp",
            "N_obs",
            "N_conc_union",
            "N_conc_min",
            "N_conc_median",
            "N_conc_max",
            "grid_coverage",
            "N_rep_min",
            "N_rep_median",
            "N_rep_max",
            "single_replicate_fraction",
            "intra_noise_median_frac_range",
            "intra_noise_p90_frac_range",
            "nonpositive_concentration_count",
        ]
        return pd.DataFrame(rows, columns=columns)

    def quality_dashboard(
        self,
        *,
        compounds: str | Iterable[str] | None = None,
        thresholds: DataQualityThresholds | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Return a graphical dashboard summarizing data-level QC."""
        from bindcurve.plotting.quality import plot_data_quality_dashboard

        return plot_data_quality_dashboard(
            self,
            compounds=compounds,
            thresholds=thresholds,
            figsize=figsize,
        )

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

    def _resolve_compound_selectors(
        self,
        selectors: str | int | Iterable[str | int],
    ) -> list[str]:
        available_compounds = self.compounds
        available_set = set(available_compounds)
        resolved: list[str] = []
        seen: set[str] = set()

        for selector in _coerce_compound_selectors(selectors):
            if isinstance(selector, str):
                compound_id = selector
                if compound_id not in available_set:
                    raise KeyError(f"Compound {compound_id!r} not found.")
            elif _is_compound_index(selector):
                index = int(selector)
                if index < 0:
                    index += len(available_compounds)
                if index < 0 or index >= len(available_compounds):
                    raise IndexError(
                        f"Compound index {int(selector)} is out of range."
                    )
                compound_id = available_compounds[index]
            else:
                raise TypeError(
                    "Compound selectors must be strings or integers."
                )

            if compound_id not in seen:
                seen.add(compound_id)
                resolved.append(compound_id)

        return resolved

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


def _compound_data_quality_row(
    compound_table: pd.DataFrame,
    *,
    thresholds: DataQualityThresholds,
) -> dict[str, object]:
    compound_id = str(compound_table["compound_id"].iloc[0])
    experiment_concentration = compound_table.groupby(
        ["experiment_id", "concentration"],
        as_index=False,
    )["response"].agg(response_sd="std", n_replicates="count")
    concentration_counts = (
        experiment_concentration.groupby("experiment_id")["concentration"].count()
    )
    observed_cells = len(experiment_concentration)
    N_exp = int(compound_table["experiment_id"].nunique())
    N_conc_union = int(compound_table["concentration"].nunique())
    expected_cells = N_exp * N_conc_union
    grid_coverage = (
        float(observed_cells / expected_cells) if expected_cells > 0 else np.nan
    )
    single_replicate_fraction = float(
        (experiment_concentration["n_replicates"] < 2).mean()
    )

    experiment_means = (
        compound_table.groupby(
            ["experiment_id", "concentration"],
            as_index=False,
        )["response"]
        .mean()
        .rename(columns={"response": "response_mean"})
    )
    experiment_ranges = (
        experiment_means.groupby("experiment_id")["response_mean"]
        .agg(lambda values: float(values.max() - values.min()))
        .rename("experiment_response_range")
        .reset_index()
    )
    experiment_concentration = experiment_concentration.merge(
        experiment_ranges,
        on="experiment_id",
        how="left",
    )
    valid_noise = experiment_concentration[
        (experiment_concentration["n_replicates"] >= 2)
        & experiment_concentration["response_sd"].notna()
        & (experiment_concentration["experiment_response_range"] > 0.0)
    ].copy()
    valid_noise["response_sd_frac_range"] = (
        valid_noise["response_sd"] / valid_noise["experiment_response_range"]
    )
    intra_noise_values = valid_noise["response_sd_frac_range"].to_numpy(dtype=float)

    flags: list[tuple[str, str]] = []
    nonpositive_concentration_count = int(
        (compound_table["concentration"] <= 0.0).sum()
    )
    if nonpositive_concentration_count > 0:
        flags.append(("red", "nonpositive concentration values present"))
    if N_exp < 2:
        flags.append(("red", "fewer than 2 independent experiments"))
    elif N_exp < thresholds.min_experiments_green:
        flags.append(("orange", f"only {N_exp} independent experiments"))
    if grid_coverage < 1.0:
        flags.append(("orange", "incomplete experiment-concentration grid"))
    if single_replicate_fraction > 0.0:
        flags.append(("orange", "single-replicate concentration cells present"))

    invalid_range_count = int(
        (experiment_ranges["experiment_response_range"] <= 0.0).sum()
    )
    if invalid_range_count > 0:
        flags.append(
            (
                "orange",
                "nonpositive experiment response range prevented full "
                "intra-noise evaluation",
            )
        )

    intra_noise_median = _nan_percentile(intra_noise_values, 50.0)
    intra_noise_p90 = _nan_percentile(intra_noise_values, 90.0)
    _append_threshold_flag(
        flags,
        value=intra_noise_median,
        orange_threshold=thresholds.max_intra_noise_median_frac_range_orange,
        red_threshold=thresholds.max_intra_noise_median_frac_range_red,
        orange_message=(
            "elevated median intra-experiment noise relative to response range"
        ),
        red_message="high median intra-experiment noise relative to response range",
    )
    _append_threshold_flag(
        flags,
        value=intra_noise_p90,
        orange_threshold=thresholds.max_intra_noise_p90_frac_range_orange,
        red_threshold=thresholds.max_intra_noise_p90_frac_range_red,
        orange_message=(
            "elevated upper-tail intra-experiment noise relative to response range"
        ),
        red_message="high upper-tail intra-experiment noise relative to response range",
    )

    status, N_flag_orange, N_flag_red, flag_text = summarize_quality_flags(flags)
    return {
        "compound_id": compound_id,
        "status": status,
        "N_flag_orange": N_flag_orange,
        "N_flag_red": N_flag_red,
        "flags": flag_text,
        "N_exp": N_exp,
        "N_obs": int(len(compound_table)),
        "N_conc_union": N_conc_union,
        "N_conc_min": int(concentration_counts.min()),
        "N_conc_median": float(concentration_counts.median()),
        "N_conc_max": int(concentration_counts.max()),
        "grid_coverage": grid_coverage,
        "N_rep_min": int(experiment_concentration["n_replicates"].min()),
        "N_rep_median": float(experiment_concentration["n_replicates"].median()),
        "N_rep_max": int(experiment_concentration["n_replicates"].max()),
        "single_replicate_fraction": single_replicate_fraction,
        "intra_noise_median_frac_range": intra_noise_median,
        "intra_noise_p90_frac_range": intra_noise_p90,
        "nonpositive_concentration_count": nonpositive_concentration_count,
    }


def _nan_percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.nanpercentile(values, percentile))


def _append_threshold_flag(
    flags: list[tuple[str, str]],
    *,
    value: float | None,
    orange_threshold: float,
    red_threshold: float,
    orange_message: str,
    red_message: str,
) -> None:
    if value is None:
        return
    if value > red_threshold:
        flags.append(("red", red_message))
    elif value > orange_threshold:
        flags.append(("orange", orange_message))


def _coerce_compound_selectors(
    selectors: str | int | Iterable[str | int],
) -> list[str | int]:
    if isinstance(selectors, str) or _is_compound_index(selectors):
        return [selectors]
    if not isinstance(selectors, Iterable):
        raise TypeError(
            "Compound selectors must be strings or integers, or iterables of them."
        )
    return list(selectors)


def _is_compound_index(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_concatenation_inputs(
    datasets: Iterable[DoseResponseData],
) -> None:
    seen_experiments_by_compound: dict[str, set[str]] = {}

    for dataset in datasets:
        experiment_keys = dataset.table[
            ["compound_id", "experiment_id"]
        ].drop_duplicates()
        for row in experiment_keys.itertuples(index=False):
            compound_id = str(row.compound_id)
            experiment_id = str(row.experiment_id)
            seen = seen_experiments_by_compound.setdefault(compound_id, set())
            if experiment_id in seen:
                raise ValueError(
                    "Cannot concatenate datasets because compound "
                    f"{compound_id!r} reuses experiment_id {experiment_id!r}."
                )
            seen.add(experiment_id)
