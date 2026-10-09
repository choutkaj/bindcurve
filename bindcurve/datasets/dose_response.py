from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from functools import cached_property
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from bindcurve.datasets.aggregation import aggregate_responses
from bindcurve.datasets.formats import (
    DataFormat,
    _normalize_format,
    _read_json_source,
    _resolve_json_format,
    _serialize_dataframe,
    _standardize_long_dataframe_columns,
    _wide_to_long_dataframe,
)


class CompoundData:
    """View of dose-response observations for one compound."""

    compound_id: str
    _table: pd.DataFrame

    def __init__(self, compound_id: str, table: pd.DataFrame) -> None:
        self.compound_id = str(compound_id)
        self._table = table.copy()

    @property
    def table(self) -> pd.DataFrame:
        """Return an isolated copy of this view's canonical table."""
        return self._table.copy()

    @property
    def experiments(self) -> list[str]:
        """Return sorted experiment identifiers for this compound."""
        return sorted(self._table["experiment_id"].astype(str).unique())

    @property
    def concentrations(self) -> np.ndarray:
        """Return sorted unique concentrations for this compound."""
        return np.sort(self._table["concentration"].unique())

    def select_experiment(self, experiment_id: str) -> CompoundData:
        """Return a view containing only one independent experiment."""
        selected = self._table[
            self._table["experiment_id"].astype(str) == str(experiment_id)
        ]
        if selected.empty:
            raise KeyError(
                f"Experiment {experiment_id!r} not found for compound "
                f"{self.compound_id!r}."
            )
        return CompoundData(
            compound_id=self.compound_id,
            table=selected,
        )

    def aggregate_replicates(
        self,
    ) -> pd.DataFrame:
        """Aggregate technical replicates by concentration using the arithmetic mean."""
        return self._replicate_summary.copy()

    @cached_property
    def _replicate_summary(self) -> pd.DataFrame:
        return aggregate_responses(
            self._table, by=["concentration"], count_name="n_replicates"
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
            table = self._table.sort_values("concentration")
        return (
            table["concentration"].to_numpy(dtype=float),
            table["response"].to_numpy(dtype=float),
        )

    def fit_observations(self) -> pd.DataFrame:
        """Aggregate responses and any known observation uncertainty for fitting.

        Technical-replicate responses retain their arithmetic mean. When each
        replicate has a known standard deviation, the standard deviation of
        that arithmetic mean is propagated exactly under independent errors.
        """
        if "sigma" not in self._table.columns:
            return self.aggregate_replicates()

        rows: list[dict[str, float | int]] = []
        for concentration, group in self._table.groupby(
            "concentration",
            sort=True,
        ):
            sigma = group["sigma"].to_numpy(dtype=float)
            n_replicates = len(group)
            rows.append(
                {
                    "concentration": float(concentration),
                    "response": float(group["response"].mean()),
                    "sigma": float(np.sqrt(np.sum(sigma**2)) / n_replicates),
                    "n_replicates": n_replicates,
                }
            )
        return pd.DataFrame(rows)


class DoseResponseData:
    """Validated long-form dose-response observations."""

    _table: pd.DataFrame
    _metadata: dict

    REQUIRED_COLUMNS = {"compound_id", "concentration", "response"}

    def __init__(
        self,
        table: pd.DataFrame,
        metadata: dict | None = None,
    ) -> None:
        self._table = table.copy()
        self._metadata = deepcopy(metadata or {})
        self._normalize_columns()
        self.validate()

    @property
    def table(self) -> pd.DataFrame:
        """Return an isolated copy of the validated canonical table."""
        return self._table.copy()

    @property
    def metadata(self) -> dict:
        """Return an isolated copy of dataset metadata."""
        return deepcopy(self._metadata)

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
        sigma_col: str | None = "sigma",
        replicate_cols: list[str] | None = None,
        replicate_prefix: str = "response_",
        metadata: dict | None = None,
    ) -> DoseResponseData:
        """Create a validated data object from a long- or wide-form DataFrame."""
        normalized_format = _normalize_format(format)

        if normalized_format == "long":
            table = _standardize_long_dataframe_columns(
                df,
                compound_col=compound_col,
                concentration_col=concentration_col,
                response_col=response_col,
                experiment_col=experiment_col,
                replicate_col=replicate_col,
                sigma_col=sigma_col,
            )
            return cls(
                table=table,
                metadata=metadata or {},
            )

        table = _wide_to_long_dataframe(
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
        sigma_col: str | None = "sigma",
        replicate_cols: list[str] | None = None,
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
            sigma_col=sigma_col,
            replicate_cols=replicate_cols,
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
        sigma_col: str | None = "sigma",
        replicate_cols: list[str] | None = None,
        replicate_prefix: str = "response_",
        metadata: dict | None = None,
    ) -> DoseResponseData:
        """Create data from a JSON string or JSON file."""
        text = _read_json_source(source)
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

        resolved_format = _resolve_json_format(
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
            sigma_col=sigma_col,
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
        return _serialize_dataframe(
            self._table,
            format=format,
            compound_col=compound_col,
            concentration_col=concentration_col,
            response_col=response_col,
            experiment_col=experiment_col,
            replicate_col=replicate_col,
            replicate_prefix=replicate_prefix,
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
        normalized_format = _normalize_format(format)
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

        for compound_id, compound_table in self._table.groupby(
            "compound_id", sort=True
        ):
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
        filtered = self._table[
            self._table["compound_id"].astype(str).isin(selected_compounds)
        ]
        if filtered.empty:
            raise ValueError("Filtering removed all compounds from the dataset.")
        return type(self)(
            table=filtered,
            metadata=self._metadata,
        )

    def remove(
        self,
        selectors: str | int | Iterable[str | int],
    ) -> DoseResponseData:
        """Return a new dataset with the selected compounds removed."""
        removed_compounds = set(self._resolve_compound_selectors(selectors))
        filtered = self._table[
            ~self._table["compound_id"].astype(str).isin(removed_compounds)
        ]
        if filtered.empty:
            raise ValueError("Filtering removed all compounds from the dataset.")
        return type(self)(
            table=filtered,
            metadata=self._metadata,
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
                raise TypeError("concatenate() accepts only DoseResponseData objects.")

        _validate_concatenation_inputs(datasets)
        first_metadata = datasets[0].metadata
        if any(dataset.metadata != first_metadata for dataset in datasets[1:]):
            raise ValueError("Cannot concatenate datasets with different metadata.")
        combined = pd.concat(
            [dataset.table for dataset in datasets],
            ignore_index=True,
        )
        return cls(
            table=combined,
            metadata=first_metadata,
        )

    @property
    def compounds(self) -> list[str]:
        """Return sorted compound identifiers."""
        return sorted(self._table["compound_id"].astype(str).unique())

    def resolve_compounds(
        self,
        selectors: str | int | Iterable[str | int] | None = None,
    ) -> list[str]:
        """Resolve, validate, and stably deduplicate compound selectors."""
        if selectors is None:
            return self.compounds
        return self._resolve_compound_selectors(selectors)

    def select_compound(self, compound_id: str) -> CompoundData:
        """Return a view containing observations for one compound."""
        selected = self._table[
            self._table["compound_id"].astype(str) == str(compound_id)
        ]
        if selected.empty:
            raise KeyError(f"Compound {compound_id!r} not found.")
        return CompoundData(
            compound_id=str(compound_id),
            table=selected,
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
                    raise IndexError(f"Compound index {int(selector)} is out of range.")
                compound_id = available_compounds[index]
            else:
                raise TypeError("Compound selectors must be strings or integers.")

            if compound_id not in seen:
                seen.add(compound_id)
                resolved.append(compound_id)

        return resolved

    def _normalize_columns(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self._table.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if "experiment_id" not in self._table.columns:
            self._table["experiment_id"] = "experiment_1"
        self._validate_identifier_values(("compound_id", "experiment_id"))

        if "replicate_id" not in self._table.columns:
            self._table["replicate_id"] = (
                self._table.groupby(["compound_id", "experiment_id", "concentration"])
                .cumcount()
                .add(1)
                .astype(str)
                .radd("replicate_")
            )
        self._validate_identifier_values(("replicate_id",))

        self._table["compound_id"] = self._table["compound_id"].astype(str)
        self._table["experiment_id"] = self._table["experiment_id"].astype(str)
        self._table["replicate_id"] = self._table["replicate_id"].astype(str)
        self._table["concentration"] = pd.to_numeric(
            self._table["concentration"], errors="raise"
        )
        self._table["response"] = pd.to_numeric(self._table["response"], errors="raise")
        if "sigma" in self._table.columns:
            self._table["sigma"] = pd.to_numeric(self._table["sigma"], errors="raise")

    def validate(self) -> None:
        """Validate the dose-response data schema and basic numerical assumptions."""
        if self._table.empty:
            raise ValueError("Dose-response table is empty.")
        concentration = self._table["concentration"].to_numpy(dtype=float)
        response = self._table["response"].to_numpy(dtype=float)
        if np.any(~np.isfinite(concentration)):
            raise ValueError("concentration must contain only finite values.")
        if np.any(~np.isfinite(response)):
            raise ValueError("response must contain only finite values.")
        if np.any(concentration <= 0.0):
            raise ValueError("All concentrations must be positive.")
        if "weight" in self._table.columns:
            # Weights are ambiguous (1/sigma, 1/sigma**2, or relative), so only
            # known standard deviations are accepted.
            raise ValueError(
                "A 'weight' column is not supported; provide the known observation "
                "standard deviation as 'sigma' instead."
            )
        if "sigma" in self._table.columns:
            sigma = self._table["sigma"].to_numpy(dtype=float)
            if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
                raise ValueError("sigma must contain finite positive values.")

        key = [
            "compound_id",
            "experiment_id",
            "concentration",
            "replicate_id",
        ]
        duplicated = self._table.duplicated(key, keep=False)
        if duplicated.any():
            raise ValueError(
                "Duplicate observations share the same compound, experiment, "
                "concentration, and replicate identifiers."
            )

    def _validate_identifier_values(self, columns: tuple[str, ...]) -> None:
        for column in columns:
            values = self._table[column]
            if values.isna().any():
                raise ValueError(f"{column} contains missing values.")
            strings = values.astype("string")
            if strings.str.strip().eq("").any():
                raise ValueError(f"{column} contains blank values.")


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
