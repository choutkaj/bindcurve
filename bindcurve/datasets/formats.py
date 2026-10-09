"""Private tabular layout and JSON-source adapters for dose-response data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

DataFormat = Literal["long", "wide"]


def _standardize_long_dataframe_columns(
    df: pd.DataFrame,
    *,
    compound_col: str,
    concentration_col: str,
    response_col: str,
    experiment_col: str,
    replicate_col: str,
    sigma_col: str | None,
) -> pd.DataFrame:
    """Rename user-provided long-form columns to the canonical schema."""
    _require_columns(
        df,
        columns=[compound_col, concentration_col, response_col],
        format_name="long",
    )
    role_columns = [compound_col, concentration_col, response_col]
    role_columns.extend(
        column
        for column in (experiment_col, replicate_col, sigma_col)
        if column is not None and column in df.columns
    )
    if len(role_columns) != len(set(role_columns)):
        raise ValueError("Input column roles must use distinct source columns.")
    rename_map = {
        compound_col: "compound_id",
        concentration_col: "concentration",
        response_col: "response",
    }
    if experiment_col in df.columns:
        rename_map[experiment_col] = "experiment_id"
    if replicate_col in df.columns:
        rename_map[replicate_col] = "replicate_id"
    if sigma_col is not None and sigma_col in df.columns:
        rename_map[sigma_col] = "sigma"
    for source, target in rename_map.items():
        if source != target and target in df.columns:
            raise ValueError(
                f"Renaming {source!r} to {target!r} would create duplicate "
                "canonical columns."
            )
    return df.rename(
        columns={
            source: target for source, target in rename_map.items() if source != target
        }
    )


def _wide_to_long_dataframe(
    df: pd.DataFrame,
    *,
    compound_col: str,
    concentration_col: str,
    experiment_col: str | None,
    replicate_cols: list[str] | None,
    replicate_prefix: str,
) -> pd.DataFrame:
    """Normalize a wide response table to the canonical long schema."""
    _require_columns(
        df,
        columns=[compound_col, concentration_col],
        format_name="wide",
    )
    selected_replicate_cols = _resolve_wide_replicate_columns(
        df,
        replicate_cols=replicate_cols,
        replicate_prefix=replicate_prefix,
    )

    id_vars = [compound_col, concentration_col]
    if experiment_col is not None:
        id_vars.append(experiment_col)
    unsupported_columns = set(df.columns) - set(id_vars) - set(selected_replicate_cols)
    if unsupported_columns:
        raise ValueError(
            "Wide input contains unsupported non-replicate columns: "
            f"{sorted(unsupported_columns)}. Use long format."
        )

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


def _resolve_wide_replicate_columns(
    df: pd.DataFrame,
    *,
    replicate_cols: list[str] | None,
    replicate_prefix: str,
) -> list[str]:
    """Resolve or discover the response columns of a wide input table."""
    if replicate_cols is not None:
        _require_columns(
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


def _normalize_format(format: str) -> str:
    """Normalize a user-provided serialization format name."""
    normalized_format = format.replace("-", "_").lower()
    if normalized_format not in {"long", "wide"}:
        raise ValueError("format must be 'long' or 'wide'.")
    return normalized_format


def _resolve_json_format(
    *,
    requested_format: str | None,
    payload_format: str | None,
) -> str:
    """Resolve the effective format for JSON deserialization."""
    normalized_requested = (
        _normalize_format(requested_format) if requested_format is not None else None
    )
    normalized_payload = (
        _normalize_format(payload_format) if payload_format is not None else None
    )

    if normalized_requested and normalized_payload:
        if normalized_requested != normalized_payload:
            raise ValueError("Requested format does not match the JSON payload format.")
        return normalized_requested
    if normalized_payload is not None:
        return normalized_payload
    if normalized_requested is not None:
        return normalized_requested
    return "long"


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


def _serialize_dataframe(
    table: pd.DataFrame,
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
    normalized_format = _normalize_format(format)

    if normalized_format == "long":
        rename_map = {
            "compound_id": compound_col,
            "concentration": concentration_col,
            "response": response_col,
            "experiment_id": experiment_col,
            "replicate_id": replicate_col,
        }
        return _rename_output_columns(table, rename_map)

    canonical_columns = {
        "compound_id",
        "experiment_id",
        "concentration",
        "replicate_id",
        "response",
    }
    unsupported_columns = set(table.columns) - canonical_columns
    if unsupported_columns:
        raise ValueError(
            "Wide serialization cannot represent these observation columns: "
            f"{sorted(unsupported_columns)}. Use long format."
        )

    table = table[
        [
            "compound_id",
            "experiment_id",
            "concentration",
            "replicate_id",
            "response",
        ]
    ].copy()
    group_cols = ["compound_id", "experiment_id", "concentration"]
    replicate_ids = table["replicate_id"].astype(str)
    invalid_ids = [
        replicate_id
        for replicate_id in replicate_ids.unique()
        if not (
            replicate_id.startswith(replicate_prefix)
            and replicate_id[len(replicate_prefix) :].isdigit()
        )
    ]
    if invalid_ids:
        raise ValueError(
            "Wide serialization requires positional replicate identifiers "
            f"matching {replicate_prefix!r} followed by an integer; got "
            f"{sorted(invalid_ids)}. Use long format."
        )
    wide = table.pivot(
        index=group_cols,
        columns="replicate_id",
        values="response",
    ).reset_index()
    response_columns = sorted(
        (column for column in wide.columns if column not in group_cols),
        key=lambda column: int(str(column)[len(replicate_prefix) :]),
    )
    wide = wide[group_cols + response_columns]
    wide.columns.name = None
    return _rename_output_columns(
        wide,
        {
            "compound_id": compound_col,
            "experiment_id": experiment_col,
            "concentration": concentration_col,
        },
    )


def _rename_output_columns(
    table: pd.DataFrame, rename_map: dict[str, str]
) -> pd.DataFrame:
    """Reject lossy output layouts before a serializer can discard columns."""
    output_columns = pd.Index(
        rename_map.get(column, column) for column in table.columns
    )
    if not output_columns.is_unique:
        raise ValueError(
            "Output column names must be unique, including metadata columns."
        )
    return table.rename(columns=rename_map)
