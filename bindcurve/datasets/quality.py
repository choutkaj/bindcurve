from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from bindcurve.quality import (
    DataQualityThresholds,
    resolve_requested_compounds,
    summarize_quality_flags,
)

if TYPE_CHECKING:
    from bindcurve.datasets.dose_response import DoseResponseData

QUALITY_COLUMNS = [
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
]


def build_data_quality_report(
    data: DoseResponseData,
    *,
    compounds: str | Iterable[str] | None,
    thresholds: DataQualityThresholds | None,
) -> pd.DataFrame:
    """Build compound-level quality-control metrics for a dataset."""
    resolved_thresholds = thresholds or DataQualityThresholds()
    selected_compounds = resolve_requested_compounds(data.compounds, compounds)
    rows = [
        _compound_data_quality_row(
            data.select_compound(compound_id).table,
            thresholds=resolved_thresholds,
        )
        for compound_id in selected_compounds
    ]
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


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
