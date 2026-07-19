from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from bindcurve.quality import (
    ResultQualityThresholds,
    resolve_requested_compounds,
    summarize_quality_flags,
)
from bindcurve.results.types import (
    ConcentrationSummary,
    FitResult,
    ReportUncertainty,
    SummaryRecord,
)

if TYPE_CHECKING:
    from bindcurve.results.core import FitResults

QUALITY_COLUMNS = [
    "compound_id",
    "status",
    "N_flag_orange",
    "N_flag_red",
    "flags",
    "parameter",
    "N_exp",
    "N_fit_success",
    "N_fit_failed",
    "fit_success_fraction",
    "R_squared_median",
    "R_squared_min",
    "RSS_total",
    "Chi_squared_total",
    "reduced_RSS_median",
    "reduced_chi_square_median",
    "covariance_missing_fraction",
    "stderr_missing_fraction",
    "parameter_at_bound_fraction",
    "inter_log10_sd",
    "inter_log10_sem",
    "inter_log10_ci95_width",
    "inter_sd_fold",
    "inter_ci95_fold",
]


def build_results_quality_report(
    results: FitResults,
    *,
    parameter: str,
    compounds: str | Iterable[str] | None,
    thresholds: ResultQualityThresholds | None,
) -> pd.DataFrame:
    """Build compound-level fit and summary QC metrics."""
    selected_compounds = resolve_requested_compounds(
        _ordered_compound_ids(results.fit_results),
        compounds,
    )
    if not selected_compounds:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    resolved_thresholds = thresholds or ResultQualityThresholds()
    summaries = list(results.summaries)
    parameter_name = _resolve_quality_parameter(
        list(results.fit_results),
        summaries,
        parameter=parameter,
        compound_ids=selected_compounds,
    )
    concentration_lookup = {
        (summary.compound_id, summary.parameter): summary
        for summary in summaries
        if isinstance(summary, ConcentrationSummary)
    }
    rows = []
    for compound_id in selected_compounds:
        compound_fits = [
            fit
            for fit in results.fit_results
            if str(fit.compound_id) == str(compound_id)
        ]
        rows.append(
            _compound_result_quality_row(
                compound_id,
                compound_fits,
                concentration_lookup.get((compound_id, parameter_name)),
                parameter_name=parameter_name,
                thresholds=resolved_thresholds,
            )
        )
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def _ordered_compound_ids(fits: Iterable[FitResult]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for fit in fits:
        compound_id = str(fit.compound_id)
        if compound_id not in seen:
            seen.add(compound_id)
            ordered.append(compound_id)
    return ordered


def _resolve_quality_parameter(
    fits: list[FitResult],
    summaries: list[SummaryRecord],
    *,
    parameter: str,
    compound_ids: list[str],
) -> str:
    selected_summaries = [
        summary
        for summary in summaries
        if isinstance(summary, ConcentrationSummary)
        and summary.compound_id in compound_ids
    ]
    selected_fits = [
        fit for fit in fits if str(fit.compound_id) in compound_ids
    ]
    summary_available = {summary.parameter for summary in selected_summaries}
    modeled_available = _available_concentration_parameters(selected_fits)
    available = summary_available | modeled_available

    if parameter == "auto":
        reportable = {
            summary.parameter for summary in selected_summaries if summary.reportable
        }
        if not reportable:
            reportable = _available_concentration_parameters(
                selected_fits,
                reportable_only=True,
            )
        if not reportable:
            raise ValueError("No reportable concentration quantity is available.")
        if len(reportable) > 1:
            raise ValueError(
                "Multiple reportable concentration quantities are available; "
                "specify parameter explicitly. Candidates: "
                f"{sorted(reportable)}"
            )
        return sorted(reportable)[0]

    requested = str(parameter)
    if requested not in available:
        raise KeyError(
            f"Unknown concentration quality parameter {requested!r}. "
            f"Available parameters: {sorted(available)}"
        )
    return requested


def _compound_result_quality_row(
    compound_id: str,
    compound_fits: list[FitResult],
    concentration_summary: ConcentrationSummary | None,
    *,
    parameter_name: str,
    thresholds: ResultQualityThresholds,
) -> dict[str, object]:
    successful = [fit for fit in compound_fits if fit.success]
    N_exp = _compound_N_exp(compound_fits)
    N_fit_success = len(successful)
    N_fit_failed = len(compound_fits) - N_fit_success
    fit_success_fraction = (
        float(N_fit_success / len(compound_fits)) if compound_fits else np.nan
    )
    covariance_missing_fraction = _fraction_of_fits(
        successful,
        lambda fit: fit.covariance is None,
    )
    stderr_missing_fraction = _fraction_of_fits(successful, _fit_has_missing_stderr)
    parameter_at_bound_fraction = _fraction_of_fits(
        successful,
        lambda fit: _fit_has_parameter_at_bound(
            fit,
            rel_tol=thresholds.bound_tolerance_rel,
            abs_tol=thresholds.bound_tolerance_abs,
        ),
    )

    r_squared_values = [
        float(fit.metrics.r_squared)
        for fit in successful
        if fit.metrics is not None and fit.metrics.r_squared is not None
    ]
    reduced_rss_values = [
        float(fit.metrics.reduced_rss)
        for fit in successful
        if fit.metrics is not None and fit.metrics.reduced_rss is not None
    ]
    rss_values = [
        float(fit.metrics.rss)
        for fit in successful
        if fit.metrics is not None
    ]
    chi_square_values = [
        float(fit.metrics.chi_square)
        for fit in successful
        if fit.metrics is not None and fit.metrics.chi_square is not None
    ]
    reduced_chi_square_values = [
        float(fit.metrics.reduced_chi_square)
        for fit in successful
        if fit.metrics is not None and fit.metrics.reduced_chi_square is not None
    ]
    inter_log10_sd = (
        concentration_summary.log10_sd if concentration_summary is not None else None
    )
    inter_log10_sem = (
        concentration_summary.log10_sem if concentration_summary is not None else None
    )
    inter_log10_ci95_width = (
        None
        if concentration_summary is None
        else _ci95_width_on_log_scale(concentration_summary)
    )
    inter_sd_fold = (
        None
        if concentration_summary is None
        else _summary_interval_fold(concentration_summary, "sd")
    )
    inter_ci95_fold = (
        None
        if concentration_summary is None
        else _summary_interval_fold(concentration_summary, "ci95")
    )

    flags: list[tuple[str, str]] = []
    if N_fit_success == 0:
        flags.append(("red", "no successful fits"))
    if N_fit_failed > 0:
        flags.append(("orange", "fit failures present"))
    if concentration_summary is None:
        flags.append(("red", "selected concentration summary unavailable"))
    if N_exp < 2:
        flags.append(("red", "fewer than 2 independent experiments"))
    elif N_exp < thresholds.min_experiments_green:
        flags.append(("orange", f"only {N_exp} independent experiments"))
    if covariance_missing_fraction is not None and covariance_missing_fraction > 0.0:
        flags.append(("orange", "missing covariance for at least one successful fit"))
    if stderr_missing_fraction is not None and stderr_missing_fraction > 0.0:
        flags.append(
            (
                "orange",
                "missing parameter standard errors for at least one successful fit",
            )
        )
    if parameter_at_bound_fraction is not None and parameter_at_bound_fraction > 0.0:
        flags.append(
            (
                "orange",
                "at least one successful fit has a varying parameter at a bound",
            )
        )
    _append_quality_threshold_flag(
        flags,
        value=inter_ci95_fold,
        orange_threshold=thresholds.max_inter_ci95_fold_orange,
        red_threshold=thresholds.max_inter_ci95_fold_red,
        orange_message="wide inter-experiment CI95 fold range",
        red_message="very wide inter-experiment CI95 fold range",
    )

    status, N_flag_orange, N_flag_red, flag_text = summarize_quality_flags(flags)
    return {
        "compound_id": compound_id,
        "status": status,
        "N_flag_orange": N_flag_orange,
        "N_flag_red": N_flag_red,
        "flags": flag_text,
        "parameter": parameter_name,
        "N_exp": N_exp,
        "N_fit_success": N_fit_success,
        "N_fit_failed": N_fit_failed,
        "fit_success_fraction": fit_success_fraction,
        "R_squared_median": _nanmedian_or_none(r_squared_values),
        "R_squared_min": _nanmin_or_none(r_squared_values),
        "RSS_total": None if not rss_values else float(sum(rss_values)),
        "Chi_squared_total": (
            None if not chi_square_values else float(sum(chi_square_values))
        ),
        "reduced_RSS_median": _nanmedian_or_none(reduced_rss_values),
        "reduced_chi_square_median": _nanmedian_or_none(
            reduced_chi_square_values
        ),
        "covariance_missing_fraction": covariance_missing_fraction,
        "stderr_missing_fraction": stderr_missing_fraction,
        "parameter_at_bound_fraction": parameter_at_bound_fraction,
        "inter_log10_sd": inter_log10_sd,
        "inter_log10_sem": inter_log10_sem,
        "inter_log10_ci95_width": inter_log10_ci95_width,
        "inter_sd_fold": inter_sd_fold,
        "inter_ci95_fold": inter_ci95_fold,
    }


def _available_concentration_parameters(
    fits: list[FitResult],
    *,
    reportable_only: bool = False,
) -> set[str]:
    parameters: set[str] = set()
    models = {id(fit.model): fit.model for fit in fits}.values()
    for model in models:
        for spec in model.concentration_parameter_specs:
            if reportable_only and not spec.reportable:
                continue
            parameters.add(spec.name)
    return parameters


def _compound_N_exp(fits: list[FitResult]) -> int:
    experiment_ids = {
        str(fit.experiment_id)
        for fit in fits
        if fit.experiment_id is not None
    }
    return len(experiment_ids) if experiment_ids else len(fits)


def _fraction_of_fits(
    fits: list[FitResult],
    predicate: Callable[[FitResult], bool],
) -> float | None:
    if not fits:
        return None
    return float(sum(predicate(fit) for fit in fits) / len(fits))


def _fit_has_missing_stderr(fit: FitResult) -> bool:
    varying = [estimate for estimate in fit.parameters.values() if estimate.vary]
    return bool(varying) and any(estimate.stderr is None for estimate in varying)


def _fit_has_parameter_at_bound(
    fit: FitResult,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    for estimate in fit.parameters.values():
        if not estimate.vary:
            continue
        if math.isfinite(estimate.min) and math.isclose(
            estimate.value,
            estimate.min,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return True
        if math.isfinite(estimate.max) and math.isclose(
            estimate.value,
            estimate.max,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return True
    return False


def _ci95_width_on_log_scale(summary: ConcentrationSummary) -> float | None:
    if summary.log10_ci95_lower is None or summary.log10_ci95_upper is None:
        return None
    return float(summary.log10_ci95_upper - summary.log10_ci95_lower)


def _summary_interval_fold(
    summary: ConcentrationSummary,
    uncertainty: ReportUncertainty,
) -> float | None:
    lower, upper = summary.linear_interval(uncertainty)
    if lower is None or upper is None or lower <= 0.0:
        return None
    return float(upper / lower)


def _nanmedian_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.nanmedian(np.asarray(values, dtype=float)))


def _nanmin_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.nanmin(np.asarray(values, dtype=float)))


def _append_quality_threshold_flag(
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
