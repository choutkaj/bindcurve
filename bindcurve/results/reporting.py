from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import pandas as pd

from bindcurve.quality import resolve_requested_compounds
from bindcurve.results.types import ConcentrationSummary, ReportUncertainty

if TYPE_CHECKING:
    from bindcurve.results.core import FitResults

ReportRepresentation = Literal["linear", "log", "both"]
RoundingMode = Literal["sigfig", "decimals"]


def format_results_report(
    results: FitResults,
    *,
    parameter: str,
    compounds: str | Iterable[str] | None,
    representation: ReportRepresentation,
    uncertainty: ReportUncertainty,
    rounding: RoundingMode,
    places_mean: int,
    places_uncertainty: int,
    unit: str | None,
    include_n_exp: bool,
) -> pd.DataFrame:
    """Build manuscript-ready concentration summaries."""
    _validate_rounding_places(rounding, places_mean, places_uncertainty)
    selected_compounds = resolve_requested_compounds(
        _ordered_compound_ids(results),
        compounds,
    )
    columns = [
        "compound_id",
        "report",
        "N_fit_successful",
        "N_fit_failed",
    ]
    if not selected_compounds:
        return pd.DataFrame(columns=columns)

    concentration_summaries = [
        summary
        for summary in results.summaries
        if isinstance(summary, ConcentrationSummary)
    ]
    selected_summaries = [
        summary
        for summary in concentration_summaries
        if summary.compound_id in selected_compounds
    ]
    if selected_summaries:
        parameter_name = _resolve_report_parameter(
            selected_summaries,
            parameter=parameter,
            compound_ids=selected_compounds,
        )
    else:
        reportable = [
            spec.name
            for spec in results.model.concentration_parameter_specs
            if spec.reportable
        ]
        if parameter == "auto":
            if len(reportable) != 1:
                raise ValueError(
                    "A report parameter is required because the model does not "
                    "have exactly one reportable concentration parameter."
                )
            parameter_name = reportable[0]
        else:
            parameter_name = str(parameter)

    summary_lookup = {
        (summary.compound_id, summary.parameter): summary
        for summary in concentration_summaries
    }
    rows: list[dict[str, object]] = []
    for compound_id in selected_compounds:
        compound_fits = [
            fit
            for fit in results.fit_results
            if str(fit.compound_id) == str(compound_id)
        ]
        n_successful = sum(fit.success for fit in compound_fits)
        n_failed = len(compound_fits) - n_successful
        summary = summary_lookup.get((compound_id, parameter_name))
        report_text = (
            "unavailable: no successful fit summary"
            if summary is None
            else _format_concentration_report(
                summary,
                representation=representation,
                uncertainty=uncertainty,
                rounding=rounding,
                places_mean=places_mean,
                places_uncertainty=places_uncertainty,
                unit=unit,
                include_n_exp=include_n_exp,
            )
        )
        rows.append(
            {
                "compound_id": compound_id,
                "report": report_text,
                "N_fit_successful": n_successful,
                "N_fit_failed": n_failed,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ordered_compound_ids(results: FitResults) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for fit in results.fit_results:
        compound_id = str(fit.compound_id)
        if compound_id not in seen:
            seen.add(compound_id)
            ordered.append(compound_id)
    return ordered


def _resolve_report_parameter(
    summaries: list[ConcentrationSummary],
    *,
    parameter: str,
    compound_ids: list[str],
) -> str:
    available = {
        summary.parameter
        for summary in summaries
        if summary.compound_id in compound_ids
    }
    reportable = {
        summary.parameter
        for summary in summaries
        if summary.compound_id in compound_ids and summary.reportable
    }
    if parameter == "auto":
        if len(reportable) == 1:
            return next(iter(reportable))
        if not reportable:
            raise ValueError(
                "No reportable concentration quantity is available; specify "
                "parameter explicitly."
            )
        raise ValueError(
            "Multiple reportable concentration quantities are available; specify "
            f"parameter explicitly from {sorted(reportable)}."
        )
    if parameter not in available:
        raise KeyError(
            f"Concentration parameter {parameter!r} is unavailable. "
            f"Available parameters: {sorted(available)}"
        )
    return parameter


def _validate_rounding_places(
    rounding: RoundingMode,
    places_mean: int,
    places_uncertainty: int,
) -> None:
    if rounding not in {"sigfig", "decimals"}:
        raise ValueError("rounding must be 'sigfig' or 'decimals'.")
    minimum = 1 if rounding == "sigfig" else 0
    for name, value in (
        ("places_mean", places_mean),
        ("places_uncertainty", places_uncertainty),
    ):
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")


def _format_number(
    value: float,
    *,
    rounding: RoundingMode,
    places: int,
) -> str:
    value = float(value)
    if rounding == "decimals":
        return f"{value:.{places}f}"
    if value == 0.0:
        decimals = max(places - 1, 0)
        return f"{0.0:.{decimals}f}"
    exponent = math.floor(math.log10(abs(value)))
    decimals = places - exponent - 1
    if -4 <= exponent < places:
        return f"{value:.{max(decimals, 0)}f}"
    return f"{value:.{places - 1}e}"


def _format_linear_clause(
    summary: ConcentrationSummary,
    *,
    uncertainty: ReportUncertainty,
    rounding: RoundingMode,
    places_mean: int,
    places_uncertainty: int,
    unit: str | None,
) -> str:
    center_text = _format_number(
        summary.center,
        rounding=rounding,
        places=places_mean,
    )
    lower, upper = summary.linear_interval(uncertainty)
    interval = ""
    if lower is not None and upper is not None:
        lower_text = _format_number(
            lower,
            rounding=rounding,
            places=places_uncertainty,
        )
        upper_text = _format_number(
            upper,
            rounding=rounding,
            places=places_uncertainty,
        )
        interval = f" [{lower_text}, {upper_text}]"
    unit_text = f" {unit}" if unit else ""
    return f"{center_text}{interval}{unit_text}"


def _format_log_clause(
    summary: ConcentrationSummary,
    *,
    uncertainty: ReportUncertainty,
    rounding: RoundingMode,
    places_mean: int,
    places_uncertainty: int,
) -> str:
    mean_text = _format_number(
        summary.log10_mean,
        rounding=rounding,
        places=places_mean,
    )
    if uncertainty == "ci95":
        lower, upper = summary.log_interval("ci95")
        if lower is None or upper is None:
            return mean_text
        lower_text = _format_number(
            lower,
            rounding=rounding,
            places=places_uncertainty,
        )
        upper_text = _format_number(
            upper,
            rounding=rounding,
            places=places_uncertainty,
        )
        return f"{mean_text} [{lower_text}, {upper_text}]"

    spread = summary.log10_sd if uncertainty == "sd" else summary.log10_sem
    if spread is None:
        return mean_text
    delta_text = _format_number(
        spread,
        rounding=rounding,
        places=places_uncertainty,
    )
    return f"{mean_text} ± {delta_text}"


def _format_concentration_report(
    summary: ConcentrationSummary,
    *,
    representation: ReportRepresentation,
    uncertainty: ReportUncertainty,
    rounding: RoundingMode,
    places_mean: int,
    places_uncertainty: int,
    unit: str | None,
    include_n_exp: bool,
) -> str:
    if representation not in {"linear", "log", "both"}:
        raise ValueError("representation must be 'linear', 'log', or 'both'.")
    if uncertainty not in {"sd", "sem", "ci95"}:
        raise ValueError("uncertainty must be 'sd', 'sem', or 'ci95'.")

    linear_text = _format_linear_clause(
        summary,
        uncertainty=uncertainty,
        rounding=rounding,
        places_mean=places_mean,
        places_uncertainty=places_uncertainty,
        unit=unit,
    )
    log_text = _format_log_clause(
        summary,
        uncertainty=uncertainty,
        rounding=rounding,
        places_mean=places_mean,
        places_uncertainty=places_uncertainty,
    )
    if representation == "linear":
        report = linear_text
    elif representation == "log":
        report = log_text
    else:
        report = (
            f"{summary.parameter}: {linear_text}; "
            f"{summary.log_parameter}: {log_text}"
        )
    return f"{report}, N_exp = {summary.N_exp}" if include_n_exp else report
