from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from bindcurve.modeling.parameters import ConcentrationParameterSpec
from bindcurve.quality import ResultQualityThresholds, summarize_quality_flags

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ReportRepresentation = Literal["linear", "log", "both"]
ReportUncertainty = Literal["sd", "sem", "ci95"]
RoundingMode = Literal["sigfig", "decimals"]


@dataclass(frozen=True)
class ParameterEstimate:
    """Estimate for one fitted parameter."""

    name: str
    value: float
    stderr: float | None = None
    vary: bool = True


@dataclass(frozen=True)
class FitMetrics:
    """Common numerical diagnostics for one fit."""

    n_data: int
    n_varying_parameters: int
    chisqr: float
    redchi: float
    aic: float | None = None
    bic: float | None = None
    r_squared: float | None = None


@dataclass
class FitResult:
    """Result for one fitted curve."""

    compound_id: str
    model_name: str
    experiment_id: str | None = None
    success: bool = True
    parameters: dict[str, ParameterEstimate] = field(default_factory=dict)
    metrics: FitMetrics | None = None
    lmfit_result: Any | None = None

    @classmethod
    def failed(
        cls,
        *,
        compound_id: str,
        model_name: str,
        experiment_id: str | None,
    ) -> FitResult:
        """Create a failed result container."""
        return cls(
            compound_id=compound_id,
            model_name=model_name,
            experiment_id=experiment_id,
            success=False,
        )

    def parameter(self, name: str) -> ParameterEstimate:
        """Return a fitted parameter by name."""
        return self.parameters[name]


@dataclass(frozen=True)
class ParameterSummary:
    """Summary of one native additive parameter across independent fits."""

    compound_id: str
    parameter: str
    N_exp: int
    mean: float
    sd: float | None
    sem: float | None
    ci95_lower: float | None
    ci95_upper: float | None


@dataclass(frozen=True)
class ConcentrationSummary:
    """Summary of one positive concentration-like quantity across fits."""

    compound_id: str
    parameter: str
    log_parameter: str
    N_exp: int
    reportable: bool
    log10_mean: float
    log10_sd: float | None
    log10_sem: float | None
    log10_ci95_lower: float | None
    log10_ci95_upper: float | None

    @property
    def center(self) -> float:
        """Return the linear-scale center derived from the log10 mean."""
        return float(10**self.log10_mean)

    def linear_interval(
        self,
        uncertainty: ReportUncertainty,
    ) -> tuple[float | None, float | None]:
        """Return a derived linear-scale interval."""
        if uncertainty == "sd":
            return _back_transform_log_interval(self.log10_mean, self.log10_sd)
        if uncertainty == "sem":
            return _back_transform_log_interval(self.log10_mean, self.log10_sem)
        if uncertainty == "ci95":
            if self.log10_ci95_lower is None or self.log10_ci95_upper is None:
                return (None, None)
            return (
                float(10**self.log10_ci95_lower),
                float(10**self.log10_ci95_upper),
            )
        raise ValueError("uncertainty must be 'sd', 'sem', or 'ci95'.")

    def log_interval(
        self,
        uncertainty: ReportUncertainty,
    ) -> tuple[float | None, float | None]:
        """Return a log10-scale interval or additive spread."""
        if uncertainty == "sd":
            return _symmetric_interval(self.log10_mean, self.log10_sd)
        if uncertainty == "sem":
            return _symmetric_interval(self.log10_mean, self.log10_sem)
        if uncertainty == "ci95":
            return (self.log10_ci95_lower, self.log10_ci95_upper)
        raise ValueError("uncertainty must be 'sd', 'sem', or 'ci95'.")


SummaryRecord: TypeAlias = ParameterSummary | ConcentrationSummary


@dataclass
class FitResults:
    """Collection of individual fits and parameter summaries."""

    fit_results: list[FitResult]
    summaries: list[SummaryRecord] = field(default_factory=list)

    def successful(self) -> list[FitResult]:
        """Return successful fits."""
        return [fit for fit in self.fit_results if fit.success]

    def failed(self) -> list[FitResult]:
        """Return failed fits."""
        return [fit for fit in self.fit_results if not fit.success]

    def fits(self) -> pd.DataFrame:
        """Represent individual fits as a DataFrame."""
        return self.fit_summary()

    def fit_summary(self) -> pd.DataFrame:
        """Represent individual fits as a DataFrame."""
        rows: list[dict[str, object]] = []
        for fit in self.fit_results:
            row: dict[str, object] = {
                "compound_id": fit.compound_id,
                "experiment_id": fit.experiment_id,
                "model": fit.model_name,
                "success": fit.success,
            }
            if fit.metrics is not None:
                row.update(
                    {
                        "n_data": fit.metrics.n_data,
                        "n_varying_parameters": fit.metrics.n_varying_parameters,
                        "chisqr": fit.metrics.chisqr,
                        "redchi": fit.metrics.redchi,
                        "aic": fit.metrics.aic,
                        "bic": fit.metrics.bic,
                        "r_squared": fit.metrics.r_squared,
                    }
                )
            for name, estimate in fit.parameters.items():
                row[name] = estimate.value
                row[f"{name}_stderr"] = estimate.stderr
            rows.append(row)
        return pd.DataFrame(rows)

    def parameters(self) -> pd.DataFrame:
        """Represent parameter summaries as a long-form DataFrame."""
        rows: list[dict[str, object]] = []
        for summary in self.summaries:
            if isinstance(summary, ConcentrationSummary):
                sd_lower, sd_upper = summary.linear_interval("sd")
                sem_lower, sem_upper = summary.linear_interval("sem")
                ci95_lower, ci95_upper = summary.linear_interval("ci95")
                rows.append(
                    {
                        "compound_id": summary.compound_id,
                        "parameter": summary.parameter,
                        "log_parameter": summary.log_parameter,
                        "summary_type": "concentration",
                        "reportable": summary.reportable,
                        "N_exp": summary.N_exp,
                        "log10_mean": summary.log10_mean,
                        "log10_sd": summary.log10_sd,
                        "log10_sem": summary.log10_sem,
                        "log10_ci95_lower": summary.log10_ci95_lower,
                        "log10_ci95_upper": summary.log10_ci95_upper,
                        "center": summary.center,
                        "sd_lower": sd_lower,
                        "sd_upper": sd_upper,
                        "sem_lower": sem_lower,
                        "sem_upper": sem_upper,
                        "ci95_lower": ci95_lower,
                        "ci95_upper": ci95_upper,
                    }
                )
                continue

            rows.append(
                {
                    "compound_id": summary.compound_id,
                    "parameter": summary.parameter,
                    "summary_type": "native",
                    "N_exp": summary.N_exp,
                    "mean": summary.mean,
                    "sd": summary.sd,
                    "sem": summary.sem,
                    "ci95_lower": summary.ci95_lower,
                    "ci95_upper": summary.ci95_upper,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Represent compound-level summaries as one row per compound."""
        successful = self.successful()
        if not successful:
            return pd.DataFrame()

        compound_ids = _ordered_compound_ids(successful)
        summary_lookup: dict[tuple[str, str], SummaryRecord] = {}
        summary_kind: dict[str, str] = {}
        alias_map: dict[str, str] = {}

        for summary in self.summaries:
            summary_lookup[(summary.compound_id, summary.parameter)] = summary
            alias_map[summary.parameter] = summary.parameter
            if isinstance(summary, ConcentrationSummary):
                summary_kind[summary.parameter] = "concentration"
                alias_map[summary.log_parameter] = summary.parameter
            else:
                summary_kind[summary.parameter] = "native"

        parameter_order: list[str] = []
        seen_parameters: set[str] = set()
        for fit in successful:
            for name in fit.parameters:
                canonical = alias_map.get(name)
                if canonical is None or canonical in seen_parameters:
                    continue
                seen_parameters.add(canonical)
                parameter_order.append(canonical)

        for name in summary_kind:
            if name not in seen_parameters:
                parameter_order.append(name)

        rows: list[dict[str, object]] = []
        for compound_id in compound_ids:
            compound_fits = [
                fit for fit in successful if str(fit.compound_id) == str(compound_id)
            ]
            row: dict[str, object] = {
                "compound_id": compound_id,
                "N_exp": len(compound_fits),
                "N_obs": _compound_n_obs(compound_fits),
            }

            for name in parameter_order:
                summary = summary_lookup.get((compound_id, name))
                if summary_kind[name] == "concentration":
                    _update_concentration_summary_row(row, name, summary)
                else:
                    _update_native_summary_row(row, name, summary)

            row["R_squared"] = _compound_r_squared(compound_fits)
            row["Chi_squared"] = _compound_chi_squared(compound_fits)
            rows.append(row)

        columns = ["compound_id", "N_exp", "N_obs"]
        for name in parameter_order:
            if summary_kind[name] == "concentration":
                columns.extend(
                    [
                        name,
                        f"{name}_SD_lower",
                        f"{name}_SD_upper",
                        f"{name}_SEM_lower",
                        f"{name}_SEM_upper",
                        f"{name}_CI95_lower",
                        f"{name}_CI95_upper",
                    ]
                )
            else:
                columns.extend(
                    [
                        name,
                        f"{name}_SD",
                        f"{name}_SEM",
                        f"{name}_CI95_lower",
                        f"{name}_CI95_upper",
                    ]
                )
        columns.extend(["R_squared", "Chi_squared"])
        return pd.DataFrame(rows, columns=columns)

    def report(
        self,
        *,
        parameter: str = "auto",
        compounds: str | Iterable[str] | None = None,
        representation: ReportRepresentation = "linear",
        uncertainty: ReportUncertainty = "sd",
        rounding: RoundingMode = "sigfig",
        places_mean: int = 2,
        places_uncertainty: int = 1,
        unit: str | None = None,
        include_n_exp: bool = False,
    ) -> pd.DataFrame:
        """Return manuscript-ready formatted concentration summaries."""
        _validate_rounding_places(rounding, places_mean, places_uncertainty)
        selected_compounds = _resolve_requested_compounds(
            self.successful(),
            compounds,
        )
        if not selected_compounds:
            return pd.DataFrame(columns=["compound_id", "report"])

        concentration_summaries = [
            summary
            for summary in self.summaries
            if isinstance(summary, ConcentrationSummary)
        ]
        parameter_name = _resolve_report_parameter(
            concentration_summaries,
            parameter=parameter,
            compound_ids=selected_compounds,
        )

        summary_lookup = {
            (summary.compound_id, summary.parameter): summary
            for summary in concentration_summaries
        }
        rows: list[dict[str, str]] = []
        for compound_id in selected_compounds:
            summary = summary_lookup.get((compound_id, parameter_name))
            if summary is None:
                raise KeyError(
                    f"Compound {compound_id!r} does not have concentration summary "
                    f"{parameter_name!r}."
                )
            report = _format_concentration_report(
                summary,
                representation=representation,
                uncertainty=uncertainty,
                rounding=rounding,
                places_mean=places_mean,
                places_uncertainty=places_uncertainty,
                unit=unit,
                include_n_exp=include_n_exp,
            )
            rows.append({"compound_id": compound_id, "report": report})

        return pd.DataFrame(rows, columns=["compound_id", "report"])

    def quality_report(
        self,
        *,
        parameter: str = "auto",
        compounds: str | Iterable[str] | None = None,
        thresholds: ResultQualityThresholds | None = None,
    ) -> pd.DataFrame:
        """Return compound-level fit and summary QC metrics."""
        selected_compounds = _resolve_requested_compounds(self.fit_results, compounds)
        if not selected_compounds:
            return pd.DataFrame()

        resolved_thresholds = thresholds or ResultQualityThresholds()
        parameter_name = _resolve_quality_parameter(
            self.fit_results,
            self.summaries,
            parameter=parameter,
            compound_ids=selected_compounds,
        )
        concentration_lookup = {
            (summary.compound_id, summary.parameter): summary
            for summary in self.summaries
            if isinstance(summary, ConcentrationSummary)
        }
        rows: list[dict[str, object]] = []
        for compound_id in selected_compounds:
            compound_fits = [
                fit
                for fit in self.fit_results
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

        columns = [
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
            "Chi_squared_total",
            "redchi_median",
            "covariance_missing_fraction",
            "stderr_missing_fraction",
            "parameter_at_bound_fraction",
            "inter_log10_sd",
            "inter_log10_sem",
            "inter_log10_ci95_width",
            "inter_sd_fold",
            "inter_ci95_fold",
        ]
        return pd.DataFrame(rows, columns=columns)

    def quality_dashboard(
        self,
        *,
        parameter: str = "auto",
        compounds: str | Iterable[str] | None = None,
        thresholds: ResultQualityThresholds | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Return a graphical dashboard summarizing results-level QC."""
        from bindcurve.plotting.quality import plot_results_quality_dashboard

        return plot_results_quality_dashboard(
            self,
            parameter=parameter,
            compounds=compounds,
            thresholds=thresholds,
            figsize=figsize,
        )


def _ordered_compound_ids(fits: list[FitResult]) -> list[str]:
    compound_ids: list[str] = []
    seen: set[str] = set()
    for fit in fits:
        compound_id = str(fit.compound_id)
        if compound_id in seen:
            continue
        seen.add(compound_id)
        compound_ids.append(compound_id)
    return compound_ids


def _sample_sd(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def _sample_sem(values: np.ndarray) -> float | None:
    sd = _sample_sd(values)
    if sd is None:
        return None
    return float(sd / np.sqrt(len(values)))


def _student_t_multiplier(sample_size: int) -> float | None:
    if sample_size < 2:
        return None
    return float(student_t.ppf(0.975, df=sample_size - 1))


def _ci95_interval(
    mean: float,
    sem: float | None,
    sample_size: int,
) -> tuple[float | None, float | None]:
    multiplier = _student_t_multiplier(sample_size)
    if multiplier is None or sem is None:
        return (None, None)
    delta = multiplier * sem
    return (float(mean - delta), float(mean + delta))


def _symmetric_interval(
    center: float,
    delta: float | None,
) -> tuple[float | None, float | None]:
    if delta is None:
        return (None, None)
    return (float(center - delta), float(center + delta))


def _back_transform_log_interval(
    log10_mean: float,
    log10_delta: float | None,
) -> tuple[float | None, float | None]:
    lower, upper = _symmetric_interval(log10_mean, log10_delta)
    if lower is None or upper is None:
        return (None, None)
    return (float(10**lower), float(10**upper))


def _compound_n_obs(fits: list[FitResult]) -> int | None:
    n_data = [fit.metrics.n_data for fit in fits if fit.metrics is not None]
    if not n_data:
        return None
    return int(sum(n_data))


def _compound_r_squared(fits: list[FitResult]) -> float | None:
    weighted_values: list[tuple[int, float]] = []
    for fit in fits:
        if fit.metrics is None or fit.metrics.r_squared is None:
            continue
        weighted_values.append((fit.metrics.n_data, float(fit.metrics.r_squared)))
    if not weighted_values:
        return None
    total_weight = sum(weight for weight, _ in weighted_values)
    if total_weight == 0:
        return None
    return float(
        sum(weight * value for weight, value in weighted_values) / total_weight
    )


def _compound_chi_squared(fits: list[FitResult]) -> float | None:
    values = [float(fit.metrics.chisqr) for fit in fits if fit.metrics is not None]
    if not values:
        return None
    return float(sum(values))


def _update_native_summary_row(
    row: dict[str, object],
    name: str,
    summary: SummaryRecord | None,
) -> None:
    if summary is None:
        row[name] = np.nan
        row[f"{name}_SD"] = np.nan
        row[f"{name}_SEM"] = np.nan
        row[f"{name}_CI95_lower"] = np.nan
        row[f"{name}_CI95_upper"] = np.nan
        return

    if not isinstance(summary, ParameterSummary):
        raise TypeError(f"Expected ParameterSummary for {name!r}.")

    row[name] = summary.mean
    row[f"{name}_SD"] = np.nan if summary.sd is None else summary.sd
    row[f"{name}_SEM"] = np.nan if summary.sem is None else summary.sem
    row[f"{name}_CI95_lower"] = (
        np.nan if summary.ci95_lower is None else summary.ci95_lower
    )
    row[f"{name}_CI95_upper"] = (
        np.nan if summary.ci95_upper is None else summary.ci95_upper
    )


def _update_concentration_summary_row(
    row: dict[str, object],
    name: str,
    summary: SummaryRecord | None,
) -> None:
    if summary is None:
        row[name] = np.nan
        row[f"{name}_SD_lower"] = np.nan
        row[f"{name}_SD_upper"] = np.nan
        row[f"{name}_SEM_lower"] = np.nan
        row[f"{name}_SEM_upper"] = np.nan
        row[f"{name}_CI95_lower"] = np.nan
        row[f"{name}_CI95_upper"] = np.nan
        return

    if not isinstance(summary, ConcentrationSummary):
        raise TypeError(f"Expected ConcentrationSummary for {name!r}.")

    sd_lower, sd_upper = summary.linear_interval("sd")
    sem_lower, sem_upper = summary.linear_interval("sem")
    ci95_lower, ci95_upper = summary.linear_interval("ci95")

    row[name] = summary.center
    row[f"{name}_SD_lower"] = np.nan if sd_lower is None else sd_lower
    row[f"{name}_SD_upper"] = np.nan if sd_upper is None else sd_upper
    row[f"{name}_SEM_lower"] = np.nan if sem_lower is None else sem_lower
    row[f"{name}_SEM_upper"] = np.nan if sem_upper is None else sem_upper
    row[f"{name}_CI95_lower"] = np.nan if ci95_lower is None else ci95_lower
    row[f"{name}_CI95_upper"] = np.nan if ci95_upper is None else ci95_upper


def _resolve_requested_compounds(
    fits: list[FitResult],
    compounds: str | Iterable[str] | None,
) -> list[str]:
    available = _ordered_compound_ids(fits)
    if compounds is None:
        return available

    if isinstance(compounds, str):
        requested = [str(compounds)]
    else:
        requested = [str(value) for value in compounds]
    missing = [compound_id for compound_id in requested if compound_id not in available]
    if missing:
        raise KeyError(f"Unknown compound(s): {missing}")
    return requested


def _resolve_report_parameter(
    summaries: list[ConcentrationSummary],
    *,
    parameter: str,
    compound_ids: list[str],
) -> str:
    selected = [
        summary
        for summary in summaries
        if summary.compound_id in compound_ids
    ]
    if parameter == "auto":
        reportable = sorted(
            {
                summary.parameter
                for summary in selected
                if summary.reportable
            }
        )
        if not reportable:
            raise ValueError("No reportable concentration quantity is available.")
        if len(reportable) > 1:
            raise ValueError(
                "Multiple reportable concentration quantities are available; "
                f"specify parameter explicitly. Candidates: {reportable}"
            )
        return reportable[0]

    requested = str(parameter)
    available = {summary.parameter for summary in selected}
    if requested not in available:
        raise KeyError(
            f"Unknown concentration report parameter {requested!r}. "
            f"Available parameters: {sorted(available)}"
        )
    return requested


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
    modeled_available = _available_concentration_parameters_from_models(selected_fits)
    available = summary_available | modeled_available

    if parameter == "auto":
        reportable = {
            summary.parameter for summary in selected_summaries if summary.reportable
        }
        if not reportable:
            reportable = _available_concentration_parameters_from_models(
                selected_fits,
                reportable_only=True,
            )
        if not reportable:
            raise ValueError("No reportable concentration quantity is available.")
        if len(reportable) > 1:
            raise ValueError(
                "Multiple reportable concentration quantities are available; "
                f"specify parameter explicitly. Candidates: {sorted(reportable)}"
            )
        return sorted(reportable)[0]

    requested = str(parameter)
    if requested not in available:
        raise KeyError(
            f"Unknown concentration quality parameter {requested!r}. "
            f"Available parameters: {sorted(available)}"
        )
    return requested


def _validate_rounding_places(
    rounding: RoundingMode,
    places_mean: int,
    places_uncertainty: int,
) -> None:
    if rounding not in {"sigfig", "decimals"}:
        raise ValueError("rounding must be 'sigfig' or 'decimals'.")
    if not isinstance(places_mean, int) or places_mean < 0:
        raise ValueError("places_mean must be a non-negative integer.")
    if not isinstance(places_uncertainty, int) or places_uncertainty < 0:
        raise ValueError("places_uncertainty must be a non-negative integer.")
    if rounding == "sigfig" and (places_mean == 0 or places_uncertainty == 0):
        raise ValueError("Significant-figure rounding requires positive place counts.")


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

    absolute = abs(value)
    exponent = math.floor(math.log10(absolute))
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
    if lower is None or upper is None:
        text = center_text
    else:
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
        text = f"{center_text} [{lower_text}, {upper_text}]"
    if unit:
        return f"{text} {unit}"
    return text


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

    delta = summary.log10_sd if uncertainty == "sd" else summary.log10_sem
    if delta is None:
        return mean_text
    delta_text = _format_number(
        delta,
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

    if include_n_exp:
        report = f"{report}, N_exp = {summary.N_exp}"
    return report


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
        lambda fit: fit.lmfit_result is None
        or getattr(fit.lmfit_result, "covar", None) is None,
    )
    stderr_missing_fraction = _fraction_of_fits(
        successful,
        _fit_has_missing_stderr,
    )
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
    redchi_values = [
        float(fit.metrics.redchi)
        for fit in successful
        if fit.metrics is not None
    ]
    chisqr_values = [
        float(fit.metrics.chisqr)
        for fit in successful
        if fit.metrics is not None
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
        "Chi_squared_total": None if not chisqr_values else float(sum(chisqr_values)),
        "redchi_median": _nanmedian_or_none(redchi_values),
        "covariance_missing_fraction": covariance_missing_fraction,
        "stderr_missing_fraction": stderr_missing_fraction,
        "parameter_at_bound_fraction": parameter_at_bound_fraction,
        "inter_log10_sd": inter_log10_sd,
        "inter_log10_sem": inter_log10_sem,
        "inter_log10_ci95_width": inter_log10_ci95_width,
        "inter_sd_fold": inter_sd_fold,
        "inter_ci95_fold": inter_ci95_fold,
    }


def _available_concentration_parameters_from_models(
    fits: list[FitResult],
    *,
    reportable_only: bool = False,
) -> set[str]:
    if not fits:
        return set()

    from bindcurve.modeling import get_model

    parameters: set[str] = set()
    model_names = {fit.model_name for fit in fits}
    for model_name in model_names:
        model = get_model(model_name)
        for spec in model.concentration_parameter_specs:
            if reportable_only and not spec.reportable:
                continue
            parameters.add(spec.parameter)
    return parameters


def _compound_N_exp(fits: list[FitResult]) -> int:
    experiment_ids = {
        str(fit.experiment_id)
        for fit in fits
        if fit.experiment_id is not None
    }
    if experiment_ids:
        return len(experiment_ids)
    return len(fits)


def _fraction_of_fits(
    fits: list[FitResult],
    predicate: Any,
) -> float | None:
    if not fits:
        return None
    matches = sum(1 for fit in fits if predicate(fit))
    return float(matches / len(fits))


def _fit_has_missing_stderr(fit: FitResult) -> bool:
    varying = [estimate for estimate in fit.parameters.values() if estimate.vary]
    if not varying:
        return False
    return any(estimate.stderr is None for estimate in varying)


def _fit_has_parameter_at_bound(
    fit: FitResult,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if fit.lmfit_result is None:
        return False
    for parameter in fit.lmfit_result.params.values():
        if not parameter.vary:
            continue
        value = float(parameter.value)
        lower = float(parameter.min)
        upper = float(parameter.max)
        if math.isfinite(lower) and math.isclose(
            value,
            lower,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return True
        if math.isfinite(upper) and math.isclose(
            value,
            upper,
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
    if not values:
        return None
    return float(np.nanmedian(np.asarray(values, dtype=float)))


def _nanmin_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.nanmin(np.asarray(values, dtype=float)))


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


def summarize_fit_parameters(
    fits: list[FitResult],
    *,
    concentration_parameter_specs: tuple[ConcentrationParameterSpec, ...],
) -> list[SummaryRecord]:
    """Summarize successful fit parameters by compound."""
    successful = [fit for fit in fits if fit.success]
    compound_ids = _ordered_compound_ids(successful)
    summaries: list[SummaryRecord] = []
    spec_by_fitted_parameter = {
        spec.fitted_parameter: spec for spec in concentration_parameter_specs
    }

    for compound_id in compound_ids:
        compound_fits = [
            fit for fit in successful if str(fit.compound_id) == str(compound_id)
        ]
        parameter_order: list[str] = []
        seen_parameters: set[str] = set()
        for fit in compound_fits:
            for name in fit.parameters:
                if name in seen_parameters:
                    continue
                seen_parameters.add(name)
                parameter_order.append(name)

        for fitted_parameter in parameter_order:
            estimates = [
                fit.parameters[fitted_parameter]
                for fit in compound_fits
                if fitted_parameter in fit.parameters
            ]
            if not estimates:
                continue

            values = np.asarray([estimate.value for estimate in estimates], dtype=float)
            spec = spec_by_fitted_parameter.get(fitted_parameter)
            if spec is None:
                mean = float(np.mean(values))
                sd = _sample_sd(values)
                sem = _sample_sem(values)
                ci95_lower, ci95_upper = _ci95_interval(mean, sem, len(values))
                summaries.append(
                    ParameterSummary(
                        compound_id=compound_id,
                        parameter=fitted_parameter,
                        N_exp=len(values),
                        mean=mean,
                        sd=sd,
                        sem=sem,
                        ci95_lower=ci95_lower,
                        ci95_upper=ci95_upper,
                    )
                )
                continue

            log_values = _to_log10_values(values, spec)
            log10_mean = float(np.mean(log_values))
            log10_sd = _sample_sd(log_values)
            log10_sem = _sample_sem(log_values)
            log10_ci95_lower, log10_ci95_upper = _ci95_interval(
                log10_mean,
                log10_sem,
                len(log_values),
            )
            summaries.append(
                ConcentrationSummary(
                    compound_id=compound_id,
                    parameter=spec.parameter,
                    log_parameter=spec.resolved_log_parameter,
                    N_exp=len(log_values),
                    reportable=spec.reportable,
                    log10_mean=log10_mean,
                    log10_sd=log10_sd,
                    log10_sem=log10_sem,
                    log10_ci95_lower=log10_ci95_lower,
                    log10_ci95_upper=log10_ci95_upper,
                )
            )
    return summaries


def _to_log10_values(
    values: np.ndarray,
    spec: ConcentrationParameterSpec,
) -> np.ndarray:
    if spec.fitted_scale == "log10":
        return np.asarray(values, dtype=float)

    linear_values = np.asarray(values, dtype=float)
    if np.any(linear_values <= 0.0):
        raise ValueError(
            f"Concentration summary for {spec.parameter!r} requires strictly "
            "positive values."
        )
    return np.log10(linear_values)
