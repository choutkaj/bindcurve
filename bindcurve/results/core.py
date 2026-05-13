from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from bindcurve.modeling.parameters import ConcentrationParameterSpec

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
    message: str | None = None
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
        message: str,
    ) -> FitResult:
        """Create a failed result container."""
        return cls(
            compound_id=compound_id,
            model_name=model_name,
            experiment_id=experiment_id,
            success=False,
            message=message,
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
                "message": fit.message,
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
