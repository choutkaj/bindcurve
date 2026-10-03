"""Across-experiment statistics, independent of tabular presentation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import t as student_t

from bindcurve.modeling.parameters import ParameterSpec
from bindcurve.results.types import (
    ConcentrationSummary,
    FitResult,
    ParameterSummary,
    SummaryRecord,
)


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


def summarize_fit_parameters(
    fits: Iterable[FitResult],
    *,
    parameter_specs: tuple[ParameterSpec, ...],
) -> list[SummaryRecord]:
    """Summarize successful fit parameters by compound."""
    successful = [fit for fit in fits if fit.success]
    grouped: dict[str, list[FitResult]] = {}
    for fit in successful:
        grouped.setdefault(str(fit.compound_id), []).append(fit)
    summaries: list[SummaryRecord] = []
    spec_by_parameter = {spec.name: spec for spec in parameter_specs}

    for compound_id, compound_fits in grouped.items():
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
            if not any(estimate.vary for estimate in estimates):
                continue

            values = np.asarray([estimate.value for estimate in estimates], dtype=float)
            spec = spec_by_parameter[fitted_parameter]
            if spec.kind != "concentration":
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
                    parameter=spec.name,
                    log_parameter=spec.resolved_log_name,
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
    spec: ParameterSpec,
) -> np.ndarray:
    linear_values = np.asarray(values, dtype=float)
    if np.any(linear_values <= 0.0):
        raise ValueError(
            f"Concentration summary for {spec.name!r} requires strictly "
            "positive values."
        )
    if np.any(~np.isfinite(linear_values)):
        raise ValueError(
            f"Concentration summary for {spec.name!r} requires finite values."
        )
    return np.log10(linear_values)
