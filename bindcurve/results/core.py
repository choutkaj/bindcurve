from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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
    """Summary of one parameter across independent fits."""

    compound_id: str
    parameter: str
    N_exp: int
    mean: float
    sd: float | None
    sem: float | None
    summary_scale: str = "linear"
    geometric_mean: float | None = None
    log10_mean: float | None = None
    log10_sd: float | None = None


@dataclass
class FitResults:
    """Collection of individual fits and parameter summaries."""

    fits: list[FitResult]
    summaries: list[ParameterSummary] = field(default_factory=list)

    def successful(self) -> list[FitResult]:
        """Return successful fits."""
        return [fit for fit in self.fits if fit.success]

    def failed(self) -> list[FitResult]:
        """Return failed fits."""
        return [fit for fit in self.fits if not fit.success]

    def fits(self) -> pd.DataFrame:
        """Represent individual fits as a DataFrame."""
        rows: list[dict[str, object]] = []
        for fit in self.fits:
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
        """Represent per-parameter summaries as a long-form DataFrame."""
        return pd.DataFrame([summary.__dict__ for summary in self.summaries])

    def summary(self) -> pd.DataFrame:
        """Represent compound-level summaries as one row per compound."""
        successful = self.successful()
        if not successful:
            return pd.DataFrame()

        compound_ids: list[str] = []
        seen_compounds: set[str] = set()
        parameter_order: list[str] = []
        seen_parameters: set[str] = set()

        for fit in successful:
            if fit.compound_id not in seen_compounds:
                seen_compounds.add(fit.compound_id)
                compound_ids.append(fit.compound_id)
            for name in fit.parameters:
                if name not in seen_parameters:
                    seen_parameters.add(name)
                    parameter_order.append(name)

        summary_lookup = {
            (summary.compound_id, summary.parameter): summary for summary in self.summaries
        }
        rows: list[dict[str, object]] = []

        for compound_id in compound_ids:
            compound_fits = [fit for fit in successful if fit.compound_id == compound_id]
            row: dict[str, object] = {
                "compound_id": compound_id,
                "N_exp": len(compound_fits),
                "N_obs": _compound_n_obs(compound_fits),
            }

            for name in parameter_order:
                summary = summary_lookup.get((compound_id, name))
                if summary is None:
                    row[name] = np.nan
                    row[f"{name}_SD"] = np.nan
                    row[f"{name}_SEM"] = np.nan
                    continue

                row[name] = _summary_value(summary)
                row[f"{name}_SD"] = (
                    np.nan if summary.sd is None else float(summary.sd)
                )
                row[f"{name}_SEM"] = (
                    np.nan if summary.sem is None else float(summary.sem)
                )

            row["R_squared"] = _compound_r_squared(compound_fits)
            row["Chi_squared"] = _compound_chi_squared(compound_fits)
            rows.append(row)

        columns = ["compound_id", "N_exp", "N_obs"]
        for name in parameter_order:
            columns.extend([name, f"{name}_SD", f"{name}_SEM"])
        columns.extend(["R_squared", "Chi_squared"])
        return pd.DataFrame(rows, columns=columns)


def _summary_value(summary: ParameterSummary) -> float:
    if summary.geometric_mean is not None:
        return float(summary.geometric_mean)
    return float(summary.mean)


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
    values = [
        float(fit.metrics.chisqr) for fit in fits if fit.metrics is not None
    ]
    if not values:
        return None
    return float(sum(values))


def summarize_fit_parameters(
    fits: list[FitResult],
    *,
    concentration_parameters: set[str] | frozenset[str],
) -> list[ParameterSummary]:
    """Summarize successful fit parameters by compound."""
    successful = [fit for fit in fits if fit.success]
    compound_ids = sorted({fit.compound_id for fit in successful})
    summaries: list[ParameterSummary] = []

    for compound_id in compound_ids:
        compound_fits = [fit for fit in successful if fit.compound_id == compound_id]
        parameter_names = sorted(
            {name for fit in compound_fits for name in fit.parameters}
        )
        for name in parameter_names:
            estimates = [
                fit.parameters[name] for fit in compound_fits if name in fit.parameters
            ]
            values = np.asarray([estimate.value for estimate in estimates], dtype=float)
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            sem = float(sd / np.sqrt(len(values))) if sd is not None else None

            if name in concentration_parameters and np.all(values > 0):
                log_values = np.log10(values)
                log10_mean = float(np.mean(log_values))
                log10_sd = (
                    float(np.std(log_values, ddof=1)) if len(values) > 1 else None
                )
                summaries.append(
                    ParameterSummary(
                        compound_id=compound_id,
                        parameter=name,
                        N_exp=len(values),
                        mean=float(np.mean(values)),
                        sd=sd,
                        sem=sem,
                        summary_scale="log10",
                        geometric_mean=float(10**log10_mean),
                        log10_mean=log10_mean,
                        log10_sd=log10_sd,
                    )
                )
            else:
                summaries.append(
                    ParameterSummary(
                        compound_id=compound_id,
                        parameter=name,
                        N_exp=len(values),
                        mean=float(np.mean(values)),
                        sd=sd,
                        sem=sem,
                    )
                )
    return summaries
