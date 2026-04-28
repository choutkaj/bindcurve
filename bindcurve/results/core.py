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
    unit: str | None = None
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
    n: int
    mean: float
    sd: float | None
    sem: float | None
    unit: str | None = None
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

    def fits_to_dataframe(self) -> pd.DataFrame:
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
                row[f"{name}_unit"] = estimate.unit
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_to_dataframe(self) -> pd.DataFrame:
        """Represent compound-level parameter summaries as a DataFrame."""
        return pd.DataFrame([summary.__dict__ for summary in self.summaries])


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
            unit = estimates[0].unit if estimates else None
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
                        n=len(values),
                        mean=float(np.mean(values)),
                        sd=sd,
                        sem=sem,
                        unit=unit,
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
                        n=len(values),
                        mean=float(np.mean(values)),
                        sd=sd,
                        sem=sem,
                        unit=unit,
                    )
                )
    return summaries
