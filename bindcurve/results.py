"""Fit results, across-experiment summaries and formatted reports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from bindcurve.data import DoseResponseData
from bindcurve.models import Model


@dataclass(frozen=True)
class FitResult:
    """Fit of one model to one experiment of one compound.

    Attributes
    ----------
    values
        All parameter values, fitted and fixed. Empty for a failed fit.
    stderr
        Standard errors of the fitted parameters, when they could be estimated.
    free
        Names of the fitted parameters, in the order of ``covariance``.
    n_data
        Number of fitted concentrations (replicate means).
    chi_square
        Weighted residual sum of squares, available when sigma is known.
    warnings
        Reasons why a converged fit may not be supported by the data.
    """

    compound_id: str
    experiment_id: str
    model: Model
    success: bool
    message: str = ""
    values: dict[str, float] = field(default_factory=dict)
    stderr: dict[str, float] = field(default_factory=dict)
    free: tuple[str, ...] = ()
    covariance: np.ndarray | None = None
    n_data: int | None = None
    rss: float | None = None
    chi_square: float | None = None
    r_squared: float | None = None
    warnings: tuple[str, ...] = ()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted curve at concentrations ``x``."""
        return self.model.evaluate(x, **self.values)


@dataclass(frozen=True)
class FitResults:
    """Experiment-level fits of one model to a dataset.

    Attributes
    ----------
    model
        The fitted model.
    data
        The fitted observations.
    fixed
        Parameter values fixed in every fit.
    fits
        One result per compound and experiment, including failed fits.
    """

    model: Model
    data: DoseResponseData
    fixed: dict[str, float]
    fits: tuple[FitResult, ...]

    @property
    def free(self) -> list[str]:
        """Names of the fitted parameters."""
        return [p.name for p in self.model.parameters if p.name not in self.fixed]

    def experiments(self) -> pd.DataFrame:
        """One row per experiment-level fit with estimates and diagnostics."""
        rows = []
        for fit in self.fits:
            row = {
                "compound_id": fit.compound_id,
                "experiment_id": fit.experiment_id,
                "success": fit.success,
                "message": fit.message,
                "warnings": "; ".join(fit.warnings) or None,
                "n_data": fit.n_data,
                "rss": fit.rss,
                "chi_square": fit.chi_square,
                "r_squared": fit.r_squared,
            }
            for name in self.free:
                row[name] = fit.values.get(name, np.nan)
                row[f"{name}_stderr"] = fit.stderr.get(name, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """One row per compound summarizing successful fits across experiments.

        Native parameters get the arithmetic mean, sample SD, SEM and a
        Student-t 95% confidence interval. Concentration parameters get the
        same statistics on log10 values (``logIC50``, ``logIC50_SD``, ...),
        the geometric mean (``IC50``) and the back-transformed interval.
        """
        rows = []
        for compound_id in self.data.compounds:
            fits = [fit for fit in self.fits if fit.compound_id == compound_id]
            successful = [fit for fit in fits if fit.success]
            row = {
                "compound_id": compound_id,
                "N_fit": len(successful),
                "N_failed": len(fits) - len(successful),
                "N_flagged": sum(bool(fit.warnings) for fit in successful),
            }
            for name in self.free:
                values = np.array([fit.values[name] for fit in successful])
                if self.model.parameter(name).concentration:
                    log = _statistics(np.log10(values))
                    row[name] = 10 ** log["mean"]
                    row[f"log{name}"] = log["mean"]
                    row[f"log{name}_SD"] = log["SD"]
                    row[f"log{name}_SEM"] = log["SEM"]
                    row[f"{name}_CI95_lower"] = 10 ** log["CI95_lower"]
                    row[f"{name}_CI95_upper"] = 10 ** log["CI95_upper"]
                else:
                    statistics = _statistics(values)
                    row[name] = statistics.pop("mean")
                    row.update({f"{name}_{key}": v for key, v in statistics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def parameters(self, compound_id: str) -> dict[str, float]:
        """Summary parameter values of one compound, ready for `Model.evaluate`.

        Fitted parameters take their `summary` centers (geometric means for
        concentrations); fixed parameters keep their fixed values.
        """
        summary = self.summary().set_index("compound_id").loc[compound_id]
        if summary["N_fit"] == 0:
            raise ValueError(f"Compound {compound_id!r} has no successful fits.")
        return {**{name: float(summary[name]) for name in self.free}, **self.fixed}

    def report(
        self,
        parameter: str | None = None,
        *,
        uncertainty: Literal["sd", "sem", "ci95"] = "sd",
        log: bool = False,
        digits: int = 2,
        unit: str | None = None,
    ) -> pd.DataFrame:
        """Format one concentration parameter per compound for publication.

        Parameters
        ----------
        parameter
            Concentration parameter to report. Defaults to the only fitted one.
        uncertainty
            Spread across experiments. On the linear scale, SD and SEM are
            shown as the back-transformed range ``10**(mean ± spread)``.
        log
            Report log10 values, e.g. ``-6.12 ± 0.15``, instead of linear ones.
        digits
            Significant figures for linear values, decimals for log values.
        unit
            Unit appended to linear values.
        """
        if uncertainty not in ("sd", "sem", "ci95"):
            raise ValueError("uncertainty must be 'sd', 'sem' or 'ci95'.")
        candidates = [
            name for name in self.free if self.model.parameter(name).concentration
        ]
        if parameter is None:
            if len(candidates) != 1:
                raise ValueError(f"Choose the parameter to report from {candidates}.")
            parameter = candidates[0]
        elif parameter not in candidates:
            raise ValueError(f"Choose the parameter to report from {candidates}.")

        summary = self.summary()
        texts = []
        for _, row in summary.iterrows():
            if row["N_fit"] == 0:
                texts.append("no successful fit")
                continue
            mean = row[f"log{parameter}"]
            if uncertainty == "ci95":
                interval = np.log10(
                    [row[f"{parameter}_CI95_lower"], row[f"{parameter}_CI95_upper"]]
                )
            else:
                spread = row[f"log{parameter}_{uncertainty.upper()}"]
                interval = np.array([mean - spread, mean + spread])
            texts.append(_format(mean, interval, uncertainty, log, digits, unit))
        report = summary[["compound_id", "N_fit", "N_failed", "N_flagged"]].copy()
        report.insert(1, "report", texts)
        return report


def _statistics(values: np.ndarray) -> dict[str, float]:
    """Mean, sample SD, SEM and Student-t 95% interval; NaN where undefined."""
    n = len(values)
    mean = float(np.mean(values)) if n > 0 else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = sd / math.sqrt(n) if n > 1 else np.nan
    half_width = float(student_t.ppf(0.975, n - 1)) * sem if n > 1 else np.nan
    return {
        "mean": mean,
        "SD": sd,
        "SEM": sem,
        "CI95_lower": mean - half_width,
        "CI95_upper": mean + half_width,
    }


def _format(mean, interval, uncertainty, log, digits, unit) -> str:
    if log:
        text = f"{mean:.{digits}f}"
        if np.all(np.isfinite(interval)):
            if uncertainty == "ci95":
                text += f" [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"
            else:
                text += f" ± {(interval[1] - interval[0]) / 2:.{digits}f}"
        return text
    text = _significant(10**mean, digits)
    if np.all(np.isfinite(interval)):
        low, high = (_significant(10**value, digits) for value in interval)
        text += f" [{low}, {high}]"
    return f"{text} {unit}" if unit else text


def _significant(value: float, digits: int) -> str:
    """Format a positive value to ``digits`` significant figures."""
    rounded = float(f"{value:.{digits - 1}e}")
    exponent = math.floor(math.log10(rounded))
    if -4 <= exponent < 6:
        return f"{rounded:.{max(digits - 1 - exponent, 0)}f}"
    return f"{rounded:.{digits - 1}e}"
