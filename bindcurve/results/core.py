from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.results.summaries import summarize_fit_parameters
from bindcurve.results.types import (
    ConcentrationSummary,
    FitResult,
    ParameterSummary,
    ReportRepresentation,
    ReportUncertainty,
    RoundingMode,
    SummaryRecord,
)


@dataclass(frozen=True)
class FitResults:
    """Collection of individual fits and parameter summaries."""

    model: BaseDoseResponseModel
    fit_results: tuple[FitResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_results", tuple(self.fit_results))
        if any(fit.model is not self.model for fit in self.fit_results):
            raise ValueError("Every fit result must reference FitResults.model.")
        identities = [
            (
                str(fit.compound_id),
                None if fit.experiment_id is None else str(fit.experiment_id),
            )
            for fit in self.fit_results
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "Duplicate fit results share the same compound and experiment "
                "identifiers."
            )
        successful = [fit for fit in self.fit_results if fit.success]
        expected_parameters = {spec.name for spec in self.model.parameter_specs}
        for fit in successful:
            actual_parameters = set(fit.parameters)
            if actual_parameters != expected_parameters:
                raise ValueError(
                    f"Successful fit {fit.compound_id!r}/{fit.experiment_id!r} "
                    "does not match the model parameter schema. "
                    f"Missing: {sorted(expected_parameters - actual_parameters)}; "
                    f"unknown: {sorted(actual_parameters - expected_parameters)}."
                )
        for spec in self.model.parameter_specs:
            estimates = [fit.parameters[spec.name] for fit in successful]
            if not estimates:
                continue
            vary_values = {estimate.vary for estimate in estimates}
            if len(vary_values) != 1:
                raise ValueError(
                    f"Parameter {spec.name!r} must use one fixed/varying status "
                    "throughout a FitResults collection."
                )
            if not spec.vary and True in vary_values:
                raise ValueError(
                    f"Model-fixed parameter {spec.name!r} cannot vary in a fit."
                )
            if False in vary_values:
                values = np.asarray(
                    [estimate.value for estimate in estimates],
                    dtype=float,
                )
                if not np.allclose(values, values[0], rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"Fixed parameter {spec.name!r} must have one global value."
                    )

    @property
    def summaries(self) -> tuple[SummaryRecord, ...]:
        """Derive across-experiment summaries from immutable fit results."""
        return tuple(
            summarize_fit_parameters(
                self.fit_results,
                parameter_specs=self.model.parameter_specs,
            )
        )

    def successful(self) -> list[FitResult]:
        """Return successful fits."""
        return [fit for fit in self.fit_results if fit.success]

    def failed(self) -> list[FitResult]:
        """Return failed fits."""
        return [fit for fit in self.fit_results if not fit.success]

    def _fits_by_compound(self) -> dict[str, list[FitResult]]:
        """Group fits once, retaining their original compound and experiment order."""
        grouped: dict[str, list[FitResult]] = {}
        for fit in self.fit_results:
            grouped.setdefault(str(fit.compound_id), []).append(fit)
        return grouped

    def fit_summary(self) -> pd.DataFrame:
        """Represent individual fits as a DataFrame."""
        rows: list[dict[str, object]] = []
        for fit in self.fit_results:
            row: dict[str, object] = {
                "compound_id": fit.compound_id,
                "experiment_id": fit.experiment_id,
                "model": fit.model_name,
                "success": fit.success,
                "failure_stage": fit.failure_stage,
                "error_type": fit.error_type,
                "error_message": fit.error_message,
                "optimizer_message": fit.optimizer_message,
                "warnings": "; ".join(fit.warnings) or None,
            }
            if fit.metrics is not None:
                row.update(
                    {
                        "n_data": fit.metrics.n_data,
                        "n_varying_parameters": fit.metrics.n_varying_parameters,
                        "rss": fit.metrics.rss,
                        "reduced_rss": fit.metrics.reduced_rss,
                        "chi_square": fit.metrics.chi_square,
                        "reduced_chi_square": fit.metrics.reduced_chi_square,
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

    def fixed_parameters(self) -> pd.DataFrame:
        """Return fixed assay and response parameters separately from estimates."""
        rows: list[dict[str, object]] = []
        for fit in self.fit_results:
            for estimate in fit.parameters.values():
                if estimate.vary:
                    continue
                rows.append(
                    {
                        "compound_id": fit.compound_id,
                        "experiment_id": fit.experiment_id,
                        "parameter": estimate.name,
                        "value": estimate.value,
                    }
                )
        return pd.DataFrame(rows)

    def parameter_values(self, compound_id: str) -> dict[str, float]:
        """Return one transparent across-experiment parameter set.

        Varying native parameters use their arithmetic means, varying
        concentration parameters use their geometric centers, and globally
        fixed parameters retain their common value.
        """
        compound_fits = [
            fit for fit in self.successful() if str(fit.compound_id) == str(compound_id)
        ]
        if not compound_fits:
            raise KeyError(f"Compound {compound_id!r} has no successful fits.")
        summary_lookup = {
            summary.parameter: summary
            for summary in summarize_fit_parameters(
                compound_fits, parameter_specs=self.model.parameter_specs
            )
        }

        values: dict[str, float] = {}
        for spec in self.model.parameter_specs:
            estimates = [
                fit.parameters[spec.name]
                for fit in compound_fits
                if spec.name in fit.parameters
            ]
            if not estimates:
                raise KeyError(
                    f"Fit results for {compound_id!r} lack parameter {spec.name!r}."
                )
            if not any(estimate.vary for estimate in estimates):
                fixed_values = np.asarray(
                    [estimate.value for estimate in estimates],
                    dtype=float,
                )
                if not np.allclose(
                    fixed_values,
                    fixed_values[0],
                    rtol=1.0e-9,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        f"Fixed parameter {spec.name!r} differs across experiments."
                    )
                values[spec.name] = float(fixed_values[0])
                continue

            summary = summary_lookup[spec.name]
            if isinstance(summary, ConcentrationSummary):
                values[spec.name] = summary.center
            else:
                values[spec.name] = summary.mean
        return values

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
        if not self.fit_results:
            return pd.DataFrame()

        fits_by_compound = self._fits_by_compound()
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
        for compound_id, all_compound_fits in fits_by_compound.items():
            compound_fits = [fit for fit in all_compound_fits if fit.success]
            row: dict[str, object] = {
                "compound_id": compound_id,
                "N_exp": _compound_N_exp(all_compound_fits),
                "N_fit_successful": len(compound_fits),
                "N_fit_failed": len(all_compound_fits) - len(compound_fits),
                "N_fit_flagged": sum(bool(fit.warnings) for fit in compound_fits),
                "N_obs": _compound_n_obs(compound_fits),
            }

            for name in parameter_order:
                summary = summary_lookup.get((compound_id, name))
                if summary_kind[name] == "concentration":
                    _update_concentration_summary_row(row, name, summary)
                else:
                    _update_native_summary_row(row, name, summary)

            row["RSS"] = _compound_rss(compound_fits)
            row["Chi_squared"] = _compound_chi_squared(compound_fits)
            rows.append(row)

        columns = [
            "compound_id",
            "N_exp",
            "N_fit_successful",
            "N_fit_failed",
            "N_fit_flagged",
            "N_obs",
        ]
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
        columns.extend(["RSS", "Chi_squared"])
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
        from bindcurve.results.reporting import format_results_report

        return format_results_report(
            self,
            parameter=parameter,
            compounds=compounds,
            representation=representation,
            uncertainty=uncertainty,
            rounding=rounding,
            places_mean=places_mean,
            places_uncertainty=places_uncertainty,
            unit=unit,
            include_n_exp=include_n_exp,
        )


def _compound_N_exp(fits: list[FitResult]) -> int:
    experiment_ids = {
        str(fit.experiment_id) for fit in fits if fit.experiment_id is not None
    }
    return len(experiment_ids) if experiment_ids else len(fits)


def _compound_n_obs(fits: list[FitResult]) -> int | None:
    n_data = [fit.metrics.n_data for fit in fits if fit.metrics is not None]
    if not n_data:
        return None
    return int(sum(n_data))


def _compound_rss(fits: list[FitResult]) -> float | None:
    values = [float(fit.metrics.rss) for fit in fits if fit.metrics is not None]
    if not values:
        return None
    return float(sum(values))


def _compound_chi_squared(fits: list[FitResult]) -> float | None:
    values = [
        float(fit.metrics.chi_square)
        for fit in fits
        if fit.metrics is not None and fit.metrics.chi_square is not None
    ]
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
