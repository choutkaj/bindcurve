from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import lmfit
import numpy as np

from bindcurve.datasets import CompoundData, DoseResponseData
from bindcurve.fitting.settings import FitSettings
from bindcurve.modeling import BaseDoseResponseModel, get_model
from bindcurve.results.core import (
    FitMetrics,
    FitResult,
    FitResults,
    ParameterEstimate,
    summarize_fit_parameters,
)


@dataclass
class FitCalculator:
    """Generic lmfit-backed calculator for dose-response models."""

    model: BaseDoseResponseModel
    settings: FitSettings = field(default_factory=FitSettings)

    def fit(
        self,
        data: DoseResponseData,
        *,
        compounds: list[str] | None = None,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> FitResults:
        """Fit the configured model to dose-response data."""
        compound_ids = compounds or data.compounds
        fits: list[FitResult] = []

        for compound_id in compound_ids:
            try:
                compound = data.select_compound(compound_id)
                fit_inputs = self._make_fit_inputs(compound)

                for experiment_id, fit_data in fit_inputs:
                    fits.append(
                        self._fit_one(
                            fit_data,
                            experiment_id=experiment_id,
                            fixed=fixed,
                            bounds=bounds,
                        )
                    )
            except Exception as exc:
                if self.settings.errors == "raise":
                    raise
                fits.append(
                    FitResult.failed(
                        compound_id=str(compound_id),
                        model_name=self.model.name,
                        experiment_id=None,
                        message=str(exc),
                    )
                )

        summaries = summarize_fit_parameters(
            fits,
            concentration_parameters=self.model.concentration_parameters,
        )
        return FitResults(fits=fits, summaries=summaries)

    def _make_fit_inputs(
        self, compound: CompoundData
    ) -> list[tuple[str | None, CompoundData]]:
        if self.settings.strategy == "per_experiment":
            return [
                (experiment_id, compound.select_experiment(experiment_id))
                for experiment_id in compound.experiments
            ]

        if self.settings.strategy == "pooled":
            return [(None, compound)]

        if self.settings.strategy == "per_compound_summary":
            summary = compound.aggregate_replicates(
                method=self.settings.replicate_aggregation
            )
            summary["compound_id"] = compound.compound_id
            summary["experiment_id"] = "compound_summary"
            summary["replicate_id"] = "summary"
            return [
                (
                    "compound_summary",
                    CompoundData(
                        compound_id=compound.compound_id,
                        table=summary,
                        concentration_unit=compound.concentration_unit,
                        response_unit=compound.response_unit,
                    ),
                )
            ]

        raise ValueError(f"Unsupported strategy: {self.settings.strategy}")

    def _fit_one(
        self,
        compound: CompoundData,
        *,
        experiment_id: str | None,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> FitResult:
        concentration, y = compound.as_xy(
            aggregate_replicates=self.settings.strategy != "pooled",
            aggregation=self.settings.replicate_aggregation,
        )
        x = self.model.transform_x(concentration)

        guesses = self.model.guess(compound)
        parameters = self.model.make_lmfit_parameters(
            guesses,
            fixed=fixed,
            bounds=bounds,
        )

        minimizer = lmfit.Minimizer(
            self.model.residual,
            parameters,
            fcn_args=(x, y),
        )
        lmfit_result = minimizer.minimize(
            method=self.settings.lmfit_method,
            max_nfev=self.settings.max_nfev,
        )

        residual = np.asarray(lmfit_result.residual, dtype=float)
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        estimates = {
            name: ParameterEstimate(
                name=name,
                value=float(parameter.value),
                stderr=None if parameter.stderr is None else float(parameter.stderr),
                unit=self.model.parameter_unit(
                    name,
                    concentration_unit=compound.concentration_unit,
                    response_unit=compound.response_unit,
                ),
                vary=bool(parameter.vary),
            )
            for name, parameter in lmfit_result.params.items()
        }

        metrics = FitMetrics(
            n_data=int(lmfit_result.ndata),
            n_varying_parameters=int(lmfit_result.nvarys),
            chisqr=float(lmfit_result.chisqr),
            redchi=float(lmfit_result.redchi),
            aic=None if lmfit_result.aic is None else float(lmfit_result.aic),
            bic=None if lmfit_result.bic is None else float(lmfit_result.bic),
            r_squared=r_squared,
        )

        return FitResult(
            compound_id=compound.compound_id,
            experiment_id=experiment_id,
            model_name=self.model.name,
            success=bool(lmfit_result.success),
            message=lmfit_result.message,
            parameters=estimates,
            metrics=metrics,
            lmfit_result=lmfit_result,
        )


def fit(
    data: DoseResponseData,
    *,
    model: str | BaseDoseResponseModel = "ic50",
    settings: FitSettings | None = None,
    compounds: list[str] | None = None,
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> FitResults:
    """Convenience function for fitting a registered model to dose-response data."""
    model_object = get_model(model) if isinstance(model, str) else model
    calculator = FitCalculator(model=model_object, settings=settings or FitSettings())
    return calculator.fit(data, compounds=compounds, fixed=fixed, bounds=bounds)
