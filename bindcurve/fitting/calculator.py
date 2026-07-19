from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import lmfit
import numpy as np

from bindcurve.datasets import CompoundData, DoseResponseData
from bindcurve.fitting.settings import FitSettings
from bindcurve.modeling import BaseDoseResponseModel, get_model
from bindcurve.results.core import FitResults
from bindcurve.results.types import FitMetrics, FitResult, ParameterEstimate


@dataclass
class _FitCalculator:
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
        compound_ids = data.resolve_compounds(compounds)
        fits: list[FitResult] = []

        for compound_id in compound_ids:
            compound = data.select_compound(compound_id)
            for experiment_id in compound.experiments:
                try:
                    fit_data = compound.select_experiment(experiment_id)
                    fits.append(
                        self._fit_one(
                            fit_data,
                            experiment_id=experiment_id,
                            fixed=fixed,
                            bounds=bounds,
                        )
                    )
                except Exception as error:
                    if self.settings.errors == "raise":
                        raise
                    fits.append(
                        FitResult.failed(
                            model=self.model,
                            compound_id=str(compound_id),
                            experiment_id=experiment_id,
                            stage="fit_experiment",
                            error=error,
                        )
                    )

        return FitResults(
            model=self.model,
            fit_results=tuple(fits),
        )

    def _fit_one(
        self,
        compound: CompoundData,
        *,
        experiment_id: str | None,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> FitResult:
        observations = compound.fit_observations()
        concentration = observations["concentration"].to_numpy(dtype=float)
        y = observations["response"].to_numpy(dtype=float)
        sigma = (
            observations["sigma"].to_numpy(dtype=float)
            if "sigma" in observations.columns
            else None
        )

        guesses = self.model.guess(compound)
        parameters = self.model.make_lmfit_parameters(
            guesses,
            fixed=fixed,
            bounds=bounds,
        )
        n_varying = sum(parameter.vary for parameter in parameters.values())
        if len(y) <= n_varying:
            raise ValueError(
                "A fit requires more concentration observations than varying "
                f"parameters; got {len(y)} observations and {n_varying} parameters."
            )

        if n_varying == 0:
            return self._build_result(
                compound=compound,
                experiment_id=experiment_id,
                parameters=parameters,
                concentration=concentration,
                y=y,
                sigma=sigma,
                covariance=None,
                variable_names=(),
                success=True,
                optimizer_message="No optimization: all parameters were fixed.",
            )

        minimizer = lmfit.Minimizer(
            self.model.residual,
            parameters,
            fcn_args=(concentration, y, sigma),
            scale_covar=sigma is None,
        )
        lmfit_result = minimizer.minimize(
            method=self.settings.lmfit_method,
            max_nfev=self.settings.max_nfev,
        )

        covariance = None
        variable_names = tuple(lmfit_result.var_names)
        if lmfit_result.covar is not None:
            optimizer_covariance = np.asarray(lmfit_result.covar, dtype=float)
            transform = self.model.parameter_jacobian(
                lmfit_result.params,
                variable_names,
            )
            covariance = transform @ optimizer_covariance @ transform.T
        return self._build_result(
            compound=compound,
            experiment_id=experiment_id,
            parameters=lmfit_result.params,
            concentration=concentration,
            y=y,
            sigma=sigma,
            covariance=covariance,
            variable_names=variable_names,
            success=bool(lmfit_result.success),
            optimizer_message=str(lmfit_result.message),
        )

    def _build_result(
        self,
        *,
        compound: CompoundData,
        experiment_id: str | None,
        parameters: lmfit.Parameters,
        concentration: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray | None,
        covariance: np.ndarray | None,
        variable_names: tuple[str, ...],
        success: bool,
        optimizer_message: str,
    ) -> FitResult:
        public_values = self.model.decode_parameters(parameters)
        predicted = self.model.evaluate(concentration, **public_values)
        metrics = self._calculate_metrics(
            y=y,
            predicted=predicted,
            sigma=sigma,
            n_varying=len(variable_names),
        )
        variable_index = {name: index for index, name in enumerate(variable_names)}

        estimates = {
            name: ParameterEstimate(
                name=name,
                value=public_values[name],
                stderr=(
                    None
                    if covariance is None or name not in variable_index
                    else float(
                        np.sqrt(
                            max(
                                covariance[
                                    variable_index[name],
                                    variable_index[name],
                                ],
                                0.0,
                            )
                        )
                    )
                ),
                vary=bool(parameter.vary),
                min=self._decode_bound(name, float(parameter.min)),
                max=self._decode_bound(name, float(parameter.max)),
            )
            for name, parameter in parameters.items()
        }

        return FitResult(
            model=self.model,
            compound_id=compound.compound_id,
            experiment_id=experiment_id,
            success=success,
            parameters=estimates,
            metrics=metrics,
            covariance=covariance,
            variable_names=variable_names,
            optimizer_message=optimizer_message,
            failure_stage=None if success else "optimization",
            error_type=None if success else "OptimizationFailure",
            error_message=None if success else optimizer_message,
        )

    @staticmethod
    def _calculate_metrics(
        *,
        y: np.ndarray,
        predicted: np.ndarray,
        sigma: np.ndarray | None,
        n_varying: int,
    ) -> FitMetrics:
        raw_residual = y - predicted
        n_data = len(y)
        degrees_of_freedom = n_data - n_varying
        rss = float(np.sum(raw_residual**2))
        reduced_rss = (
            float(rss / degrees_of_freedom) if degrees_of_freedom > 0 else None
        )
        chi_square = (
            float(np.sum((raw_residual / sigma) ** 2))
            if sigma is not None
            else None
        )
        reduced_chi_square = (
            float(chi_square / degrees_of_freedom)
            if chi_square is not None and degrees_of_freedom > 0
            else None
        )
        if sigma is None:
            likelihood_term = -np.inf if rss == 0.0 else n_data * np.log(rss / n_data)
            aic = float(likelihood_term + 2.0 * n_varying)
            bic = float(likelihood_term + n_varying * np.log(n_data))
        else:
            # Known heteroscedastic sigma requires the actual Gaussian
            # likelihood, including its observation-specific normalization.
            negative_twice_log_likelihood = float(
                chi_square
                + np.sum(np.log(2.0 * np.pi) + 2.0 * np.log(sigma))
            )
            aic = float(negative_twice_log_likelihood + 2.0 * n_varying)
            bic = float(
                negative_twice_log_likelihood + n_varying * np.log(n_data)
            )
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else None
        return FitMetrics(
            n_data=n_data,
            n_varying_parameters=n_varying,
            rss=rss,
            reduced_rss=reduced_rss,
            chi_square=chi_square,
            reduced_chi_square=reduced_chi_square,
            aic=aic,
            bic=bic,
            r_squared=r_squared,
        )

    def _decode_bound(self, name: str, value: float) -> float:
        spec = self.model.parameter_spec(name)
        if spec.scale == "log10":
            return float(10**value)
        return value


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
    calculator = _FitCalculator(model=model_object, settings=settings or FitSettings())
    return calculator.fit(data, compounds=compounds, fixed=fixed, bounds=bounds)
