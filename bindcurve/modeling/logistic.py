from __future__ import annotations

import numpy as np
from scipy.special import expit

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.guesses import midpoint_guess
from bindcurve.modeling.parameters import (
    STRICTLY_POSITIVE_PARAMETER_MIN,
    ParameterSpec,
)


def _fraction_response_from_ic50(
    concentration: np.ndarray,
    *,
    IC50: float,
    hill_slope: float,
) -> np.ndarray:
    concentration = np.asarray(concentration, dtype=float)
    log_concentration = np.full_like(concentration, -np.inf, dtype=float)
    np.log(concentration, out=log_concentration, where=concentration > 0.0)
    log_ratio = np.log(IC50) - log_concentration
    return expit(hill_slope * log_ratio)


class IC50Model(BaseDoseResponseModel):
    """Identifiable four-parameter inhibitory IC50 / Hill model.

    The model is written as::

        y = ymin + amplitude / (1 + (x / IC50) ** hill_slope)

    ``amplitude`` and ``hill_slope`` are strictly positive. Consequently the
    model has one inhibitory orientation and cannot reproduce the same curve by
    swapping asymptotes and reversing the Hill-slope sign.
    """

    name = "ic50"
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("amplitude", min=STRICTLY_POSITIVE_PARAMETER_MIN),
        ParameterSpec(
            "IC50",
            min=STRICTLY_POSITIVE_PARAMETER_MIN,
            kind="concentration",
            scale="log10",
            log_name="logIC50",
        ),
        ParameterSpec(
            "hill_slope",
            initial=1.0,
            min=STRICTLY_POSITIVE_PARAMETER_MIN,
        ),
    )

    def _component_arrays(
        self,
        concentration: np.ndarray,
        *,
        IC50: float,
        hill_slope: float,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return {
            "fraction_response": _fraction_response_from_ic50(
                concentration,
                IC50=IC50,
                hill_slope=hill_slope,
            )
        }

    def response_from_components(
        self,
        components: dict[str, np.ndarray],
        *,
        ymin: float,
        amplitude: float,
        **params: float,
    ) -> np.ndarray:
        fraction = np.asarray(components["fraction_response"], dtype=float)
        return float(ymin) + float(amplitude) * fraction

    def guess(self, compound: CompoundData) -> dict[str, float]:
        guesses = midpoint_guess(compound, concentration_parameter="IC50")
        ymin = guesses.pop("ymin")
        ymax = guesses.pop("ymax")
        return {
            "ymin": ymin,
            "amplitude": max(ymax - ymin, np.finfo(float).eps),
            "IC50": guesses["IC50"],
            "hill_slope": 1.0,
        }
