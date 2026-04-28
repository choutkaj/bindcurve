from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.parameters import ParameterSpec


class IC50Model(BaseDoseResponseModel):
    """Four-parameter IC50 / Hill inhibition model.

    The model is written as::

        y = ymin + (ymax - ymin) / (1 + (IC50 / x) ** hill_slope)

    A negative ``hill_slope`` describes a decreasing inhibition curve, while a
    positive ``hill_slope`` describes an increasing response curve.
    """

    name = "ic50"
    concentration_parameters = frozenset({"IC50"})
    response_parameters = frozenset({"ymin", "ymax"})
    parameter_specs = (
        ParameterSpec("ymin", unit_kind="response"),
        ParameterSpec("ymax", unit_kind="response"),
        ParameterSpec("IC50", min=0.0, unit_kind="concentration"),
        ParameterSpec("hill_slope", initial=-1.0, unit_kind=None),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        IC50: float,
        hill_slope: float,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return ymin + (ymax - ymin) / (1.0 + (IC50 / x) ** hill_slope)

    def guess(self, compound: CompoundData) -> dict[str, float]:
        table = compound.aggregate_replicates(method="mean")
        response = table["response"].to_numpy(dtype=float)
        concentration = table["concentration"].to_numpy(dtype=float)

        ymin = float(np.nanmin(response))
        ymax = float(np.nanmax(response))
        midpoint = ymin + 0.5 * (ymax - ymin)
        midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
        ic50 = float(concentration[midpoint_index])

        # Default direction is inferred from the dose-response trend.
        hill_slope = 1.0 if response[-1] > response[0] else -1.0

        return {
            "ymin": ymin,
            "ymax": ymax,
            "IC50": ic50,
            "hill_slope": hill_slope,
        }
