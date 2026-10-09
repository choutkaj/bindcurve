"""Empirical four-parameter inhibitory dose-response model."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from bindcurve.models.base import PLATEAUS, Model, Parameter


class IC50Model(Model):
    """Four-parameter inhibitory model, ``ymin + (ymax - ymin) / (1 + (x/IC50)**h)``.

    ``ymax`` is the response at zero concentration, ``ymin`` the response at
    high concentration, and the Hill slope ``h`` is positive.
    """

    name = "ic50"
    parameters = (
        *PLATEAUS,
        Parameter("IC50", concentration=True),
        Parameter("hill_slope", min=np.finfo(float).tiny),
    )

    def _fraction(self, x, *, IC50, hill_slope, **_):
        # 1 / (1 + (x/IC50)**h) written as a logistic in log x, finite at x = 0.
        with np.errstate(divide="ignore"):
            log_ratio = np.log(IC50) - np.log(x)
        return expit(hill_slope * log_ratio)

    def guess(self, x, y):
        return {**super().guess(x, y), "hill_slope": 1.0}
