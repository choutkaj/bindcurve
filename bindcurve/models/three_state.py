"""Complete (three-state) competition between a tracer and a competitor."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from bindcurve.models.base import PLATEAUS, BindingModel, Parameter


class ThreeStateModel(BindingModel):
    """Tracer and competitor bind receptor mutually exclusively.

    The concentration axis is total competitor ``LT``. With
    ``nonspecific=True``, competitor is also immobilized in proportion to its
    free concentration, ``L_nonspecific = N * L``, which enters the receptor
    balance as an effective ``(1 + N) * Kd``. ``Kd`` stays the microscopic
    constant.
    """

    def __init__(self, *, nonspecific: bool) -> None:
        self.nonspecific = nonspecific
        self.name = "comp_3st_total" if nonspecific else "comp_3st_specific"
        self.parameters = (
            *PLATEAUS,
            *(
                Parameter(name, concentration=True, fixed=True)
                for name in ("RT", "LsT", "Kds")
            ),
            *([Parameter("N", fixed=True, min=0.0)] if nonspecific else []),
            Parameter("Kd", concentration=True),
        )

    def _species(self, LT, *, RT, LsT, Kds, Kd, N=0.0, **_):
        K = (1.0 + N) * Kd
        R = free_receptor(_receptor_balance, LT, RT, LsT, Kds, K)
        Ls = LsT * Kds / (Kds + R)
        L = LT * Kd / (K + R)
        species = {
            "R": R,
            "Ls": Ls,
            "L": L,
            "RLs": R * Ls / Kds,
            "RL": R * L / Kd,
            "Fbs": R / (Kds + R),
        }
        if self.nonspecific:
            species["L_nonspecific"] = N * L
        return species


def _receptor_balance(r, lt, lst, kds, k):
    """Normalized ``R + RLs + RL - RT`` as a function of free receptor ``r``."""
    # RL - RT is combined algebraically: near LT = RT, subtracting 1 from a
    # nearly saturated RL would lose the terms that set free receptor.
    return r + lst * r / (kds + r) + ((lt - 1.0) * r - k) / (k + r)


def free_receptor(balance, LT, RT, *constants) -> np.ndarray:
    """Solve a receptor balance for free receptor at every total competitor.

    ``balance(r, lt, *constants)`` is the balance with every concentration
    divided by ``RT``, which puts the root in [0, 1] for any unit.
    """
    scaled = [constant / RT for constant in constants]
    roots = [
        brentq(
            balance,
            0.0,
            1.0,
            args=(lt / RT, *scaled),
            # An absolute tolerance below any root keeps tiny physical roots.
            xtol=np.finfo(float).tiny,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=1000,
        )
        for lt in np.ravel(LT)
    ]
    return RT * np.reshape(roots, np.shape(LT))
