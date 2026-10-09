"""Incomplete (four-state) competition with a ternary receptor complex."""

from __future__ import annotations

import numpy as np

from bindcurve.models.base import PLATEAUS, BindingModel, Parameter
from bindcurve.models.three_state import free_receptor


class FourStateModel(BindingModel):
    """Tracer and competitor may bind receptor simultaneously.

    ``Kd3 = [RL][Ls] / [RLLs]`` is the tracer dissociation constant from the
    competitor-bound receptor; thermodynamic consistency fixes the fourth
    constant. The concentration axis is total competitor ``LT``. With
    ``nonspecific=True``, competitor is also immobilized as ``N * L``.
    """

    def __init__(self, *, nonspecific: bool) -> None:
        self.nonspecific = nonspecific
        self.name = "comp_4st_total" if nonspecific else "comp_4st_specific"
        self.parameters = (
            *PLATEAUS,
            *(
                Parameter(name, concentration=True, fixed=True)
                for name in ("RT", "LsT", "Kds", "Kd3")
            ),
            *([Parameter("N", fixed=True, min=0.0)] if nonspecific else []),
            Parameter("Kd", concentration=True),
        )

    def _species(self, LT, *, RT, LsT, Kds, Kd3, Kd, N=0.0, **_):
        # Nonspecific binding is equivalent to an effective (1 + N) * Kd acting
        # on an apparent free competitor (1 + N) * L.
        K = (1.0 + N) * Kd
        # Given free receptor, the free ligands follow from a quadratic. A
        # consistent four-state system has one positive equilibrium and the
        # receptor balance changes sign on [0, RT], so bracketing finds it.
        R = free_receptor(_receptor_balance, LT, RT, LsT, Kds, K, Kd3)
        L_apparent, Ls = _free_ligands(R, LT, LsT, Kds, K, Kd3)
        RLLs = R * L_apparent * Ls / (K * Kd3)
        RLs = R * Ls / Kds
        L = L_apparent / (1.0 + N)
        species = {
            "R": R,
            "Ls": Ls,
            "L": L,
            "RLs": RLs,
            "RL": R * L_apparent / K,
            "RLLs": RLLs,
            "Fbs": (RLs + RLLs) / LsT,
        }
        if self.nonspecific:
            species["L_nonspecific"] = N * L
        return species


def _free_ligands(R, LT, LsT, Kds, Kd, Kd3):
    """Free competitor and tracer for given free receptor ``R``.

    Eliminating free tracer from both ligand balances gives
    ``A*L**2 + B*L + C = 0`` with exactly one non-negative root.
    """
    a = 1.0 + R / Kds
    b = 1.0 + R / Kd
    c = R / (Kd * Kd3)
    A = b * c
    B = a * b + c * (LsT - LT)
    C = -a * LT
    root = np.sqrt(np.maximum(B**2 - 4.0 * A * C, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        # Each form avoids cancellation for its sign of B; -C / B is the A -> 0 limit.
        L = np.where(B >= 0.0, -2.0 * C / (B + root), (root - B) / (2.0 * A))
        L = np.where(A > np.finfo(float).tiny, L, -C / B)
    L = np.maximum(L, 0.0)
    return L, LsT / (a + c * L)


def _receptor_balance(r, lt, lst, kds, kd, kd3):
    """Normalized ``R + RLs + RL + RLLs - RT`` as a function of free receptor."""
    L, Ls = _free_ligands(r, lt, lst, kds, kd, kd3)
    return r * (1.0 + Ls / kds + L / kd + L * Ls / (kd * kd3)) - 1.0
