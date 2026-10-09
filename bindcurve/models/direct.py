"""Direct binding of a labeled tracer to a titrated receptor."""

from __future__ import annotations

import numpy as np

from bindcurve.models.base import PLATEAUS, BindingModel, Parameter


class DirectSimpleModel(BindingModel):
    """One-site binding with free receptor on the concentration axis.

    ``Fbs = R / (Kds + R)``. Use it when total receptor approximates free
    receptor, i.e. when tracer depletion is negligible.
    """

    name = "dir_simple"
    parameters = (*PLATEAUS, Parameter("Kds", concentration=True))

    def _species(self, R, *, Kds, **_):
        return {"R": R, "Fbs": R / (Kds + R)}


class DirectModel(BindingModel):
    """One-site binding with tracer depletion, titrating total receptor ``RT``.

    With ``nonspecific=True``, tracer is additionally immobilized in proportion
    to its free concentration, ``Ls_nonspecific = Ns * Ls`` (Roehrl, Wang &
    Wagner, Biochemistry 2004, 43, 16056). The fitted plateaus absorb the
    nonspecific baseline, so the response maps the specific fraction
    ``Fbs = RLs / LsT``.
    """

    def __init__(self, *, nonspecific: bool) -> None:
        self.nonspecific = nonspecific
        self.name = "dir_total" if nonspecific else "dir_specific"
        self.parameters = (
            *PLATEAUS,
            Parameter("LsT", concentration=True, fixed=True),
            *([Parameter("Ns", fixed=True, min=0.0)] if nonspecific else []),
            Parameter("Kds", concentration=True),
        )

    def _species(self, RT, *, LsT, Kds, Ns=0.0, **_):
        # Receptor balance R**2 + a*R - K*RT = 0 with K = (1 + Ns) * Kds, solved
        # in the form that avoids cancellation for either sign of a.
        K = (1.0 + Ns) * Kds
        a = K + LsT - RT
        root = np.sqrt(a**2 + 4.0 * K * RT)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.where(a >= 0.0, 2.0 * K * RT / (a + root), (root - a) / 2.0)
        Ls = LsT / (1.0 + Ns + R / Kds)
        RLs = R * Ls / Kds
        species = {"R": R, "Ls": Ls, "RLs": RLs, "Fbs": RLs / LsT}
        if self.nonspecific:
            species["Ls_nonspecific"] = Ns * Ls
        return species
