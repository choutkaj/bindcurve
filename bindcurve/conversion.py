"""IC50-to-Kd conversions for one-site competitive binding equilibria.

Each function accepts scalars or arrays (e.g. columns of
`FitResults.summary`) and returns the competitor dissociation constant in the
same concentration unit. IC50 values that are physically incompatible with the
assay constants give NaN.
"""

from __future__ import annotations

import numpy as np


def cheng_prusoff(IC50, *, LsT: float, Kds: float):
    """Cheng-Prusoff approximation, ``Kd = IC50 / (1 + LsT / Kds)``.

    Accurate when tracer depletion and receptor concentration are negligible.
    """
    IC50 = np.asarray(IC50, dtype=float)
    _check_positive(IC50=IC50, LsT=LsT, Kds=Kds)
    return _result(IC50 / (1.0 + LsT / Kds))


def munson_rodbard(IC50, *, LsT: float, Kds: float, y0: float):
    """Exact finite-concentration correction of Munson and Rodbard.

    ``y0`` is the bound-to-free tracer ratio without competitor. The correction
    term is subtracted, as in the authors' erratum (J Receptor Res. 1989, 9,
    511).
    """
    IC50 = np.asarray(IC50, dtype=float)
    _check_positive(IC50=IC50, LsT=LsT, Kds=Kds)
    if not y0 >= 0.0:
        raise ValueError("y0 must be non-negative.")
    denominator = 1.0 + LsT * (y0 + 2.0) / (2.0 * Kds * (y0 + 1.0)) + y0
    return _result(IC50 / denominator - Kds * y0 / (y0 + 2.0))


def coleska(IC50, *, RT: float, LsT: float, Kds: float):
    """Exact correction of Nikolovska-Coleska et al. from RT, LsT and Kds.

    Follows appendix A.4.2 of Anal Biochem. 2004, 332, 261-273.
    """
    IC50 = np.asarray(IC50, dtype=float)
    _check_positive(IC50=IC50, RT=RT, LsT=LsT, Kds=Kds)
    # Competitor-free state: R0**2 + a*R0 - Kds*RT = 0, solved without cancellation.
    a = LsT + Kds - RT
    root = np.hypot(a, 2.0 * np.sqrt(Kds * RT))
    R0 = 2.0 * Kds * RT / (a + root) if a >= 0.0 else (root - a) / 2.0
    Ls0 = LsT / (1.0 + R0 / Kds)
    RLs50 = RT * Ls0 / (Kds + Ls0) / 2.0
    Ls50 = LsT - RLs50
    RL50 = RT - Kds * RLs50 / Ls50 - RLs50
    L50 = IC50 - RL50
    return _result(L50 / (Ls50 / Kds + R0 / Kds + 1.0))


def _check_positive(**values) -> None:
    # NaN IC50 values, e.g. from compounds without successful fits, pass through.
    for name, value in values.items():
        value = np.asarray(value, dtype=float)
        if np.any(value <= 0.0) or np.any(np.isinf(value)):
            raise ValueError(f"{name} must be positive and finite.")


def _result(Kd: np.ndarray) -> float | np.ndarray:
    """Replace physically impossible (non-positive) constants with NaN."""
    Kd = np.where(Kd > 0.0, Kd, np.nan)
    return float(Kd) if Kd.ndim == 0 else Kd
