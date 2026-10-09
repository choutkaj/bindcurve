"""Independent scientific references; none of them call bindcurve's solvers."""

from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import brentq, least_squares


def three_state_reference(*, RT, LT, LsT, Kds, Kd, N=0.0):
    """80-digit bisection of the receptor balance R + RLs + RL = RT."""
    with localcontext() as context:
        context.prec = 80
        rt, lt, lst, kds, kd, n = (Decimal(str(v)) for v in (RT, LT, LsT, Kds, Kd, N))
        lower, upper = Decimal(0), rt
        for _ in range(300):
            r = (lower + upper) / 2
            if r + lst * r / (kds + r) + lt * r / ((1 + n) * kd + r) > rt:
                upper = r
            else:
                lower = r
        return {
            "R": float(r),
            "Ls": float(lst * kds / (kds + r)),
            "L": float(lt * kd / ((1 + n) * kd + r)),
            "Fbs": float(r / (kds + r)),
        }


def four_state_reference(*, RT, LT, LsT, Kds, Kd, Kd3):
    """Solve all three four-state mass balances simultaneously for Fbs."""

    def species(log_free):
        R, Ls, L = np.exp(log_free)
        return R, Ls, L, R * Ls / Kds, R * L / Kd, R * L * Ls / (Kd * Kd3)

    def residuals(log_free):
        R, Ls, L, RLs, RL, RLLs = species(log_free)
        return [
            (R + RLs + RL + RLLs) / RT - 1.0,
            (Ls + RLs + RLLs) / LsT - 1.0,
            (L + RL + RLLs) / LT - 1.0,
        ]

    starts = ([RT, LsT, LT], [RT * 1e-3, LsT * 1e-3, LT], [RT * 1e-6, LsT, LT * 0.5])
    best = min(
        (
            least_squares(residuals, np.log(s), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            for s in starts
        ),
        key=lambda solution: solution.cost,
    )
    assert best.cost < 1e-24
    _, _, _, RLs, _, RLLs = species(best.x)
    return (RLs + RLLs) / LsT


def competitive_ic50(*, RT, LsT, Kds, Kd):
    """Exact total competitor halving RLs in the three-state model, and y0."""
    R0 = brentq(lambda R: R + LsT * R / (Kds + R) - RT, 0.0, RT, xtol=1e-300)
    Ls0 = LsT / (1.0 + R0 / Kds)
    RLs50 = R0 * Ls0 / Kds / 2.0
    Ls50 = LsT - RLs50
    R50 = Kds * RLs50 / Ls50
    RL50 = RT - R50 - RLs50
    return Kd * RL50 / R50 + RL50, R0 / Kds


def observations(model, potency, fixed):
    """Total concentrations and exact responses (0-100) built from free species.

    Choosing the free titrant first avoids solving for it, so these data are
    independent of the production solvers.
    """
    free = np.logspace(-3, 2, 16)
    if model == "ic50":
        return free, 100.0 / (1.0 + free / potency)
    if model.startswith("dir_"):
        fraction = free / ((1.0 + fixed.get("Ns", 0.0)) * potency + free)
        return free + fixed.get("LsT", 0.0) * fraction, 100.0 * fraction

    RT, LsT, Kds = fixed["RT"], fixed["LsT"], fixed["Kds"]
    totals, responses = [], []
    for L in free:
        ternary = L / (potency * fixed["Kd3"]) if "Kd3" in fixed else 0.0
        tracer = 1.0 / Kds + ternary
        R = brentq(
            lambda R, L=L, tracer=tracer: (
                R * (1.0 + L / potency) + LsT * R * tracer / (1.0 + R * tracer) - RT
            ),
            0.0,
            RT,
            xtol=1e-15,
        )
        Ls = LsT / (1.0 + R * tracer)
        RL, RLs, RLLs = R * L / potency, R * Ls / Kds, R * Ls * ternary
        totals.append((1.0 + fixed.get("N", 0.0)) * L + RL + RLLs)
        responses.append(100.0 * (RLs + RLLs) / LsT)
    return np.array(totals), np.array(responses)


COMPETITIVE = {"RT": 0.05, "LsT": 0.005, "Kds": 0.02}
MODEL_CASES = [
    ("ic50", "IC50", {"hill_slope": 1.0}),
    ("dir_simple", "Kds", {}),
    ("dir_specific", "Kds", {"LsT": 0.4}),
    ("dir_total", "Kds", {"LsT": 0.4, "Ns": 0.25}),
    ("comp_3st_specific", "Kd", COMPETITIVE),
    ("comp_3st_total", "Kd", {**COMPETITIVE, "N": 0.35}),
    ("comp_4st_specific", "Kd", {**COMPETITIVE, "Kd3": 0.5}),
    ("comp_4st_total", "Kd", {**COMPETITIVE, "Kd3": 0.5, "N": 0.35}),
]
