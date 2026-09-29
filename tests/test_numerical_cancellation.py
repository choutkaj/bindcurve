"""Cancellation regressions exercised through the public model API."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import bindcurve as bc


def assert_relative(actual, expected):
    # A default absolute tolerance would hide zeroed tiny concentrations.
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)


def three_state_reference(*, RT, LT, LsT, Kds, Kd, N=0):
    """Independent high-precision bisection of the original physical balance."""
    with localcontext() as context:
        context.prec = 80
        rt, lt, lst, ks, kd, n = map(
            lambda value: Decimal(str(value)), (RT, LT, LsT, Kds, Kd, N)
        )
        lower, upper = Decimal(0), rt
        for _ in range(300):
            r = (lower + upper) / 2
            balance = r + lst * r / (ks + r) + lt * r / ((1 + n) * kd + r)
            if balance > rt:
                upper = r
            else:
                lower = r
        return {
            "R": float(r),
            "Fbs": float(r / (ks + r)),
            "Ls": float(lst * ks / (ks + r)),
            "L": float(lt * kd / ((1 + n) * kd + r)),
        }


def check_three_state_balances(c, *, Kds, Kd, N=0):
    assert_relative(c["R"] + c["RLs"] + c["RL"], c["RT"])
    assert_relative(c["Ls"] + c["RLs"], c["LsT"])
    assert_relative((1 + N) * c["L"] + c["RL"], c["LT"])
    assert_relative(c["R"] * c["Ls"], Kds * c["RLs"])
    assert_relative(c["R"] * c["L"], Kd * c["RL"])
    assert_relative(c["Fbs"], c["RLs"] / c["LsT"])
    if "L_nonspecific_bound" in c:
        assert_relative(c["L_nonspecific_bound"], N * c["L"])
        assert_relative(c["L_bound_total"], c["RL"] + N * c["L"])


@pytest.mark.parametrize(
    ("RT", "LsT", "expected"),
    [(1, 1e-16, 0.5), (1e8, 1e-8, 0.9999999900000001), (1e16, 1, 1)],
)
def test_direct_tiny_bound_tracer(RT, LsT, expected):
    params = dict(ymin=0, ymax=1, LsT=LsT, Kds=1)
    result = bc.get_model("dir_specific").evaluate_components([RT], **params)
    c = result.components
    total = bc.get_model("dir_total").evaluate_components([RT], Ns=0, **params)
    assert_relative(result.response, expected)
    for name in c:
        assert_relative(c[name], total.components[name])
    assert_relative(c["R"] + c["RLs"], RT)
    assert_relative(c["Ls"] + c["RLs"], LsT)
    assert_relative(c["R"] * c["Ls"], c["RLs"])
    assert_relative(c["Fbs"], c["RLs"] / LsT)


@pytest.mark.parametrize(
    ("model_name", "extra"),
    [
        ("comp_3st_specific", {}),
        ("comp_3st_total", {"N": 0}),
        ("comp_3st_total", {"N": 0.35}),
    ],
)
@pytest.mark.parametrize("tiny_species", ["L", "Ls"])
def test_three_state_tiny_free_ligand(model_name, extra, tiny_species):
    if tiny_species == "L":
        LT, LsT, Kds, Kd = 1e-4, 1e-16, 1, 1e-20
    else:
        LT, LsT, Kds, Kd = 1e-16, 1e-4, 1e-20, 1
    params = dict(RT=1, LsT=LsT, Kds=Kds, Kd=Kd, **extra)
    result = bc.get_model(model_name).evaluate_components(
        [LT], ymin=0, ymax=1, **params
    )
    reference = three_state_reference(LT=LT, **params)
    assert_relative(reference[tiny_species], 1.000100010001e-24)
    for name, expected in reference.items():
        assert_relative(result.components[name], expected)
    assert_relative(result.response, reference["Fbs"])
    check_three_state_balances(result.components, Kds=Kds, Kd=Kd, **extra)


@pytest.mark.parametrize(
    ("model_name", "extra"),
    [
        ("comp_3st_specific", {}),
        ("comp_3st_total", {"N": 0}),
        ("comp_3st_total", {"N": 0.35}),
    ],
)
@pytest.mark.parametrize("scale", [1e-9, 1, 1e9])
def test_three_state_tight_stoichiometric_binding(model_name, extra, scale):
    params = dict(
        RT=scale, LsT=1e-14 * scale, Kds=1e-12 * scale, Kd=1e-24 * scale, **extra
    )
    reference = three_state_reference(LT=scale, **params)
    if extra.get("N", 0) == 0:
        assert_relative(reference["Fbs"], 0.4993757812462231)
    result = bc.get_model(model_name).evaluate_components(
        [scale], ymin=0, ymax=1, **params
    )
    for name, expected in reference.items():
        assert_relative(result.components[name], expected)
    assert_relative(result.response, reference["Fbs"])
    check_three_state_balances(
        result.components, Kds=params["Kds"], Kd=params["Kd"], **extra
    )
