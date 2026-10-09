import numpy as np
import pytest
from references import four_state_reference, three_state_reference

import bindcurve as bc
from bindcurve.models import MODELS


def assert_relative(actual, expected, rtol=2e-13):
    # No absolute tolerance, so zeroed tiny concentrations cannot hide.
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)


def test_registry_lists_all_models():
    assert sorted(MODELS) == [
        "comp_3st_specific",
        "comp_3st_total",
        "comp_4st_specific",
        "comp_4st_total",
        "dir_simple",
        "dir_specific",
        "dir_total",
        "ic50",
    ]
    with pytest.raises(KeyError, match="Unknown model 'ec50'"):
        bc.get_model("ec50")


@pytest.mark.parametrize("hill_slope", [0.5, 2.0])
def test_ic50_midpoint_monotonicity_and_limits(hill_slope):
    x = np.array([0.0, 1e-300, 3.7e-3, 3.7, 3.7e3, 1e300])
    y = bc.get_model("ic50").evaluate(
        x, ymin=2.0, ymax=10.0, IC50=3.7, hill_slope=hill_slope
    )
    assert y[0] == 10.0
    assert y[3] == pytest.approx(6.0)
    assert y[-1] == pytest.approx(2.0)
    assert np.all(np.diff(y[1:]) < 0.0)


def test_models_reject_wrong_or_invalid_parameters():
    model = bc.get_model("ic50")
    valid = {"ymin": 0.0, "ymax": 100.0, "IC50": 1.0, "hill_slope": 1.0}
    with pytest.raises(TypeError, match="takes"):
        model.evaluate([1.0], **valid, amplitude=1.0)
    with pytest.raises(ValueError, match="hill_slope"):
        model.evaluate([1.0], **{**valid, "hill_slope": -1.0})
    with pytest.raises(ValueError, match="IC50 must be positive"):
        model.evaluate([1.0], **{**valid, "IC50": 0.0})
    with pytest.raises(ValueError, match="non-negative"):
        model.evaluate([-1.0], **valid)


def test_dir_simple_midpoint_and_limits():
    y = bc.get_model("dir_simple").evaluate(
        [0.0, 2.5, 2.5e12], ymin=0.0, ymax=1.0, Kds=2.5
    )
    assert y[0] == 0.0
    assert y[1] == pytest.approx(0.5)
    assert y[2] == pytest.approx(1.0, rel=1e-11)


def test_dir_specific_matches_roehrl_equation_6():
    RT = np.logspace(-4, 3, 80)
    total = 1.8 + 0.35 + RT
    # Rationalized Roehrl et al. eq 6, free of subtractive cancellation.
    expected = 2.0 * RT / (total + np.sqrt(total**2 - 4.0 * 0.35 * RT))
    observed = bc.get_model("dir_specific").evaluate(
        RT, ymin=0, ymax=1, LsT=0.35, Kds=1.8
    )
    assert_relative(observed, expected, rtol=2e-13)


def test_dir_total_mass_balances_and_apparent_kd_shift():
    LsT, Ns, Kds = 0.4, 0.25, 2.2
    model = bc.get_model("dir_total")
    RT = np.concatenate(([0.0], np.logspace(-4, 1, 50)))
    s = model.species(RT, ymin=0, ymax=1, LsT=LsT, Ns=Ns, Kds=Kds)
    np.testing.assert_allclose(s["R"] + s["RLs"], RT, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(
        s["Ls"] + s["Ls_nonspecific"] + s["RLs"], LsT, rtol=1e-13
    )
    np.testing.assert_allclose(Kds * s["RLs"], s["R"] * s["Ls"], rtol=1e-13, atol=1e-16)
    # Half saturation of the specific fraction at free receptor (1 + Ns) * Kds.
    midpoint = model.species(
        [(1 + Ns) * Kds + LsT / 2], ymin=0, ymax=1, LsT=LsT, Ns=Ns, Kds=Kds
    )
    assert midpoint["R"][0] == pytest.approx((1 + Ns) * Kds)
    assert midpoint["Fbs"][0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("RT", "LsT", "expected"),
    [(1, 1e-16, 0.5), (1e8, 1e-8, 0.9999999900000001), (1e16, 1, 1)],
)
def test_direct_binding_keeps_tiny_bound_tracer(RT, LsT, expected):
    specific = bc.get_model("dir_specific").species(
        [RT], ymin=0, ymax=1, LsT=LsT, Kds=1
    )
    total = bc.get_model("dir_total").species(
        [RT], ymin=0, ymax=1, LsT=LsT, Ns=0, Kds=1
    )
    assert_relative(specific["Fbs"], expected)
    for name, value in specific.items():
        assert_relative(total[name], value)
    assert_relative(specific["R"] + specific["RLs"], RT)
    assert_relative(specific["Ls"] + specific["RLs"], LsT)


def test_dir_specific_keeps_tiny_free_receptor_in_tracer_excess():
    s = bc.get_model("dir_specific").species([1.0], ymin=0, ymax=1, LsT=1e15, Kds=1.0)
    assert s["R"][0] == pytest.approx(1e-15, rel=1e-12)
    assert s["R"][0] + s["RLs"][0] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("N", [None, 0.0, 0.6])
def test_three_state_obeys_equilibria_and_mass_balances(N):
    name = "comp_3st_specific" if N is None else "comp_3st_total"
    extra = {} if N is None else {"N": N}
    LT = np.logspace(-7, 5, 100)
    s = bc.get_model(name).species(
        LT, ymin=0, ymax=1, RT=0.7, LsT=0.2, Kds=0.3, Kd=1.9, **extra
    )
    assert_relative(s["R"] + s["RLs"] + s["RL"], 0.7, rtol=2e-12)
    assert_relative(s["Ls"] + s["RLs"], 0.2, rtol=2e-12)
    assert_relative((1 + (N or 0)) * s["L"] + s["RL"], LT, rtol=2e-12)
    assert_relative(0.3 * s["RLs"], s["R"] * s["Ls"], rtol=2e-12)
    assert_relative(1.9 * s["RL"], s["R"] * s["L"], rtol=2e-12)
    if N is not None:
        assert_relative(s["L_nonspecific"], N * s["L"])


@pytest.mark.parametrize("N", [None, 0.35])
@pytest.mark.parametrize(
    ("LT", "LsT", "Kds", "Kd", "tiny"),
    [(1e-4, 1e-16, 1, 1e-20, "L"), (1e-16, 1e-4, 1e-20, 1, "Ls")],
)
def test_three_state_keeps_tiny_free_ligands(N, LT, LsT, Kds, Kd, tiny):
    name = "comp_3st_specific" if N is None else "comp_3st_total"
    extra = {} if N is None else {"N": N}
    params = dict(RT=1, LsT=LsT, Kds=Kds, Kd=Kd, **extra)
    reference = three_state_reference(LT=LT, **params)
    assert reference[tiny] == pytest.approx(1.000100010001e-24, rel=1e-12)
    s = bc.get_model(name).species([LT], ymin=0, ymax=1, **params)
    for key, value in reference.items():
        assert_relative(s[key], value)


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_three_state_tight_stoichiometric_binding(scale):
    params = dict(RT=scale, LsT=1e-14 * scale, Kds=1e-12 * scale, Kd=1e-24 * scale)
    reference = three_state_reference(LT=scale, **params)
    assert reference["Fbs"] == pytest.approx(0.4993757812462231, rel=1e-13)
    s = bc.get_model("comp_3st_specific").species([scale], ymin=0, ymax=1, **params)
    for key, value in reference.items():
        assert_relative(s[key], value)


def test_three_state_is_stable_for_extreme_concentration_ratios():
    RT = 2.172723469932662e-6
    s = bc.get_model("comp_3st_specific").species(
        [106139.16205049057],
        ymin=0,
        ymax=1,
        RT=RT,
        LsT=686450.9185295302,
        Kds=0.00012087951717196983,
        Kd=0.00001652150832486005,
    )
    assert 0.0 < s["R"][0] <= RT
    assert s["R"][0] + s["RLs"][0] + s["RL"][0] == pytest.approx(RT, rel=2e-12)


@pytest.mark.parametrize(
    "params",
    [
        dict(RT=0.05, LsT=0.005, Kds=0.02, Kd=1.6, Kd3=0.5),
        dict(RT=0.05, LsT=0.005, Kds=0.02, Kd=1.6, Kd3=0.005),  # cooperative
        dict(RT=0.05, LsT=0.005, Kds=0.02, Kd=1.6, Kd3=50.0),  # anti-cooperative
        dict(
            RT=1.1230881504118417,
            LsT=0.16772085032232287,
            Kds=0.0959937419194935,
            Kd=0.013895014894051528,
            Kd3=0.5947156861737775,
        ),  # tight competition
    ],
)
def test_four_state_matches_simultaneous_mass_balance_solution(params):
    LT = np.logspace(-3, 2, 12)
    computed = bc.get_model("comp_4st_specific").evaluate(LT, ymin=0, ymax=1, **params)
    expected = [four_state_reference(LT=lt, **params) for lt in LT]
    np.testing.assert_allclose(computed, expected, rtol=1e-10, atol=1e-13)


def test_four_state_limits():
    params = dict(RT=0.7, LsT=0.2, Kds=0.3, Kd=1.9)
    model = bc.get_model("comp_4st_specific")
    direct = bc.get_model("dir_specific")
    no_competitor = model.evaluate([0.0], ymin=0, ymax=1, Kd3=2.1, **params)
    assert no_competitor == pytest.approx(
        direct.evaluate([0.7], ymin=0, ymax=1, LsT=0.2, Kds=0.3), rel=2e-10
    )
    # Saturating competitor: tracer binds RL with Kd3 (Roehrl et al. eq 28).
    saturated = model.evaluate([1e12], ymin=0, ymax=1, Kd3=2.1, **params)
    assert saturated == pytest.approx(
        direct.evaluate([0.7], ymin=0, ymax=1, LsT=0.2, Kds=2.1), rel=2e-10
    )
    # Kd3 = Kds: competitor occupancy does not change tracer binding.
    LT = np.concatenate(([0.0], np.logspace(-8, 8, 80)))
    flat = model.evaluate(LT, ymin=0, ymax=1, Kd3=0.3, **params)
    np.testing.assert_allclose(flat, flat[0], rtol=3e-9)


def test_four_state_total_obeys_equilibria_and_mass_balances():
    N, Kds, Kd, Kd3 = 0.6, 0.3, 1.9, 2.1
    LT = np.logspace(-7, 5, 80)
    s = bc.get_model("comp_4st_total").species(
        LT, ymin=0, ymax=1, RT=0.7, LsT=0.2, Kds=Kds, Kd=Kd, Kd3=Kd3, N=N
    )
    assert_relative(s["R"] + s["RLs"] + s["RL"] + s["RLLs"], 0.7, rtol=2e-8)
    assert_relative(s["Ls"] + s["RLs"] + s["RLLs"], 0.2, rtol=2e-8)
    assert_relative(s["L"] + s["L_nonspecific"] + s["RL"] + s["RLLs"], LT, rtol=2e-8)
    assert_relative(s["L_nonspecific"], N * s["L"])
    assert_relative(Kds * s["RLs"], s["R"] * s["Ls"], rtol=2e-8)
    assert_relative(Kd * s["RL"], s["R"] * s["L"], rtol=2e-8)
    assert_relative(Kd3 * s["RLLs"], s["RL"] * s["Ls"], rtol=2e-8)
    assert_relative(s["Fbs"], (s["RLs"] + s["RLLs"]) / 0.2)


def test_four_state_keeps_tiny_free_receptor_in_tracer_excess():
    s = bc.get_model("comp_4st_specific").species(
        [1.0], ymin=0, ymax=1, RT=1.0, LsT=1e15, Kds=1.0, Kd=1.0, Kd3=1.0
    )
    assert s["R"][0] > 0.0
    assert s["R"][0] + s["RLs"][0] + s["RL"][0] + s["RLLs"][0] == pytest.approx(
        1.0, rel=1e-12
    )


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("ic50", dict(IC50=2.5, hill_slope=1.3)),
        ("dir_specific", dict(LsT=0.35, Kds=1.8)),
        ("dir_total", dict(LsT=0.4, Ns=0.25, Kds=2.2)),
        ("comp_3st_total", dict(RT=0.7, LsT=0.2, Kds=0.3, Kd=1.9, N=0.6)),
        ("comp_4st_specific", dict(RT=0.05, LsT=0.005, Kds=0.02, Kd=1.6, Kd3=0.5)),
    ],
)
def test_models_are_invariant_to_concentration_units(name, params):
    model = bc.get_model(name)
    x = np.logspace(-4, 3, 40)
    scale = 1e9
    scaled = {
        key: value * scale if model.parameter(key).concentration else value
        for key, value in params.items()
    }
    reference = model.evaluate(x, ymin=0, ymax=1, **params)
    np.testing.assert_allclose(
        model.evaluate(x * scale, ymin=0, ymax=1, **scaled),
        reference,
        rtol=2e-10,
        atol=1e-14,
    )
