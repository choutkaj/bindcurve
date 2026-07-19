from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

import bindcurve as bc


@pytest.mark.parametrize("hill_slope", [0.5, 2.0])
def test_ic50_is_midpoint_and_strictly_decreasing(hill_slope):
    model = bc.get_model("ic50")
    IC50 = 3.7
    concentration = IC50 * np.array([1.0e-3, 1.0, 1.0e3])

    response = model.evaluate(
        concentration,
        ymin=2.0,
        amplitude=8.0,
        IC50=IC50,
        hill_slope=hill_slope,
    )

    assert response[1] == pytest.approx(6.0)
    assert np.all(np.diff(response) < 0.0)


def test_ic50_model_is_invariant_to_concentration_units():
    model = bc.get_model("ic50")
    concentration = np.logspace(-3, 3, 31)
    reference = model.evaluate(
        concentration,
        ymin=1.0,
        amplitude=8.0,
        IC50=2.5,
        hill_slope=1.3,
    )

    scale = 1.0e9
    scaled = model.evaluate(
        concentration * scale,
        ymin=1.0,
        amplitude=8.0,
        IC50=2.5 * scale,
        hill_slope=1.3,
    )

    assert np.allclose(scaled, reference, rtol=1.0e-14, atol=1.0e-14)


def test_dir_simple_has_the_one_site_binding_midpoint_and_limits():
    model = bc.get_model("dir_simple")
    response = model.evaluate(
        np.array([0.0, 2.5, 2.5e12]),
        ymin=0.0,
        ymax=1.0,
        Kds=2.5,
    )

    assert response[0] == 0.0
    assert response[1] == pytest.approx(0.5)
    assert response[2] == pytest.approx(1.0, rel=1.0e-11)


def test_dir_specific_matches_roehrl_equation_6():
    RT = np.logspace(-4, 3, 80)
    LsT = 0.35
    Kds = 1.8
    total = Kds + LsT + RT
    discriminant = np.sqrt(total**2 - 4.0 * LsT * RT)
    # Rationalized form of Roehrl et al. eq 6 avoids subtractive cancellation.
    expected_Fbs = 2.0 * RT / (total + discriminant)

    observed = bc.get_model("dir_specific").evaluate(
        RT,
        ymin=0.0,
        ymax=1.0,
        LsT=LsT,
        Kds=Kds,
    )

    assert np.allclose(observed, expected_Fbs, rtol=2.0e-13)


@pytest.mark.parametrize(
    ("model_name", "parameters"),
    [
        ("dir_specific", {"LsT": 0.35, "Kds": 1.8}),
        ("dir_total", {"LsT": 0.4, "Ns": 0.25, "Kds": 2.2}),
    ],
)
def test_depletion_aware_direct_models_are_unit_invariant(model_name, parameters):
    model = bc.get_model(model_name)
    RT = np.logspace(-4, 3, 60)
    reference = model.evaluate(
        RT,
        ymin=0.0,
        ymax=1.0,
        **parameters,
    )

    scale = 1.0e12
    scaled_parameters = {
        name: value if name == "Ns" else value * scale
        for name, value in parameters.items()
    }
    scaled = model.evaluate(
        RT * scale,
        ymin=0.0,
        ymax=1.0,
        **scaled_parameters,
    )

    assert np.allclose(scaled, reference, rtol=1.0e-12, atol=1.0e-14)


def test_dir_specific_preserves_tiny_free_receptor_in_tracer_excess():
    components = bc.get_model("dir_specific").evaluate_components(
        np.array([1.0]),
        ymin=0.0,
        ymax=1.0,
        LsT=1.0e15,
        Kds=1.0,
    ).components

    assert components["R"][0] > 0.0
    assert components["R"][0] == pytest.approx(1.0e-15, rel=1.0e-12)
    assert components["R"][0] + components["RLs"][0] == pytest.approx(
        1.0, rel=1.0e-12
    )


def test_three_state_specific_obeys_independent_equilibria_and_mass_balances():
    components = bc.get_model("comp_3st_specific").evaluate_components(
        np.logspace(-7, 5, 100),
        ymin=0.0,
        ymax=1.0,
        RT=0.7,
        LsT=0.2,
        Kds=0.3,
        Kd=1.9,
    ).components

    assert np.allclose(
        components["RT"],
        components["R"] + components["RLs"] + components["RL"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        components["LsT"],
        components["Ls"] + components["RLs"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        components["LT"],
        components["L"] + components["RL"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        0.3 * components["RLs"],
        components["R"] * components["Ls"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        1.9 * components["RL"],
        components["R"] * components["L"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )


def test_three_state_total_obeys_roehrl_nonspecific_mass_balance():
    N = 0.6
    components = bc.get_model("comp_3st_total").evaluate_components(
        np.logspace(-7, 5, 100),
        ymin=0.0,
        ymax=1.0,
        RT=0.7,
        LsT=0.2,
        Kds=0.3,
        Kd=1.9,
        N=N,
    ).components

    assert np.allclose(
        components["RT"],
        components["R"] + components["RLs"] + components["RL"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        components["LT"],
        components["L"]
        + components["RL"]
        + components["L_nonspecific_bound"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        components["L_nonspecific_bound"],
        N * components["L"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        components["L_bound_total"],
        components["RL"] + components["L_nonspecific_bound"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        1.9 * components["RL"],
        components["R"] * components["L"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )


def test_three_state_solver_is_stable_for_extreme_concentration_ratios():
    RT = 2.172723469932662e-6
    components = bc.get_model("comp_3st_specific").evaluate_components(
        np.array([106139.16205049057]),
        ymin=0.0,
        ymax=1.0,
        RT=RT,
        LsT=686450.9185295302,
        Kds=0.00012087951717196983,
        Kd=0.00001652150832486005,
    ).components

    receptor_sum = (
        components["R"][0]
        + components["RLs"][0]
        + components["RL"][0]
    )
    assert 0.0 < components["R"][0] <= RT
    assert receptor_sum == pytest.approx(RT, rel=2.0e-12)


@pytest.mark.parametrize("model_name", ["comp_3st_specific", "comp_3st_total"])
def test_three_state_models_are_unit_invariant(model_name):
    model = bc.get_model(model_name)
    concentration = np.logspace(-4, 3, 50)
    parameters = {"RT": 0.7, "LsT": 0.2, "Kds": 0.3, "Kd": 1.9}
    if model_name.endswith("total"):
        parameters["N"] = 0.6
    reference = model.evaluate(
        concentration,
        ymin=0.0,
        ymax=1.0,
        **parameters,
    )

    scale = 1.0e9
    scaled_parameters = {
        name: value if name == "N" else value * scale
        for name, value in parameters.items()
    }
    scaled = model.evaluate(
        concentration * scale,
        ymin=0.0,
        ymax=1.0,
        **scaled_parameters,
    )

    assert np.allclose(scaled, reference, rtol=2.0e-12, atol=1.0e-14)


def test_four_state_reduces_to_direct_binding_without_competitor():
    parameters = {"RT": 0.7, "LsT": 0.2, "Kds": 0.3, "Kd": 1.9, "Kd3": 2.1}
    four_state = bc.get_model("comp_4st_specific").evaluate(
        np.array([0.0]),
        ymin=0.0,
        ymax=1.0,
        **parameters,
    )[0]
    direct = bc.get_model("dir_specific").evaluate(
        np.array([parameters["RT"]]),
        ymin=0.0,
        ymax=1.0,
        LsT=parameters["LsT"],
        Kds=parameters["Kds"],
    )[0]

    assert four_state == pytest.approx(direct, rel=2.0e-10, abs=1.0e-12)


def test_four_state_high_competitor_limit_matches_roehrl_equation_28():
    parameters = {"RT": 0.7, "LsT": 0.2, "Kds": 0.3, "Kd": 1.9, "Kd3": 2.1}
    high_competitor = bc.get_model("comp_4st_specific").evaluate(
        np.array([1.0e12]),
        ymin=0.0,
        ymax=1.0,
        **parameters,
    )[0]
    asymptotic_direct = bc.get_model("dir_specific").evaluate(
        np.array([parameters["RT"]]),
        ymin=0.0,
        ymax=1.0,
        LsT=parameters["LsT"],
        Kds=parameters["Kd3"],
    )[0]

    assert high_competitor == pytest.approx(
        asymptotic_direct, rel=2.0e-10, abs=1.0e-12
    )


def test_four_state_is_competitor_independent_when_kd3_equals_kds():
    concentration = np.concatenate(([0.0], np.logspace(-8, 8, 80)))
    response = bc.get_model("comp_4st_specific").evaluate(
        concentration,
        ymin=0.0,
        ymax=1.0,
        RT=0.7,
        LsT=0.2,
        Kds=0.3,
        Kd=1.9,
        Kd3=0.3,
    )

    assert np.allclose(response, response[0], rtol=3.0e-9, atol=1.0e-11)


def test_four_state_total_obeys_all_equilibria_and_mass_balances():
    N = 0.6
    Kds = 0.3
    Kd = 1.9
    Kd3 = 2.1
    components = bc.get_model("comp_4st_total").evaluate_components(
        np.logspace(-7, 5, 80),
        ymin=0.0,
        ymax=1.0,
        RT=0.7,
        LsT=0.2,
        Kds=Kds,
        Kd=Kd,
        Kd3=Kd3,
        N=N,
    ).components

    assert np.allclose(
        components["RT"],
        components["R"]
        + components["RLs"]
        + components["RL"]
        + components["RLLs"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        components["LsT"],
        components["Ls"] + components["RLs"] + components["RLLs"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        components["LT"],
        components["L"]
        + components["RL"]
        + components["RLLs"]
        + components["L_nonspecific_bound"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        components["L_nonspecific_bound"],
        N * components["L"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        Kds * components["RLs"],
        components["R"] * components["Ls"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        Kd * components["RL"],
        components["R"] * components["L"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        Kd3 * components["RLLs"],
        components["RL"] * components["Ls"],
        rtol=2.0e-8,
        atol=1.0e-11,
    )
    assert np.allclose(
        components["L_bound_specific"],
        components["RL"] + components["RLLs"],
    )
    assert np.allclose(
        components["RLs_plus_RLLs"],
        components["RLs"] + components["RLLs"],
    )
    assert np.allclose(
        components["Fbs"],
        components["RLs_plus_RLLs"] / components["LsT"],
    )
    assert np.allclose(
        components["L_bound_total"],
        components["RL"]
        + components["RLLs"]
        + components["L_nonspecific_bound"],
    )


def _forward_competitive_ic50(*, RT, LsT, Kds, Kd):
    R0 = brentq(
        lambda free: free + LsT * free / (Kds + free) - RT,
        0.0,
        RT,
    )
    Ls0 = LsT / (1.0 + R0 / Kds)
    RLs0 = R0 * Ls0 / Kds
    RLs50 = RLs0 / 2.0
    Ls50 = LsT - RLs50
    R50 = Kds * RLs50 / Ls50
    RL50 = RT - R50 - RLs50
    L50 = Kd * RL50 / R50
    return L50 + RL50, RLs0 / Ls0


@pytest.mark.parametrize(
    ("RT", "LsT", "Kds", "expected_kd"),
    [
        (0.05, 0.005, 0.02, 1.6),
        (0.5, 2.0, 4.0, 3.0),
        (30.0, 5.0, 17.92, 430.0),
        (8.0, 25.0, 0.7, 0.03),
    ],
)
def test_exact_ic50_conversions_recover_independent_equilibrium(
    RT,
    LsT,
    Kds,
    expected_kd,
):
    IC50, y0 = _forward_competitive_ic50(
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=expected_kd,
    )

    coleska = bc.coleska(IC50=IC50, RT=RT, LsT=LsT, Kds=Kds)
    munson_rodbard = bc.cheng_prusoff_corrected(
        IC50=IC50,
        LsT=LsT,
        Kds=Kds,
        y0=y0,
    )

    assert coleska == pytest.approx(expected_kd, rel=2.0e-12)
    assert munson_rodbard == pytest.approx(expected_kd, rel=2.0e-12)


def test_cheng_prusoff_is_low_receptor_limit_of_exact_conversion():
    RT = 1.0e-9
    LsT = 2.0
    Kds = 4.0
    expected_kd = 3.0
    IC50, _ = _forward_competitive_ic50(
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=expected_kd,
    )

    approximate = bc.cheng_prusoff(IC50=IC50, LsT=LsT, Kds=Kds)

    assert approximate == pytest.approx(expected_kd, rel=2.0e-9)


def test_all_ic50_conversions_are_unit_invariant():
    RT = 0.5
    LsT = 2.0
    Kds = 4.0
    expected_kd = 3.0
    IC50, y0 = _forward_competitive_ic50(
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=expected_kd,
    )
    scale = 1.0e9

    assert bc.cheng_prusoff(
        IC50=IC50 * scale,
        LsT=LsT * scale,
        Kds=Kds * scale,
    ) == pytest.approx(
        bc.cheng_prusoff(IC50=IC50, LsT=LsT, Kds=Kds) * scale,
        rel=2.0e-15,
    )
    assert bc.cheng_prusoff_corrected(
        IC50=IC50 * scale,
        LsT=LsT * scale,
        Kds=Kds * scale,
        y0=y0,
    ) == pytest.approx(expected_kd * scale, rel=2.0e-14)
    assert bc.coleska(
        IC50=IC50 * scale,
        RT=RT * scale,
        LsT=LsT * scale,
        Kds=Kds * scale,
    ) == pytest.approx(expected_kd * scale, rel=2.0e-14)


def test_munson_rodbard_rejects_physically_impossible_ic50():
    with pytest.raises(ValueError, match="non-positive"):
        bc.cheng_prusoff_corrected(IC50=0.01, LsT=2.0, Kds=4.0, y0=0.5)
