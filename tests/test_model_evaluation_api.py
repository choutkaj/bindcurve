from __future__ import annotations

import numpy as np
import pytest

import bindcurve as bc


@pytest.mark.parametrize(
    ("model_name", "params", "expected_components"),
    [
        ("dir_simple", {"Kds": 0.2}, {"R", "Fbs"}),
        (
            "dir_specific",
            {"LsT": 0.1, "Kds": 0.2},
            {"RT", "R", "LsT", "Ls", "RLs", "Fbs"},
        ),
        (
            "dir_total",
            {"LsT": 0.1, "Ns": 0.3, "Kds": 0.2},
            {
                "RT",
                "R",
                "LsT",
                "Ls",
                "RLs",
                "Ls_nonspecific_bound",
                "Ls_bound_total",
                "Fbs_specific",
                "Fbs_total",
                "Fbs",
            },
        ),
        (
            "comp_3st_specific",
            {"RT": 0.2, "LsT": 0.1, "Kds": 0.3, "Kd": 0.4},
            {"LT", "RT", "R", "LsT", "Ls", "L", "RLs", "RL", "Fbs"},
        ),
        (
            "comp_3st_total",
            {"RT": 0.2, "LsT": 0.1, "Kds": 0.3, "Kd": 0.4, "N": 0.5},
            {
                "LT",
                "RT",
                "R",
                "LsT",
                "Ls",
                "L",
                "RLs",
                "RL",
                "L_nonspecific_bound",
                "L_bound_total",
                "Fbs",
            },
        ),
        (
            "comp_4st_specific",
            {"RT": 0.2, "LsT": 0.1, "Kds": 0.3, "Kd": 0.4, "Kd3": 0.6},
            {
                "LT",
                "RT",
                "R",
                "LsT",
                "Ls",
                "L",
                "RLs",
                "RL",
                "RLLs",
                "Fbs",
            },
        ),
        (
            "comp_4st_total",
            {
                "RT": 0.2,
                "LsT": 0.1,
                "Kds": 0.3,
                "Kd": 0.4,
                "Kd3": 0.6,
                "N": 0.5,
            },
            {
                "LT",
                "RT",
                "R",
                "LsT",
                "Ls",
                "L",
                "RLs",
                "RL",
                "RLLs",
                "RLs_plus_RLLs",
                "L_bound_specific",
                "L_nonspecific_bound",
                "L_bound_total",
                "Fbs",
            },
        ),
    ],
)
def test_binding_component_schemas_use_only_canonical_symbols(
    model_name,
    params,
    expected_components,
):
    evaluation = bc.get_model(model_name).evaluate_components(
        np.array([0.4]),
        ymin=0.0,
        ymax=1.0,
        **params,
    )

    assert set(evaluation.components) == expected_components


def test_ic50_evaluate_components_uses_raw_concentration_axis():
    concentration = np.logspace(-2, 2, 11)
    model = bc.get_model("ic50")

    evaluation = model.evaluate_components(
        concentration,
        ymin=0.0,
        ymax=100.0,
        IC50=10**0.25,
        hill_slope=1.15,
    )
    direct = model.evaluate(
        concentration,
        ymin=0.0,
        ymax=100.0,
        IC50=10**0.25,
        hill_slope=1.15,
    )

    assert isinstance(evaluation, bc.ModelEvaluation)
    assert np.allclose(evaluation.concentration, concentration)
    assert np.allclose(evaluation.response, direct)
    assert "fraction_response" in evaluation.components


def test_direct_specific_evaluate_components_returns_consistent_mass_balances():
    concentration = np.logspace(-4, 1, 50)
    model = bc.get_model("dir_specific")

    evaluation = model.evaluate_components(
        concentration,
        ymin=0.0,
        ymax=1.0,
        LsT=0.35,
        Kds=1.8,
    )
    components = evaluation.components

    assert np.allclose(components["RT"], concentration)
    assert np.allclose(
        components["RT"],
        components["R"] + components["RLs"],
    )
    assert np.allclose(
        components["LsT"],
        components["Ls"] + components["RLs"],
    )
    assert np.allclose(evaluation.response, components["Fbs"])


def test_direct_total_evaluate_components_tracks_specific_and_nonspecific_tracer():
    concentration = np.concatenate(([0.0], np.logspace(-4, 1, 50)))
    model = bc.get_model("dir_total")

    evaluation = model.evaluate_components(
        concentration,
        ymin=0.0,
        ymax=1.0,
        LsT=0.4,
        Ns=0.25,
        Kds=2.2,
    )
    components = evaluation.components

    assert np.allclose(components["RT"], concentration)
    assert np.allclose(
        components["RT"],
        components["R"] + components["RLs"],
    )
    assert np.allclose(
        components["Ls_bound_total"],
        components["RLs"] + components["Ls_nonspecific_bound"],
    )
    assert np.allclose(
        components["Ls_nonspecific_bound"],
        0.25 * components["Ls"],
    )
    assert np.allclose(
        2.2 * components["RLs"],
        components["R"] * components["Ls"],
    )
    assert np.allclose(
        components["LsT"],
        components["Ls"] + components["Ls_bound_total"],
    )
    assert np.allclose(
        components["Fbs_specific"],
        components["RLs"] / components["LsT"],
    )
    assert np.allclose(
        components["Fbs_total"],
        components["Ls_bound_total"] / components["LsT"],
    )
    assert components["Fbs_total"][0] == pytest.approx(0.25 / 1.25)
    assert components["Fbs"][0] == pytest.approx(0.0)
    assert np.allclose(evaluation.response, components["Fbs"])


def test_direct_total_midpoint_uses_roehrl_apparent_kd_shift():
    LsT = 0.4
    Ns = 0.25
    Kds = 2.2
    effective_kd = (1.0 + Ns) * Kds
    receptor_total_at_midpoint = effective_kd + LsT / 2.0

    evaluation = bc.get_model("dir_total").evaluate_components(
        np.array([receptor_total_at_midpoint]),
        ymin=0.0,
        ymax=1.0,
        LsT=LsT,
        Ns=Ns,
        Kds=Kds,
    )

    assert evaluation.components["R"][0] == pytest.approx(effective_kd)
    assert evaluation.response[0] == pytest.approx(0.5)


def test_comp_3st_specific_evaluate_components_returns_species_balances():
    concentration = np.logspace(-4, 2, 60)
    model = bc.get_model("comp_3st_specific")

    evaluation = model.evaluate_components(
        concentration,
        ymin=0.0,
        ymax=1.0,
        RT=0.05,
        LsT=0.005,
        Kds=0.02,
        Kd=1.6,
    )
    components = evaluation.components

    assert np.allclose(
        components["RT"],
        components["R"] + components["RLs"] + components["RL"],
        atol=1.0e-10,
    )
    assert np.allclose(
        components["LsT"],
        components["Ls"] + components["RLs"],
        atol=1.0e-10,
    )
    assert np.allclose(
        components["LT"],
        components["L"] + components["RL"],
        atol=1.0e-10,
    )
    assert np.allclose(evaluation.response, components["Fbs"])


def test_comp_4st_specific_evaluate_components_returns_species_balances():
    concentration = np.logspace(-4, 2, 60)
    model = bc.get_model("comp_4st_specific")

    evaluation = model.evaluate_components(
        concentration,
        ymin=0.0,
        ymax=1.0,
        RT=0.05,
        LsT=0.005,
        Kds=0.02,
        Kd=1.6,
        Kd3=0.5,
    )
    components = evaluation.components

    assert np.allclose(
        components["RT"],
        components["R"]
        + components["RLs"]
        + components["RL"]
        + components["RLLs"],
        atol=1.0e-9,
    )
    assert np.allclose(
        components["LsT"],
        components["Ls"] + components["RLs"] + components["RLLs"],
        atol=1.0e-9,
    )
    assert np.allclose(
        components["LT"],
        components["L"] + components["RL"] + components["RLLs"],
        atol=1.0e-9,
    )
    assert np.allclose(
        components["Fbs"],
        (components["RLs"] + components["RLLs"]) / components["LsT"],
    )
    assert np.allclose(evaluation.response, components["Fbs"])
