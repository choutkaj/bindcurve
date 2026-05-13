from __future__ import annotations

import numpy as np

import bindcurve as bc


def test_logic50_predict_and_evaluate_components_use_raw_concentration_axis():
    concentration = np.logspace(-2, 2, 11)
    model = bc.get_model("logic50")

    evaluation = model.evaluate_components(
        concentration,
        ymin=0.0,
        ymax=100.0,
        logIC50=0.25,
        hill_slope=-1.15,
    )
    predicted = model.predict(
        concentration,
        ymin=0.0,
        ymax=100.0,
        logIC50=0.25,
        hill_slope=-1.15,
    )
    direct = model.evaluate(
        model.transform_x(concentration),
        ymin=0.0,
        ymax=100.0,
        logIC50=0.25,
        hill_slope=-1.15,
    )

    assert isinstance(evaluation, bc.ModelEvaluation)
    assert np.allclose(evaluation.concentration, concentration)
    assert np.allclose(evaluation.transformed_x, np.log10(concentration))
    assert np.allclose(evaluation.response, predicted)
    assert np.allclose(predicted, direct)
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

    assert np.allclose(components["R_total"], concentration)
    assert np.allclose(
        components["R_total"],
        components["R_free"] + components["RLstar"],
    )
    assert np.allclose(
        components["Lstar_total"],
        components["Lstar_free"] + components["RLstar"],
    )
    assert np.allclose(evaluation.response, components["fraction_bound"])


def test_direct_total_evaluate_components_tracks_specific_and_nonspecific_tracer():
    concentration = np.logspace(-4, 1, 50)
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

    assert np.allclose(components["R_total"], concentration)
    assert np.allclose(
        components["R_total"],
        components["R_free"] + components["RLstar"],
    )
    assert np.allclose(
        components["Lstar_bound_total"],
        components["RLstar"] + components["Lstar_nonspecific_bound"],
    )
    assert np.allclose(
        components["Lstar_total"],
        components["Lstar_free"] + components["Lstar_bound_total"],
    )
    assert np.allclose(evaluation.response, components["fraction_bound"])


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
        components["R_total"],
        components["R_free"] + components["RLstar"] + components["RL"],
        atol=1.0e-10,
    )
    assert np.allclose(
        components["Lstar_total"],
        components["Lstar_free"] + components["RLstar"],
        atol=1.0e-10,
    )
    assert np.allclose(
        components["L_total"],
        components["L_free"] + components["RL"],
        atol=1.0e-10,
    )
    assert np.allclose(evaluation.response, components["fraction_tracer_bound"])


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
        components["R_total"],
        components["R_free"] + components["RS"] + components["RL"] + components["RLS"],
        atol=1.0e-9,
    )
    assert np.allclose(
        components["Lstar_total"],
        components["Lstar_free"] + components["RS"] + components["RLS"],
        atol=1.0e-9,
    )
    assert np.allclose(
        components["L_total"],
        components["L_free"] + components["RL"] + components["RLS"],
        atol=1.0e-9,
    )
    assert np.allclose(evaluation.response, components["fraction_tracer_bound"])

