from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def legacy_cheng_prusoff(LsT, Kds, IC50):
    return IC50 / (1.0 + LsT / Kds)


def munson_rodbard(LsT, Kds, y0, IC50):
    denominator = 1.0 + (LsT * (y0 + 2.0) / (2.0 * Kds * (y0 + 1.0)) + y0)
    return IC50 / denominator - Kds * (y0 / (y0 + 2.0))


def coleska_forward_ic50(RT, LsT, Kds, Kd):
    a = LsT + Kds - RT
    b = -Kds * RT
    R0 = (-a + np.sqrt(a**2 - 4.0 * b)) / 2.0
    Ls0 = LsT / (1.0 + R0 / Kds)
    RLs0 = RT / (1.0 + Kds / Ls0)
    RLs50 = RLs0 / 2.0
    Ls50 = LsT - RLs50
    R50 = Kds * RLs50 / Ls50
    RL50 = RT - R50 - RLs50
    L50 = Kd * RL50 / R50
    return L50 + RL50


def test_cheng_prusoff_matches_legacy_formula():
    result = bc.cheng_prusoff(IC50=10.0, LsT=2.0, Kds=4.0)

    assert result == pytest.approx(legacy_cheng_prusoff(2.0, 4.0, 10.0))


def test_cheng_prusoff_corrected_matches_munson_rodbard_erratum():
    result = bc.cheng_prusoff_corrected(IC50=10.0, LsT=2.0, Kds=4.0, y0=0.25)

    assert result == pytest.approx(munson_rodbard(2.0, 4.0, 0.25, 10.0))


def test_cheng_prusoff_corrected_matches_erratum_numerical_example():
    result = bc.cheng_prusoff_corrected(IC50=1.0, LsT=0.1, Kds=1.0, y0=0.1)

    assert result == pytest.approx(0.7889, abs=5.0e-5)


def test_coleska_recovers_kd_from_competitive_equilibrium():
    expected_kd = 3.0
    IC50 = coleska_forward_ic50(RT=0.5, LsT=2.0, Kds=4.0, Kd=expected_kd)

    result = bc.coleska(IC50=IC50, RT=0.5, LsT=2.0, Kds=4.0)

    assert result == pytest.approx(expected_kd)


@pytest.mark.parametrize(
    ("RT", "IC50", "reported_kd"),
    [
        (30.0, 1160.0, 430.0),
        (60.0, 2520.0, 570.0),
        (120.0, 3100.0, 400.0),
        (240.0, 8100.0, 550.0),
    ],
)
def test_coleska_matches_nikolovska_coleska_table_2(RT, IC50, reported_kd):
    # Table 2 uses 5 nM tracer with Kd = 17.92 nM. Its reported Ki values are
    # rounded to tens of nM (doi:10.1016/j.ab.2004.05.055).
    result = bc.coleska(IC50=IC50, RT=RT, LsT=5.0, Kds=17.92)

    assert result == pytest.approx(reported_kd, abs=9.0)


def test_scalar_conversion_returns_result_container():
    result = bc.convert_ic50_to_kd(
        model="cheng_prusoff",
        IC50=10.0,
        LsT=2.0,
        Kds=4.0,
    )

    assert isinstance(result, bc.IC50ConversionResult)
    assert result.compound_id is None
    assert result.Kd == pytest.approx(legacy_cheng_prusoff(2.0, 4.0, 10.0))


def test_dataframe_conversion_with_confidence_limits():
    df = pd.DataFrame(
        {
            "compound_id": ["cmpd_a", "cmpd_b"],
            "IC50": [10.0, 20.0],
            "lower_IC50": [8.0, 18.0],
            "upper_IC50": [12.0, 22.0],
        }
    )

    converted = bc.convert_ic50_to_kd(
        df,
        model="cheng_prusoff",
        LsT=2.0,
        Kds=4.0,
        lower_col="lower_IC50",
        upper_col="upper_IC50",
    )

    expected_kd = [
        legacy_cheng_prusoff(2.0, 4.0, 10.0),
        legacy_cheng_prusoff(2.0, 4.0, 20.0),
    ]
    expected_lower = [
        legacy_cheng_prusoff(2.0, 4.0, 8.0),
        legacy_cheng_prusoff(2.0, 4.0, 18.0),
    ]
    expected_upper = [
        legacy_cheng_prusoff(2.0, 4.0, 12.0),
        legacy_cheng_prusoff(2.0, 4.0, 22.0),
    ]

    assert list(converted["compound_id"]) == ["cmpd_a", "cmpd_b"]
    assert list(converted["model"]) == ["cheng_prusoff", "cheng_prusoff"]
    assert np.allclose(converted["Kd"], expected_kd)
    assert np.allclose(converted["lower_Kd"], expected_lower)
    assert np.allclose(converted["upper_Kd"], expected_upper)


def test_dataframe_conversion_supports_fit_results_column_names():
    df = pd.DataFrame(
        {
            "compound_id": ["cmpd_a"],
            "IC50": [10.0],
            "IC50_lower_ci": [8.0],
            "IC50_upper_ci": [12.0],
        }
    )

    converted = bc.convert_ic50_to_kd(
        df,
        model="cheng_prusoff",
        LsT=2.0,
        Kds=4.0,
        lower_col="IC50_lower_ci",
        upper_col="IC50_upper_ci",
    )

    assert converted.loc[0, "lower_Kd"] == pytest.approx(
        legacy_cheng_prusoff(2.0, 4.0, 8.0)
    )
    assert converted.loc[0, "upper_Kd"] == pytest.approx(
        legacy_cheng_prusoff(2.0, 4.0, 12.0)
    )


def test_conversion_rejects_missing_required_constants():
    with pytest.raises(ValueError, match="LsT"):
        bc.convert_ic50_to_kd(model="cheng_prusoff", IC50=10.0, Kds=4.0)

    with pytest.raises(ValueError, match="RT"):
        bc.convert_ic50_to_kd(model="coleska", IC50=10.0, LsT=2.0, Kds=4.0)

    with pytest.raises(ValueError, match="y0"):
        bc.convert_ic50_to_kd(
            model="cheng_prusoff_corrected",
            IC50=10.0,
            LsT=2.0,
            Kds=4.0,
        )


def test_conversion_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="IC50"):
        bc.cheng_prusoff(IC50=0.0, LsT=2.0, Kds=4.0)

    with pytest.raises(ValueError, match="Kds"):
        bc.cheng_prusoff(IC50=10.0, LsT=2.0, Kds=-1.0)


def test_coleska_rejects_physically_incompatible_ic50():
    with pytest.raises(ValueError, match="non-positive"):
        bc.coleska(IC50=0.01, RT=0.5, LsT=2.0, Kds=4.0)
