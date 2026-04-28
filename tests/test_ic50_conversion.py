from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


def legacy_cheng_prusoff(LsT, Kds, IC50):
    return IC50 / (1.0 + LsT / Kds)


def legacy_cheng_prusoff_corr(LsT, Kds, y0, IC50):
    return IC50 / (1.0 + (LsT * (y0 + 2.0) / (2.0 * Kds * (y0 + 1.0)) + y0)) + Kds * (
        y0 / (y0 + 2.0)
    )


def legacy_coleska(RT, LsT, Kds, IC50):
    a = LsT + Kds - RT
    b = -Kds * RT
    R0 = (-a + np.sqrt(a**2 - 4.0 * b)) / 2.0
    Ls0 = LsT / (1.0 + R0 / Kds)
    RLs0 = RT / (1.0 + Kds / Ls0)
    RLs50 = RLs0 / 2.0
    Ls50 = LsT - RLs50
    RL50 = RT + Kds * (RLs50 / Ls50) + RLs50
    L50 = IC50 - RL50
    return L50 / ((Ls50 / Kds) + (R0 / Kds) + 1.0)


def test_cheng_prusoff_matches_legacy_formula():
    result = bc.cheng_prusoff(IC50=10.0, LsT=2.0, Kds=4.0)

    assert result == pytest.approx(legacy_cheng_prusoff(2.0, 4.0, 10.0))


def test_cheng_prusoff_corrected_matches_legacy_formula():
    result = bc.cheng_prusoff_corrected(IC50=10.0, LsT=2.0, Kds=4.0, y0=0.25)

    assert result == pytest.approx(legacy_cheng_prusoff_corr(2.0, 4.0, 0.25, 10.0))


def test_coleska_matches_legacy_formula():
    result = bc.coleska(IC50=10.0, RT=0.5, LsT=2.0, Kds=4.0)

    assert result == pytest.approx(legacy_coleska(0.5, 2.0, 4.0, 10.0))


def test_scalar_conversion_returns_result_container():
    result = bc.convert_ic50_to_kd(
        model="cheng_prusoff",
        IC50=10.0,
        LsT=2.0,
        Kds=4.0,
        unit="uM",
    )

    assert isinstance(result, bc.IC50ConversionResult)
    assert result.compound_id is None
    assert result.Kd == pytest.approx(legacy_cheng_prusoff(2.0, 4.0, 10.0))
    assert result.unit == "uM"


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
        unit="uM",
    )

    assert list(converted["compound_id"]) == ["cmpd_a", "cmpd_b"]
    assert list(converted["model"]) == ["cheng_prusoff", "cheng_prusoff"]
    assert np.allclose(converted["Kd"], [legacy_cheng_prusoff(2.0, 4.0, 10.0), legacy_cheng_prusoff(2.0, 4.0, 20.0)])
    assert np.allclose(converted["lower_Kd"], [legacy_cheng_prusoff(2.0, 4.0, 8.0), legacy_cheng_prusoff(2.0, 4.0, 18.0)])
    assert np.allclose(converted["upper_Kd"], [legacy_cheng_prusoff(2.0, 4.0, 12.0), legacy_cheng_prusoff(2.0, 4.0, 22.0)])
    assert set(converted["unit"]) == {"uM"}


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

    assert converted.loc[0, "lower_Kd"] == pytest.approx(legacy_cheng_prusoff(2.0, 4.0, 8.0))
    assert converted.loc[0, "upper_Kd"] == pytest.approx(legacy_cheng_prusoff(2.0, 4.0, 12.0))


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
