import numpy as np
import pytest
from references import competitive_ic50

import bindcurve as bc


@pytest.mark.parametrize(
    ("RT", "LsT", "Kds", "Kd"),
    [
        (0.05, 0.005, 0.02, 1.6),
        (0.5, 2.0, 4.0, 3.0),
        (30.0, 5.0, 17.92, 430.0),
        (8.0, 25.0, 0.7, 0.03),
    ],
)
def test_exact_conversions_recover_the_equilibrium_kd(RT, LsT, Kds, Kd):
    IC50, y0 = competitive_ic50(RT=RT, LsT=LsT, Kds=Kds, Kd=Kd)
    assert bc.coleska(IC50, RT=RT, LsT=LsT, Kds=Kds) == pytest.approx(Kd, rel=2e-12)
    assert bc.munson_rodbard(IC50, LsT=LsT, Kds=Kds, y0=y0) == pytest.approx(
        Kd, rel=2e-12
    )


def test_cheng_prusoff_is_the_low_receptor_limit():
    IC50, _ = competitive_ic50(RT=1e-9, LsT=2.0, Kds=4.0, Kd=3.0)
    assert bc.cheng_prusoff(IC50, LsT=2.0, Kds=4.0) == pytest.approx(3.0, rel=2e-9)


def test_munson_rodbard_matches_the_erratum_example():
    assert bc.munson_rodbard(1.0, LsT=0.1, Kds=1.0, y0=0.1) == pytest.approx(
        0.7889, abs=5e-5
    )


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
    # 5 nM tracer with Kd = 17.92 nM; reported Ki values are rounded to 10 nM.
    assert bc.coleska(IC50, RT=RT, LsT=5.0, Kds=17.92) == pytest.approx(
        reported_kd, abs=9.0
    )


def test_conversions_are_unit_invariant():
    IC50, y0 = competitive_ic50(RT=0.5, LsT=2.0, Kds=4.0, Kd=3.0)
    s = 1e9
    assert bc.cheng_prusoff(IC50 * s, LsT=2 * s, Kds=4 * s) == pytest.approx(
        bc.cheng_prusoff(IC50, LsT=2.0, Kds=4.0) * s, rel=2e-15
    )
    assert bc.munson_rodbard(IC50 * s, LsT=2 * s, Kds=4 * s, y0=y0) == pytest.approx(
        3 * s, rel=2e-14
    )
    assert bc.coleska(IC50 * s, RT=0.5 * s, LsT=2 * s, Kds=4 * s) == pytest.approx(
        3 * s, rel=2e-14
    )


def test_conversions_accept_arrays_and_mark_impossible_values():
    # Exact conversions need IC50 above the receptor-bound competitor at 50%.
    converted = bc.coleska(np.array([0.01, 2.0, np.nan]), RT=0.5, LsT=2.0, Kds=4.0)
    assert np.isnan(converted[0]) and converted[1] > 0.0 and np.isnan(converted[2])
    assert np.isnan(bc.munson_rodbard(0.01, LsT=2.0, Kds=4.0, y0=0.5))
    with pytest.raises(ValueError, match="LsT must be positive"):
        bc.cheng_prusoff(1.0, LsT=0.0, Kds=1.0)
