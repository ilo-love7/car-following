import math

import numpy as np
import pytest

from carfollowing.models import (IDM, OVM, FVD, Helly, GHR, Gipps, Newell,
                         MODEL_REGISTRY, make_model, DEFAULT_PARAMS)


ALL = [IDM(), OVM(), FVD(), Helly(), GHR(), Gipps(), Newell()]


def test_registry_complete():
    assert set(MODEL_REGISTRY) == {"IDM", "OVM", "FVD", "Helly", "GHR",
                                   "Gipps", "Newell"}
    for name in MODEL_REGISTRY:
        m = make_model(name)
        assert m.name == name
        assert m.get_params() == {k: float(v) for k, v in DEFAULT_PARAMS[name].items()}


def test_with_params_does_not_mutate():
    a = IDM()
    b = a.with_params(v0=42.0)
    assert a.v0 == 30.0 and b.v0 == 42.0


def test_idm_equilibrium_gap_formula():
    m = IDM(v0=30.0, T=1.2, s0=2.0, delta=4.0)
    for v in (5.0, 12.0, 20.0, 27.0):
        s = m.equilibrium_gap(v)
        expected = (m.s0 + v * m.T) / math.sqrt(1.0 - (v / m.v0) ** 4)
        assert s == pytest.approx(expected, rel=1e-12)
        # acceleration must vanish at the equilibrium gap
        assert m.accel(s, v, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_idm_no_equilibrium_above_free_speed():
    m = IDM(v0=30.0)
    assert math.isinf(m.equilibrium_gap(30.0))
    assert math.isinf(m.equilibrium_gap(35.0))


def test_equilibrium_roundtrip():
    for m in [IDM(), Helly(), Newell()]:
        for v in (5.0, 15.0, 25.0):
            if not np.isfinite(m.equilibrium_gap(v)):
                continue
            s = m.equilibrium_gap(v)
            assert m.equilibrium_speed(s) == pytest.approx(v, rel=1e-3, abs=1e-3)


def test_sign_conventions_at_equilibrium():
    """f_s > 0, f_v < 0 and f_dv <= 0 are required by the stability theory."""
    h = 1e-4
    for m in ALL:
        if m.name == "GHR":
            continue  # GHR has no unique equilibrium
        v = 12.0
        s = m.equilibrium_gap(v)
        if not np.isfinite(s):
            continue
        f = m.accel
        fs = (f(s + h, v, 0.0) - f(s - h, v, 0.0)) / (2 * h)
        fv = (f(s, v + h, 0.0) - f(s, v - h, 0.0)) / (2 * h)
        fdv = (f(s, v, h) - f(s, v, -h)) / (2 * h)
        assert fs > 0, m.name
        assert fv < 0, m.name
        assert fdv <= 1e-9, m.name


def test_free_speeds():
    assert IDM(v0=30.0).free_speed() == 30.0
    assert Newell(v_free=28.0).free_speed() == 28.0
    assert Gipps(v_des=25.0).free_speed() == 25.0
    assert OVM(v_max=33.0).free_speed() == pytest.approx(0.5 * 33.0 * (1 + math.tanh(2.0)))
    assert Helly().free_speed() is None


def test_ghr_degenerate_equilibrium():
    m = GHR()
    # zero closing speed -> zero acceleration regardless of gap
    assert m.accel(10.0, 20.0, 0.0) == 0.0
    assert m.accel(50.0, 20.0, 0.0) == 0.0
    # closing in (dv>0) -> deceleration
    assert m.accel(10.0, 20.0, 2.0) < 0.0
