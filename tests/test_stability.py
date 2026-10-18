import numpy as np
import pytest

from carfollowing.models import IDM, Helly, OVM, FVD
from carfollowing.stability import (partial_derivatives, linear_stability, ring_eigenvalues,
                            max_ring_growth, helly_string_stability,
                            helly_string_stability_condition,
                            numeric_string_stability)


def test_helly_partial_derivatives_are_analytic():
    m = Helly(C1=0.3, C2=0.6, T=1.2, s0=2.0)
    s = m.equilibrium_gap(20.0)      # = s0 + T*v
    fs, fv, fdv = partial_derivatives(m, s, 20.0)
    assert fs == pytest.approx(m.C1, rel=1e-4)
    assert fv == pytest.approx(-m.C1 * m.T, rel=1e-4)
    assert fdv == pytest.approx(-m.C2, rel=1e-4)


def test_helly_margin_closed_form():
    m = Helly(C1=0.3, C2=0.6, T=1.2)
    ls = linear_stability(m, 20.0)
    expected = m.C1 - m.C1 * m.C2 * m.T - (m.C1 * m.T) ** 2 / 2.0
    assert ls["margin"] == pytest.approx(expected, rel=1e-6)
    # margin is independent of speed for the linear model
    assert linear_stability(m, 5.0)["margin"] == pytest.approx(expected, rel=1e-6)


def test_string_stability_condition_matches_transfer_function():
    rng = np.random.default_rng(0)
    omegas = np.logspace(-2, 1.5, 200)
    for _ in range(20):
        C1 = rng.uniform(0.05, 1.0)
        C2 = rng.uniform(0.05, 1.5)
        T = 1.2
        cond = helly_string_stability_condition(C1, C2, T)
        gmax = helly_string_stability(C1, C2, T, omegas).max()
        assert cond == (gmax <= 1.0 + 1e-9)


def test_ring_growth_sign_matches_margin():
    # a clearly unstable and a clearly stable IDM configuration
    unstable = IDM(T=0.4)
    stable = IDM(T=2.0)
    g_un = max_ring_growth(unstable, 40, 6.0)
    g_st = max_ring_growth(stable, 40, 20.0)
    m_un = linear_stability(unstable, 6.0)["margin"]
    m_st = linear_stability(stable, 20.0)["margin"]
    assert (g_un > 0) == (m_un > 0) and g_un > 0
    assert (g_st > 0) == (m_st > 0) and g_st < 0


def test_ring_eigenvalue_excludes_neutral_mode():
    m = IDM(T=2.0)
    ev = ring_eigenvalues(m, 20, 15.0)
    # there is an exact neutral translation mode at 0
    assert np.min(np.abs(ev)) < 1e-9
    assert max_ring_growth(m, 20, 15.0) < 0.0


def test_numeric_string_stability_matches_analytic():
    m = Helly(C1=0.4, C2=0.9, T=1.2)          # string-stable set
    assert helly_string_stability_condition(m.C1, m.C2, m.T)
    omega = 0.6
    num = numeric_string_stability(m, omega, v_eq=20.0, amplitude=0.4, dt=0.05,
                                   t_end=300.0)
    ana = float(helly_string_stability(m.C1, m.C2, m.T, np.array([omega]))[0])
    assert num == pytest.approx(ana, abs=0.05)


def test_fvd_more_stable_than_ovm():
    """Adding the speed-difference term lowers the stability margin."""
    o = linear_stability(OVM(), 12.0)["margin"]
    f = linear_stability(FVD(), 12.0)["margin"]
    assert f < o
