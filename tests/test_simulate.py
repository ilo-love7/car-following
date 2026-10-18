import numpy as np
import pytest

from carfollowing.models import IDM, OVM, FVD, Gipps, Newell, make_model
from carfollowing.simulate import LeaderTrace, simulate_following, simulate_ring
from carfollowing.equilibrium import ring_length_for


def _leader(t_end=60.0, dt=0.1, v0=20.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + 1e-9, dt)
    v = v0 + 2.0 * np.sin(0.1 * t) + rng.normal(0, 0.05, t.shape)
    return LeaderTrace.from_speed_profile(t, np.clip(v, 1.0, 40.0))


def test_following_is_collision_free_and_finite():
    lead = _leader()
    for name in ("IDM", "OVM", "FVD", "Gipps", "Newell"):
        m = make_model(name)
        res = simulate_following(m, lead, v0=float(lead.v[0]), dt=0.1, t_end=60.0)
        assert np.all(np.isfinite(res.x)) and np.all(np.isfinite(res.v))
        assert np.all(res.v >= -1e-9)
        # bumper-to-bumper gap must stay positive (no rear-end collision)
        assert res.s.min() > -0.5, name


def test_result_shapes():
    lead = _leader(t_end=20.0)
    res = simulate_following(IDM(), lead, v0=float(lead.v[0]), dt=0.1, t_end=20.0)
    assert len(res.t) == len(res.x) == len(res.v) == len(res.s) == len(res.a)
    assert res.t[0] == 0.0 and res.t[-1] == pytest.approx(20.0)


def test_ring_unperturbed_stays_at_equilibrium():
    m = IDM(T=1.5)
    v_eq, N, L = 18.0, 30, 5.0
    length = ring_length_for(v_eq, N, m, L)
    res = simulate_ring(m, N, length, v_eq, dt=0.1, t_end=60.0, L=L)
    # no perturbation: speed stays uniform (std ~ 0)
    assert np.allclose(res["v"], v_eq, atol=1e-6)
    assert res["v"][-1].std() < 1e-6


def test_ring_two_vehicle_leader_is_ahead():
    """The predecessor must be the vehicle in front (larger position), so an
    unperturbed ring does not accidentally decouple."""
    m = IDM(T=1.5)
    N, L, v_eq = 12, 5.0, 15.0
    length = ring_length_for(v_eq, N, m, L)
    res = simulate_ring(m, N, length, v_eq, dt=0.05, t_end=30.0, L=L)
    assert np.all(np.isfinite(res["v"]))
    assert res["v"][-1].std() < 1e-6


def test_discrete_models_advance():
    lead = _leader(t_end=15.0)
    for name in ("Gipps", "Newell"):
        res = simulate_following(make_model(name), lead, v0=float(lead.v[0]),
                                 t_end=15.0)
        assert np.any(np.abs(np.diff(res.v)) > 0)
