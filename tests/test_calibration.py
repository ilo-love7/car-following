import numpy as np
import pytest

from carfollowing.simulate import LeaderTrace
from carfollowing.datasets.synthetic import generate_platoon
from carfollowing.calibration import calibrate_following


def _pair(dt=0.2, t_end=80.0, seed=0):
    p = generate_platoon(n_followers=2, t_end=t_end, dt=dt, seed=seed)
    lead, t, s, v = p.pair(1)
    return lead, t, s, v, p.params[0]


def test_calibration_recovers_known_idm():
    lead, t, s, v, true = _pair()
    res = calibrate_following("IDM", lead, t, s, v,
                              fit_params=["v0", "T", "s0", "a", "b"],
                              n_restarts=3, dt=0.2, seed=0)
    assert res.success
    assert res.rmse_gap < 0.2
    assert res.rmse_speed < 0.2
    # recovered parameters close to the ground truth
    for k in ("v0", "T", "s0", "a", "b"):
        assert res.fitted[k] == pytest.approx(true[k], rel=0.15, abs=0.1)


def test_calibration_reduces_error_vs_default():
    """Calibrated IDM must fit the observed pair better than default IDM."""
    lead, t, s, v, true = _pair()
    res = calibrate_following("IDM", lead, t, s, v, n_restarts=2, dt=0.2, seed=1)
    from carfollowing.models import IDM
    from carfollowing.simulate import simulate_following
    default = IDM()
    sim = simulate_following(default, lead, v0=float(v[0]), s0=float(s[0]),
                             dt=0.2, t_end=float(t[-1]))
    rmse_default = float(np.sqrt(np.mean((np.interp(t, sim.t, sim.s) - s) ** 2)))
    assert res.rmse_gap <= rmse_default + 1e-6


def test_calibration_result_has_all_params():
    lead, t, s, v, true = _pair()
    res = calibrate_following("IDM", lead, t, s, v, n_restarts=1, dt=0.2, seed=2)
    for k in ("v0", "T", "s0", "a", "b", "delta"):
        assert k in res.params
