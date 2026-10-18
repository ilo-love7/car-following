import numpy as np

from carfollowing.datasets.synthetic import (generate_platoon, realistic_speed_profile,
                                     PlatoonData)


def test_profile_is_bounded_and_deterministic():
    t = np.arange(0.0, 300.0, 0.1)
    a = realistic_speed_profile(t, v_mean=25.0, seed=1)
    b = realistic_speed_profile(t, v_mean=25.0, seed=1)
    assert np.allclose(a, b)
    assert a.min() >= 3.0 and a.max() <= 40.0


def test_platoon_is_finite_collision_free_and_deterministic():
    p1 = generate_platoon(n_followers=4, t_end=150.0, dt=0.2, seed=5)
    p2 = generate_platoon(n_followers=4, t_end=150.0, dt=0.2, seed=5)
    assert p1.n_vehicles == 5
    for tr in p1.trajectories:
        assert np.all(np.isfinite(tr["x"]))
    for i in range(1, p1.n_vehicles):
        assert np.all(np.isfinite(p1.trajectories[i]["s"]))
        assert np.nanmin(p1.trajectories[i]["s"]) > -0.5      # no collision
        assert np.allclose(p1.trajectories[i]["x"], p2.trajectories[i]["x"])


def test_pair_interface():
    p = generate_platoon(n_followers=3, t_end=60.0, dt=0.2, seed=0)
    lead, t, s, v = p.pair(2)
    assert len(t) == len(s) == len(v)
    assert lead.t.shape == t.shape
    assert len(p.params) == 3
    assert isinstance(p, PlatoonData)


def test_heterogeneous_params_differ():
    p = generate_platoon(n_followers=4, t_end=40.0, dt=0.2, seed=3)
    v0s = [pp["v0"] for pp in p.params]
    assert len(set(round(x, 3) for x in v0s)) > 1
