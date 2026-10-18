import numpy as np

from carfollowing.models import IDM
from carfollowing.lanechange import (Vehicle, MOBILParams, two_lane_incentive,
                             simulate_two_lane)


def test_incentive_positive_when_leader_is_slow_and_target_lane_free():
    m = IDM(v0=25.0)
    slow = Vehicle(vid=0, lane=0, x=100.0, v=12.0, model=m)
    fast = Vehicle(vid=1, lane=0, x=55.0, v=25.0, model=m)
    U, safe = two_lane_incentive(fast, [slow, fast], target_lane=1,
                                 p=MOBILParams(politeness=0.0))
    assert U > 0.0
    assert safe


def test_safety_blocked_by_close_new_follower():
    m = IDM(v0=25.0)
    slow = Vehicle(vid=0, lane=0, x=120.0, v=12.0, model=m)
    fast = Vehicle(vid=1, lane=0, x=70.0, v=26.0, model=m)
    # a fast, very close follower already in the target lane
    tail = Vehicle(vid=2, lane=1, x=68.0, v=20.0, model=m)
    U, safe = two_lane_incentive(fast, [slow, fast, tail], target_lane=1,
                                 p=MOBILParams(politeness=0.0))
    # inserting in front of a close follower must not be declared safe
    assert not safe


def test_two_lane_simulation_runs():
    res = simulate_two_lane(n_lane_vehicles=10, t_end=60.0, seed=1,
                            mobil=MOBILParams(politeness=0.2, threshold=0.2))
    assert res["x"].shape[0] == len(res["t"])
    lane = res["lane"]
    changes = int(np.sum(np.abs(np.diff(np.nan_to_num(lane, nan=-1.0), axis=0)) > 0.5))
    assert changes >= 1
    assert np.all(res["v"][np.isfinite(res["v"])] >= -1e-9)
