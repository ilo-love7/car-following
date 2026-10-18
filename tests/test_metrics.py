import numpy as np
import pytest

from carfollowing.metrics import (time_to_collision, time_headway, drac, safety_report,
                          rmse, mae)


def test_ttc_closing_and_opening():
    s = np.array([20.0, 20.0, 30.0])
    dv = np.array([10.0, -5.0, 0.0])       # only the first is closing in
    ttc = time_to_collision(s, dv)
    assert ttc[0] == 2.0
    assert np.isinf(ttc[1]) and np.isinf(ttc[2])


def test_time_headway():
    s = np.array([20.0, 5.0])
    v = np.array([10.0, 0.0])              # second is at standstill
    thw = time_headway(s, v)
    assert thw[0] == 2.0
    assert np.isinf(thw[1])


def test_drac():
    s = np.array([10.0, 10.0])
    v = np.array([14.0, 10.0])
    vl = np.array([10.0, 10.0])
    d = drac(s, v, vl)
    assert d[0] == 4.0 ** 2 / (2 * 10.0)
    assert d[1] == 0.0


def test_rmse_mae():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert rmse(a, b) == 0.0 and mae(a, b) == 0.0
    c = np.array([2.0, 3.0, 4.0])
    assert rmse(a, c) == 1.0


def test_safety_report_flags_collision():
    s = np.array([5.0, -1.0, 10.0])
    v = np.array([10.0, 10.0, 10.0])
    vl = np.array([8.0, 8.0, 8.0])
    rep = safety_report(s, v, vl)
    assert rep["collisions"] == 1
    assert rep["min_gap"] < 0
    # time-headways: 0.5 s, -0.1 s, 1.0 s (all finite)
    assert rep["mean_thw"] == pytest.approx((0.5 - 0.1 + 1.0) / 3.0)
