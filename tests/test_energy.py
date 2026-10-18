import numpy as np
import pytest

from carfollowing.energy import EVParams, energy_from_speed, compare_styles

G = 9.81


def test_constant_speed_matches_analytic():
    p = EVParams(mass=1500.0, Cd=0.30, area=2.2, Crr=0.010,
                 eta_drive=0.90, eta_regen=0.65, aux_power=300.0, grade=0.0)
    v = 20.0
    t = np.arange(0.0, 100.0, 0.1)
    vv = np.full_like(t, v)
    r = energy_from_speed(t, vv, p)

    F = 0.5 * 1.2 * p.Cd * p.area * v ** 2 + p.mass * G * p.Crr   # a = 0
    P_batt = F * v / p.eta_drive + p.aux_power
    expected_E = P_batt * t[-1]
    assert r["E"][-1] == pytest.approx(expected_E, rel=1e-3)
    assert r["distance"][-1] == pytest.approx(v * t[-1], rel=1e-6)


def test_regeneration_reduces_energy():
    t = np.arange(0.0, 60.0, 0.1)
    v = 15.0 + 8.0 * np.sin(2 * np.pi * t / 20.0)      # heavy accel/brake cycling
    with_regen = compare_styles(t, v, EVParams(eta_regen=0.65))["energy_kWh"]
    no_regen = compare_styles(t, v, EVParams(eta_regen=0.0))["energy_kWh"]
    assert with_regen < no_regen


def test_consumption_units_are_kwh_per_100km():
    t = np.arange(0.0, 200.0, 0.1)
    v = np.full_like(t, 25.0)
    s = compare_styles(t, v, EVParams())
    # 25 m/s for 200 s = 5 km; consumption should be a few tens of kWh/100km
    assert s["distance_km"] == pytest.approx(5.0, rel=1e-3)
    assert 5.0 < s["kWh_per_100km"] < 40.0
