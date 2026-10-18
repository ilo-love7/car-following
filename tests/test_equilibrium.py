import numpy as np
import pytest

from carfollowing.equilibrium import (fundamental_diagram, capacity_point, ring_length_for,
                              EquilibriumCurve)
from carfollowing.models import IDM, Helly, Newell, OVM, make_model


def test_fundamental_diagram_columns_and_monotonic_density():
    fd = fundamental_diagram(IDM())
    assert list(fd.columns) == ["v", "gap", "headway", "density", "flow"]
    assert (fd["flow"] > 0).all()
    # density decreases as speed increases (free-flow branch)
    assert (np.diff(fd["density"]) < 1e-9).all()


def test_capacity_reasonable():
    cap = capacity_point(IDM())
    # IDM with default parameters gives ~2100 veh/h near 65 km/h
    assert 1500.0 < cap["capacity"] < 2600.0
    assert 10.0 < cap["v_crit"] < 30.0
    assert cap["rho_jam"] == pytest.approx(1000.0 / (2.0 + 5.0), rel=1e-6)


def test_capacity_for_all_models():
    for name in ("IDM", "OVM", "FVD", "Helly", "Newell"):
        cap = capacity_point(make_model(name))
        assert cap["capacity"] > 0.0
        assert cap["rho_crit"] > 0.0


def test_ring_length_places_equilibrium():
    m = IDM()
    v = 15.0
    n = 25
    L = 5.0
    length = ring_length_for(v, n, m, L)
    assert length == pytest.approx(n * (m.equilibrium_gap(v) + L), rel=1e-12)


def test_equilibrium_curve_wrapper():
    ec = EquilibriumCurve(IDM())
    v = 18.0
    assert ec.gap(v) == IDM().equilibrium_gap(v)
    assert ec.headway(v) == pytest.approx(ec.gap(v) + 5.0)
    assert ec.flow(v) > 0.0
