"""Steady-state (equilibrium) analysis: microscopic fundamental diagram,
capacity and the ring geometry that is consistent with a chosen equilibrium.

The *microscopic fundamental diagram* follows from the equilibrium relation
``s_e(v)`` of a model: with the space headway ``h = s_e(v) + L`` the density is
``rho = 1/h`` and the flow is ``q = rho * v``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import CarFollowingModel

__all__ = ["ring_length_for", "fundamental_diagram", "capacity_point", "EquilibriumCurve"]


def ring_length_for(v_eq: float, n_vehicles: int, model: CarFollowingModel,
                    L: float = 5.0) -> float:
    """Ring circumference placing ``n_vehicles`` at the equilibrium gap of ``v_eq``."""
    return n_vehicles * (model.equilibrium_gap(v_eq) + L)


def _effective_vmax(model: CarFollowingModel, hard_cap: float = 40.0,
                    gap_cap: float = 1e4) -> float:
    """Largest speed (<= hard_cap) that has a finite, non-degenerate equilibrium gap."""
    def ok(v: float) -> bool:
        try:
            g = model.equilibrium_gap(v)
        except Exception:
            return False
        return np.isfinite(g) and g < gap_cap

    if ok(hard_cap):
        return hard_cap
    lo, hi = 1e-3, hard_cap
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def fundamental_diagram(model: CarFollowingModel, v_max: float | None = None,
                        n: int = 200, L: float = 5.0) -> pd.DataFrame:
    """Microscopic fundamental diagram of a model.

    Returns a DataFrame with columns ``v`` (speed), ``gap``, ``headway``,
    ``density`` (veh/km) and ``flow`` (veh/h).
    """
    if v_max is None:
        v_max = _effective_vmax(model)
        fs = model.free_speed()
        if fs is not None:
            v_max = min(v_max, fs)
    v = np.linspace(1e-3, 0.999 * v_max, n)
    gap = np.array([model.equilibrium_gap(float(vi)) for vi in v])
    headway = gap + L                       # [m]
    density = 1000.0 / headway              # [veh/km]
    flow = density * v * 3.6                # [veh/h]
    return pd.DataFrame({"v": v, "gap": gap, "headway": headway,
                         "density": density, "flow": flow})


def capacity_point(model: CarFollowingModel, L: float = 5.0) -> dict[str, float]:
    """Free-flow speed, capacity, critical density and jam density.

    Jam density uses the standstill gap of the model when available.
    """
    fd = fundamental_diagram(model, L=L)
    i = int(np.argmax(fd["flow"].to_numpy()))
    row = fd.iloc[i]
    p = model.get_params()
    s_min = p.get("s0", 2.0)
    jam_density = 1000.0 / (s_min + L)
    v_free = float(fd["v"].iloc[-1])
    return {
        "v_free": v_free,
        "capacity": float(row["flow"]),
        "v_crit": float(row["v"]),
        "rho_crit": float(row["density"]),
        "rho_jam": float(jam_density),
    }


class EquilibriumCurve:
    """Convenience wrapper caching the equilibrium relation of a model."""

    def __init__(self, model: CarFollowingModel, L: float = 5.0):
        self.model = model
        self.L = L

    def gap(self, v: float) -> float:
        return self.model.equilibrium_gap(v)

    def speed(self, s: float) -> float:
        return self.model.equilibrium_speed(s)

    def headway(self, v: float) -> float:
        return self.model.equilibrium_gap(v) + self.L

    def flow(self, v: float) -> float:
        return (1000.0 / self.headway(v)) * v * 3.6
