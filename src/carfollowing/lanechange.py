"""MOBIL lane-changing model and a minimal two-lane microscopic simulation.

MOBIL (Kesting, Treiber & Helbing, 2007) decides lane changes from an
*incentive* criterion

    U = a_i^new - a_i^cur + p [ (a_n^new - a_n^old) + (a_o^new - a_o^old) ]

(``i`` the changing vehicle, ``n`` the new follower, ``o`` the old follower,
``p`` the politeness factor) and a *safety* criterion ``a_n^new >= -b_safe``.
Longitudinal motion uses any :class:`~carfollowing.models.CarFollowingModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .models import CarFollowingModel, IDM

__all__ = ["MOBILParams", "Vehicle", "two_lane_incentive", "simulate_two_lane"]

B_SAFE = 3.0  # maximum safe imposed deceleration on the new follower [m/s^2]


@dataclass
class MOBILParams:
    politeness: float = 0.2       # p
    threshold: float = 0.2        # a_thr [m/s^2]
    b_safe: float = B_SAFE
    bias_right: float = 0.0       # keep-right bias [m/s^2]


@dataclass
class Vehicle:
    vid: int
    lane: int
    x: float
    v: float
    length: float = 5.0
    model: CarFollowingModel = field(default_factory=IDM)
    target_lane: int | None = None
    last_change: float = -1e9   # time of the most recent lane change [s]


def _leader(vehicles: list[Vehicle], lane: int, x: float, exclude: int | None = None):
    best = None
    for w in vehicles:
        if w.lane != lane or w.vid == exclude:
            continue
        if w.x > x and (best is None or w.x < best.x):
            best = w
    return best


def _follower(vehicles: list[Vehicle], lane: int, x: float, exclude: int | None = None):
    best = None
    for w in vehicles:
        if w.lane != lane or w.vid == exclude:
            continue
        if w.x < x and (best is None or w.x > best.x):
            best = w
    return best


def _accel(model: CarFollowingModel, self_x: float, self_v: float,
           lead: Vehicle | None) -> float:
    if lead is None:
        return model.accel(1e9, self_v, 0.0)
    s = lead.x - self_x - 0.5 * (lead.length + 5.0)
    return model.accel(max(s, 1e-3), self_v, self_v - lead.v)


def two_lane_incentive(veh: Vehicle, vehicles: list[Vehicle],
                       target_lane: int, p: MOBILParams) -> tuple[float, bool]:
    """MOBIL incentive ``U`` and safety flag for ``veh`` changing to ``target_lane``."""
    model = veh.model
    a_cur = _accel(model, veh.x, veh.v, _leader(vehicles, veh.lane, veh.x, veh.vid))

    new_lead = _leader(vehicles, target_lane, veh.x, veh.vid)
    new_foll = _follower(vehicles, target_lane, veh.x, veh.vid)
    a_new = _accel(model, veh.x, veh.v, new_lead)

    # new follower n: its leader becomes veh
    a_n_old = _accel(model, new_foll.x, new_foll.v,
                     _leader(vehicles, target_lane, new_foll.x, new_foll.vid)) if new_foll else 0.0
    a_n_new = _accel(model, new_foll.x, new_foll.v, veh) if new_foll else 0.0

    # old follower o: its leader becomes veh's old leader
    old_foll = _follower(vehicles, veh.lane, veh.x, veh.vid)
    a_o_old = _accel(model, old_foll.x, old_foll.v, veh) if old_foll else 0.0
    a_o_new = _accel(model, old_foll.x, old_foll.v,
                     _leader(vehicles, veh.lane, veh.x, veh.vid)) if old_foll else 0.0

    U = (a_new - a_cur) + p.politeness * ((a_n_new - a_n_old) + (a_o_new - a_o_old))
    if target_lane < veh.lane:
        U += p.bias_right
    else:
        U -= p.bias_right
    safe = (new_foll is None) or (a_n_new >= -p.b_safe)
    return float(U), bool(safe)


def simulate_two_lane(n_lane_vehicles: int = 12, length: int = 2, lane_gap: float = 60.0,
                      v0: float = 25.0, dt: float = 0.1, t_end: float = 60.0,
                      mobil: MOBILParams | None = None, seed: int = 1,
                      dwell_time: float = 5.0) -> dict[str, np.ndarray]:
    """Minimal two-lane microsimulation with IDM longitudinal + MOBIL lateral control.

    A slow vehicle is placed in the left lane to trigger overtaking manoeuvres.
    ``dwell_time`` enforces a minimum time between successive lane changes of the
    same vehicle (avoids chattering).  Returns trajectories per lane over time.
    """
    if mobil is None:
        mobil = MOBILParams()
    rng = np.random.default_rng(seed)
    vehicles: list[Vehicle] = []
    vid = 0
    for lane in range(length):
        for k in range(n_lane_vehicles):
            v = v0 * rng.uniform(0.92, 1.08)
            vehicles.append(Vehicle(vid=vid, lane=lane, x=k * lane_gap, v=v,
                                    model=IDM(v0=v0, T=1.2, s0=2.0, a=1.5, b=2.0)))
            vid += 1
    # a slow leader in lane 0 near the front
    vehicles[0].v = v0 * 0.55
    vehicles[0].model = IDM(v0=v0 * 0.55, T=1.2, s0=2.0, a=1.5, b=2.0)

    n_steps = int(round(t_end / dt))
    tarr = np.arange(n_steps + 1) * dt
    # record: array [n_steps+1, n_veh] of lane index +1 (0 = absent) and x, v
    ids = [w.vid for w in vehicles]
    idx = {w.vid: i for i, w in enumerate(vehicles)}
    n = len(vehicles)
    X = np.full((n_steps + 1, n), np.nan)
    V = np.full((n_steps + 1, n), np.nan)
    LANE = np.full((n_steps + 1, n), np.nan)

    def snapshot(step):
        for w in vehicles:
            i = idx[w.vid]
            X[step, i] = w.x; V[step, i] = w.v; LANE[step, i] = w.lane

    snapshot(0)
    for step in range(n_steps):
        t_now = step * dt
        # lateral: decide
        makes = {}
        for w in vehicles:
            if t_now - w.last_change < dwell_time:
                continue
            other = 1 - w.lane if length == 2 else (w.lane + 1) % length
            U, safe = two_lane_incentive(w, vehicles, other, mobil)
            if U > mobil.threshold and safe:
                makes[w.vid] = other
        for w in vehicles:
            if w.vid in makes:
                w.lane = makes[w.vid]
                w.last_change = t_now
        # longitudinal: RK-style Euler with IDM
        new = {}
        for w in vehicles:
            lead = _leader(vehicles, w.lane, w.x, w.vid)
            a = _accel(w.model, w.x, w.v, lead)
            v_new = max(0.0, w.v + dt * a)
            new[w.vid] = (w.x + 0.5 * (w.v + v_new) * dt, v_new)
        for w in vehicles:
            w.x, w.v = new[w.vid]
        snapshot(step + 1)

    return {"t": tarr, "x": X, "v": V, "lane": LANE, "ids": np.array(ids)}
