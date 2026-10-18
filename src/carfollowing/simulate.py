"""Time-domain simulation of car-following: a single follower on a leader
trace, and a ring road of ``N`` interacting vehicles.

Continuous models are integrated with a classical 4th-order Runge-Kutta
scheme; discrete (reaction-time) models are advanced with their own update rule
and a trapezoidal position update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .models import CarFollowingModel

__all__ = ["LeaderTrace", "simulate_following", "simulate_ring", "FollowResult"]

L_VEHICLE = 5.0  # default vehicle length [m]


@dataclass
class LeaderTrace:
    """Velocity / position history of the vehicle ahead, used as boundary data."""

    t: np.ndarray
    x: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        if not (len(self.t) == len(self.x) == len(self.v)):
            raise ValueError("t, x, v must have equal length")

    def position(self, t: float) -> float:
        return float(np.interp(t, self.t, self.x))

    def speed(self, t: float) -> float:
        return float(np.interp(t, self.t, self.v))

    @classmethod
    def from_speed_profile(cls, t: np.ndarray, v: np.ndarray, x0: float = 0.0) -> "LeaderTrace":
        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)
        x = np.empty_like(v)
        x[0] = x0
        # trapezoidal integration of the speed profile
        x[1:] = x0 + np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))
        return cls(t=t, x=x, v=v)


@dataclass
class FollowResult:
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    s: np.ndarray
    a: np.ndarray
    v_lead: np.ndarray
    x_lead: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in
                ("t", "x", "v", "s", "a", "v_lead", "x_lead")}


def _rk4_step(model: CarFollowingModel, t: float, x: float, v: float,
              dt: float, leader: Callable[[float], tuple[float, float]],
              L: float) -> tuple[float, float, float]:
    def rhs(tt: float, xx: float, vv: float) -> tuple[float, float]:
        xl, vl = leader(tt)
        s = xl - xx - L
        dv = vv - vl
        return vv, model.accel(s, vv, dv)

    dx1, dv1 = rhs(t, x, v)
    dx2, dv2 = rhs(t + 0.5 * dt, x + 0.5 * dt * dx1, v + 0.5 * dt * dv1)
    dx3, dv3 = rhs(t + 0.5 * dt, x + 0.5 * dt * dx2, v + 0.5 * dt * dv2)
    dx4, dv4 = rhs(t + dt, x + dt * dx3, v + dt * dv3)
    x_new = x + dt / 6.0 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
    v_new = v + dt / 6.0 * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
    return x_new, v_new, v_new - v


def _safe_initial_gap(model: CarFollowingModel, v0: float) -> float:
    """A finite initial equilibrium gap, robust to ``v0`` above the model's free speed."""
    fs = model.free_speed()
    v = v0 if fs is None else min(v0, 0.95 * fs)
    g = model.equilibrium_gap(v)
    if not np.isfinite(g):
        g = model.equilibrium_gap(0.5 * fs) if fs else 50.0
    if not np.isfinite(g):
        g = 50.0
    return float(g)


def simulate_following(model: CarFollowingModel, leader: LeaderTrace,
                       v0: float, s0: float | None = None,
                       dt: float | None = None, L: float = L_VEHICLE,
                       t_end: float | None = None) -> FollowResult:
    """Simulate one follower behind ``leader``.

    ``s0`` is the initial net gap; if omitted the model's equilibrium gap at
    ``v0`` is used (a clean start).  ``dt`` defaults to 0.1 s for continuous
    models and the model reaction time for discrete ones.
    """
    t = leader.t
    t_end = float(t[-1]) if t_end is None else float(t_end)
    if dt is None:
        dt = getattr(model, "tau", 0.1) if model.is_discrete else 0.1
    n = int(round(t_end / dt)) + 1
    tt = np.linspace(0.0, t_end, n)

    if s0 is None:
        s0 = _safe_initial_gap(model, v0)
    x = leader.position(0.0) - L - s0
    v = float(v0)

    def leader_at(ts: float) -> tuple[float, float]:
        return leader.position(ts), leader.speed(ts)

    xs = np.empty(n); vs = np.empty(n); acc = np.empty(n)
    ss = np.empty(n); vl_arr = np.empty(n); xl_arr = np.empty(n)
    xs[0] = x; vs[0] = v
    xl0, vl0 = leader_at(0.0)
    ss[0] = xl0 - x - L; acc[0] = model.accel(ss[0], v, v - vl0)
    vl_arr[0] = vl0; xl_arr[0] = xl0

    for i in range(1, n):
        ts = tt[i - 1]
        xl, vl = leader_at(ts)
        s = xl - x - L
        if model.is_discrete:
            v_new = model.next_speed(s, v, vl, dt)
            x = x + 0.5 * (v + v_new) * dt
            a = (v_new - v) / dt
            v = v_new
        else:
            # internal sub-stepping keeps RK4 stable when the prescribed leader
            # brakes hard and the sample interval dt is coarse
            nsub = max(1, int(np.ceil(dt / 0.02)))
            h = dt / nsub
            a = 0.0
            for _ in range(nsub):
                x, v, a = _rk4_step(model, ts, x, v, h, leader_at, L)
                v = max(v, 0.0)
                ts += h
        xs[i] = x; vs[i] = v; acc[i] = a
        xl2, vl2 = leader_at(tt[i])
        ss[i] = xl2 - x - L
        vl_arr[i] = vl2; xl_arr[i] = xl2

    return FollowResult(t=tt, x=xs, v=vs, s=ss, a=acc, v_lead=vl_arr, x_lead=xl_arr)


def simulate_ring(model: CarFollowingModel, n_vehicles: int, ring_length: float,
                  v_eq: float, dt: float = 0.05, t_end: float = 300.0,
                  L: float = L_VEHICLE, perturbation: float = 0.0,
                  seed: int = 0,
                  models: list[CarFollowingModel] | None = None,
                  x0: np.ndarray | None = None, v0: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Ring-road simulation with periodic boundaries.

    All vehicles travel in the same direction and follow the vehicle directly
    ahead; the platoon is closed into a ring.  A small sinusoidal speed
    perturbation of amplitude ``perturbation`` is added at ``t = 0`` to the
    homogeneous equilibrium state; alternatively pass explicit initial
    conditions ``x0`` / ``v0`` (used by the stability experiments to excite all
    modes).
    """
    if n_vehicles < 2:
        raise ValueError("a ring needs at least 2 vehicles")
    if models is None:
        models = [model] * n_vehicles

    # Homogeneous equilibrium state: uniform spacing, identical speed.
    spacing = ring_length / n_vehicles
    x = np.arange(n_vehicles) * spacing
    v = np.full(n_vehicles, v_eq)
    if perturbation:
        v = v + perturbation * np.sin(2 * np.pi * np.arange(n_vehicles) / n_vehicles)
    if x0 is not None:
        x = np.asarray(x0, float).copy()
    if v0 is not None:
        v = np.asarray(v0, float).copy()
    gap = spacing - L

    n_steps = int(round(t_end / dt))
    tarr = np.arange(n_steps + 1) * dt
    X = np.empty((n_steps + 1, n_vehicles))
    V = np.empty((n_steps + 1, n_vehicles))
    X[0] = x; V[0] = v

    continuous = not models[0].is_discrete

    def accel_all(xs: np.ndarray, vs: np.ndarray) -> np.ndarray:
        a = np.empty(n_vehicles)
        for i in range(n_vehicles):
            j = (i + 1) % n_vehicles  # vehicle directly ahead (positions increase)
            gap_i = (xs[j] - xs[i]) % ring_length - L
            dv = vs[i] - vs[j]
            a[i] = models[i].accel(gap_i, vs[i], dv)
        return a

    for k in range(n_steps):
        x_prev = X[k]; v_prev = V[k]
        if continuous:
            k1x, k1v = v_prev, accel_all(x_prev, v_prev)
            k2x, k2v = v_prev + 0.5 * dt * k1v, accel_all(x_prev + 0.5 * dt * k1x, v_prev + 0.5 * dt * k1v)
            k3x, k3v = v_prev + 0.5 * dt * k2v, accel_all(x_prev + 0.5 * dt * k2x, v_prev + 0.5 * dt * k2v)
            k4x, k4v = v_prev + dt * k3v, accel_all(x_prev + dt * k3x, v_prev + dt * k3v)
            x_new = x_prev + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v_new = v_prev + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            v_new = np.maximum(v_new, 0.0)
        else:
            v_new = np.empty(n_vehicles)
            for i in range(n_vehicles):
                j = (i + 1) % n_vehicles
                gap_i = (x_prev[j] - x_prev[i]) % ring_length - L
                v_new[i] = models[i].next_speed(gap_i, v_prev[i], v_prev[j], dt)
            x_new = x_prev + 0.5 * (v_prev + v_new) * dt
        X[k + 1] = x_new; V[k + 1] = v_new

    return {"t": tarr, "x": X, "v": V, "n": np.array(n_vehicles), "ring_length": np.array(ring_length),
            "gap": np.array(gap), "v_eq": np.array(v_eq)}
