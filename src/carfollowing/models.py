"""Car-following (longitudinal) models.

All quantities are SI: distance in m, speed in m/s, acceleration in m/s^2,
time in s.  The *net gap* ``s`` is bumper-to-bumper distance (i.e. the clear
space between the rear bumper of the leader and the front bumper of the
follower); it is produced from positions as ``s = x_lead - x_follower - L``
where ``L`` is the vehicle length.

Sign convention
---------------
The closing speed is ``dv = v_follower - v_leader``.  A follower that is
faster than its leader therefore has ``dv > 0`` and shrinks the gap.  Every
physically sensible model must satisfy

    f_s = d(accel)/ds  > 0      (larger gap -> accelerate)
    f_v = d(accel)/dv  < 0      (faster -> accelerate less)
    f_dv = d(accel)/ddv < 0     (closing in -> decelerate)

which is exactly what the linear stability analysis in :mod:`carfollowing.stability`
relies on.

Continuous-time models expose :meth:`accel`.  Discrete (reaction-time) models
set ``is_discrete = True`` and implement :meth:`next_speed` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

__all__ = [
    "CarFollowingModel",
    "IDM",
    "OVM",
    "FVD",
    "Helly",
    "GHR",
    "Gipps",
    "Newell",
    "MODEL_REGISTRY",
    "make_model",
    "DEFAULT_PARAMS",
]

# Guard against division by a vanishing gap (a crashed pair).  The value is far
# below any physically meaningful gap but keeps every model finite.
_S_MIN = 1e-3


class CarFollowingModel:
    """Base class for longitudinal car-following models."""

    name: str = "base"
    is_discrete: bool = False
    #: parameter names in canonical order (used by calibration / registry)
    param_names: tuple[str, ...] = ()

    def accel(self, s: float, v: float, dv: float) -> float:  # pragma: no cover
        raise NotImplementedError

    def next_speed(self, s: float, v: float, v_lead: float, dt: float) -> float:
        """Discrete update.  Default: forward Euler on :meth:`accel`."""
        return v + dt * self.accel(s, v, v - v_lead)

    def free_accel(self, v: float) -> float:
        """Acceleration with no leader (infinite gap)."""
        return self.accel(1e9, v, 0.0)

    def free_speed(self) -> float | None:
        """Free-flow desired speed, if the model defines one."""
        return None

    # -- parameters -------------------------------------------------------
    def get_params(self) -> dict[str, float]:
        raise NotImplementedError

    def with_params(self, **kw: float) -> "CarFollowingModel":
        raise NotImplementedError

    # -- equilibrium helpers ---------------------------------------------
    def equilibrium_speed(self, s: float) -> float:
        """Steady-state speed for a given equilibrium gap (numeric by default)."""
        lo, hi = 1e-6, 120.0
        f = lambda v: self.accel(max(s, _S_MIN), v, 0.0)  # noqa: E731
        if f(lo) < 0:
            return 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def equilibrium_gap(self, v: float) -> float:
        """Steady-state gap for a given equilibrium speed (numeric by default)."""
        if v <= 0:
            return _S_MIN
        lo, hi = _S_MIN, 5e3
        f = lambda s: self.accel(s, v, 0.0)  # noqa: E731
        if f(hi) < 0:
            raise RuntimeError(f"{self.name}: no equilibrium gap found for v={v}")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 1. Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000)
# ---------------------------------------------------------------------------
@dataclass
class IDM(CarFollowingModel):
    """Intelligent Driver Model.

    ``a = a_max * [ 1 - (v/v0)^delta - (s*/s)^2 ]`` with the desired gap
    ``s* = s0 + max(0, v T + v dv / (2 sqrt(a_max b)))``.
    """

    v0: float = 30.0     # desired speed [m/s]
    T: float = 1.2       # safe time headway [s]
    s0: float = 2.0      # minimum (jam) gap [m]
    a: float = 1.5       # maximum acceleration [m/s^2]
    b: float = 2.0       # comfortable deceleration [m/s^2]
    delta: float = 4.0   # free-flow exponent

    name = "IDM"
    param_names = ("v0", "T", "s0", "a", "b", "delta")

    def accel(self, s: float, v: float, dv: float) -> float:
        s = max(s, _S_MIN)
        s_star = self.s0 + max(0.0, v * self.T + v * dv / (2.0 * math.sqrt(self.a * self.b)))
        return self.a * (1.0 - (v / self.v0) ** self.delta - (s_star / s) ** 2)

    def equilibrium_gap(self, v: float) -> float:
        if v <= 0.0:
            return self.s0
        denom = 1.0 - (v / self.v0) ** self.delta
        if denom <= 0.0:
            return math.inf  # no finite equilibrium gap above the desired speed
        return (self.s0 + v * self.T) / math.sqrt(denom)

    def equilibrium_speed(self, s: float) -> float:
        # Solve s = (s0 + v T)/sqrt(1-(v/v0)^delta) for v numerically.
        return super().equilibrium_speed(s)

    def free_speed(self) -> float:
        return self.v0

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "IDM":
        d = asdict(self)
        d.update(kw)
        return IDM(**d)


# ---------------------------------------------------------------------------
# 2. Optimal Velocity Model (Bando et al., 1995)
# ---------------------------------------------------------------------------
@dataclass
class OVM(CarFollowingModel):
    """Optimal Velocity Model.

    ``a = alpha * (V(s) - v)`` with the Bando optimal-velocity function
    ``V(s) = v_max/2 * (tanh(s/h - 2) + tanh(2))``.
    """

    v_max: float = 33.0   # maximum speed [m/s]
    h: float = 25.0       # optimal-velocity length scale [m]
    alpha: float = 1.2    # sensitivity / relaxation rate [1/s]

    name = "OVM"
    param_names = ("v_max", "h", "alpha")

    def optimal_velocity(self, s: float) -> float:
        s = max(s, _S_MIN)
        return 0.5 * self.v_max * (math.tanh(s / self.h - 2.0) + math.tanh(2.0))

    def accel(self, s: float, v: float, dv: float) -> float:
        return self.alpha * (self.optimal_velocity(s) - v)

    def equilibrium_speed(self, s: float) -> float:
        return self.optimal_velocity(s)

    def free_speed(self) -> float:
        return 0.5 * self.v_max * (1.0 + math.tanh(2.0))

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "OVM":
        d = asdict(self)
        d.update(kw)
        return OVM(**d)


# ---------------------------------------------------------------------------
# 3. Full Velocity Difference model (Jiang, Wu & Zhu, 2001)
# ---------------------------------------------------------------------------
@dataclass
class FVD(OVM):
    """Full Velocity Difference model = OVM + a speed-difference term.

    ``a = alpha (V(s) - v) - lam * dv`` (``dv = v - v_lead``).  With ``lam = 0``
    it degenerates exactly to :class:`OVM`.
    """

    lam: float = 0.3

    name = "FVD"
    param_names = ("v_max", "h", "alpha", "lam")

    def accel(self, s: float, v: float, dv: float) -> float:
        return self.alpha * (self.optimal_velocity(s) - v) - self.lam * dv

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "FVD":
        d = asdict(self)
        d.update(kw)
        return FVD(**d)


# ---------------------------------------------------------------------------
# 4. Helly / linear ACC model
# ---------------------------------------------------------------------------
@dataclass
class Helly(CarFollowingModel):
    """Linear adaptive-cruise-control (Helly) model.

    ``a = C1 (s - s0 - T v) - C2 dv``.  Being linear it admits an exact
    analytic string-stability transfer function (see :mod:`carfollowing.stability`).
    """

    C1: float = 0.3       # gap-error gain [1/s^2]
    C2: float = 0.6       # closing-speed gain [1/s]
    T: float = 1.2        # desired time headway [s]
    s0: float = 2.0       # standstill gap [m]

    name = "Helly"
    param_names = ("C1", "C2", "T", "s0")

    def accel(self, s: float, v: float, dv: float) -> float:
        return self.C1 * (s - self.s0 - self.T * v) - self.C2 * dv

    def equilibrium_gap(self, v: float) -> float:
        return self.s0 + self.T * v

    def equilibrium_speed(self, s: float) -> float:
        return max(0.0, (s - self.s0) / self.T)

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "Helly":
        d = asdict(self)
        d.update(kw)
        return Helly(**d)


# ---------------------------------------------------------------------------
# 5. Gazis-Herman-Rothery (General Motors) model
# ---------------------------------------------------------------------------
@dataclass
class GHR(CarFollowingModel):
    """General Motors model: ``a = c * v^m * (v_lead - v) / s^l``.

    Note: with ``dv = 0`` the acceleration vanishes for *any* gap, so GHR has no
    unique equilibrium relation.  It is included for historical completeness and
    for calibration comparisons, not for fundamental-diagram analysis.
    """

    c: float = 0.6
    m: float = 0.0
    l: float = 1.5

    name = "GHR"
    param_names = ("c", "m", "l")

    def accel(self, s: float, v: float, dv: float) -> float:
        s = max(s, _S_MIN)
        vm = max(v, 1e-3) ** self.m
        return -self.c * vm * dv / (s ** self.l)

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "GHR":
        d = asdict(self)
        d.update(kw)
        return GHR(**d)


# ---------------------------------------------------------------------------
# 6. Gipps model (discrete, safety based)
# ---------------------------------------------------------------------------
@dataclass
class Gipps(CarFollowingModel):
    """Gipps (1981) safe-speed model, evaluated on a fixed reaction time.

    ``v(t+tau) = min(v_acc, v_safe)`` with

        v_acc  = v + 2.5 a tau (1 - v/v_des) sqrt(0.025 + v/v_des)
        v_safe = -b tau + sqrt( b^2 tau^2 + v_lead^2 + 2 b (s - s0 - v tau) )

    The square root is clipped at zero.  This is a discrete-time model:
    ``dt`` in :meth:`next_speed` is ignored in favour of the model's own
    reaction time ``tau``.  Its equilibrium relation is ``s_e = s0 + 2 v tau``.
    """

    v_des: float = 30.0
    a: float = 1.5
    b: float = 2.0
    s0: float = 2.0
    tau: float = 1.0

    name = "Gipps"
    is_discrete = True
    param_names = ("v_des", "a", "b", "s0", "tau")

    def next_speed(self, s: float, v: float, v_lead: float, dt: float) -> float:
        s = max(s, _S_MIN)
        v_over = min(v / self.v_des, 1.0)
        v_acc = v + 2.5 * self.a * self.tau * (1.0 - v_over) * math.sqrt(0.025 + v_over)
        v_acc = max(v_acc, 0.0)
        disc = (self.b * self.tau) ** 2 + v_lead ** 2 + \
            2.0 * self.b * (s - self.s0 - v * self.tau)
        v_safe = -self.b * self.tau + math.sqrt(disc) if disc > 0.0 else 0.0
        return max(min(v_acc, v_safe), 0.0)

    def equilibrium_gap(self, v: float) -> float:
        return self.s0 + 2.0 * v * self.tau

    def equilibrium_speed(self, s: float) -> float:
        return max(0.0, (s - self.s0) / (2.0 * self.tau))

    def accel(self, s: float, v: float, dv: float) -> float:
        return (self.next_speed(s, v, v - dv, self.tau) - v) / self.tau

    def free_speed(self) -> float:
        return self.v_des

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "Gipps":
        d = asdict(self)
        d.update(kw)
        return Gipps(**d)


# ---------------------------------------------------------------------------
# 7. Newell simplified (kinematic safe-speed) model
# ---------------------------------------------------------------------------
@dataclass
class Newell(CarFollowingModel):
    """Newell simplified car-following model.

    The follower adopts the safe speed ``v = min(v_free, (s - s0)/tau)``,
    i.e. it keeps at least the jam spacing ``s0`` while preserving a reaction
    time ``tau`` of clearance.  Implemented as a discrete speed assignment.
    """

    v_free: float = 30.0
    tau: float = 1.2
    s0: float = 2.0

    name = "Newell"
    is_discrete = True
    param_names = ("v_free", "tau", "s0")

    def safe_speed(self, s: float) -> float:
        return min(self.v_free, max(0.0, (max(s, _S_MIN) - self.s0) / self.tau))

    def next_speed(self, s: float, v: float, v_lead: float, dt: float) -> float:
        return self.safe_speed(s)

    def accel(self, s: float, v: float, dv: float) -> float:
        return (self.safe_speed(s) - v) / self.tau

    def equilibrium_gap(self, v: float) -> float:
        return self.s0 + min(v, self.v_free) * self.tau

    def equilibrium_speed(self, s: float) -> float:
        return self.safe_speed(s)

    def free_speed(self) -> float:
        return self.v_free

    def get_params(self) -> dict[str, float]:
        return asdict(self)

    def with_params(self, **kw: float) -> "Newell":
        d = asdict(self)
        d.update(kw)
        return Newell(**d)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[CarFollowingModel]] = {
    "IDM": IDM,
    "OVM": OVM,
    "FVD": FVD,
    "Helly": Helly,
    "GHR": GHR,
    "Gipps": Gipps,
    "Newell": Newell,
}

#: Sensible calibrated default parameters for a typical motorway scenario.
DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "IDM": dict(v0=30.0, T=1.2, s0=2.0, a=1.5, b=2.0, delta=4.0),
    "OVM": dict(v_max=33.0, h=25.0, alpha=1.2),
    "FVD": dict(v_max=33.0, h=25.0, alpha=1.2, lam=0.3),
    "Helly": dict(C1=0.3, C2=0.6, T=1.2, s0=2.0),
    "GHR": dict(c=0.6, m=0.0, l=1.5),
    "Gipps": dict(v_des=30.0, a=1.5, b=2.0, s0=2.0, tau=1.0),
    "Newell": dict(v_free=30.0, tau=1.2, s0=2.0),
}


def make_model(name: str, **overrides: float) -> CarFollowingModel:
    """Instantiate a registered model, optionally overriding parameters."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    params = dict(DEFAULT_PARAMS[name])
    params.update(overrides)
    return MODEL_REGISTRY[name](**params)
