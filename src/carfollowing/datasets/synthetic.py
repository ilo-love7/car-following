"""Synthetic car-following benchmark data.

Because raw trajectory measurements (NGSIM, highD) require registration and are
large, this module generates a *reproducible* benchmark: a platoon of
heterogeneous IDM drivers following a realistic leader speed profile, with
optional measurement noise.  It is the default data source for calibration,
prediction and energy experiments, and mirrors the structure of a real
extracted car-following pair so the NGSIM adapter can be swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..models import IDM
from ..simulate import LeaderTrace, simulate_following

__all__ = ["PlatoonData", "realistic_speed_profile", "generate_platoon", "IDM_PRIORS"]

#: plausible population priors for passenger-car IDM parameters
IDM_PRIORS: dict[str, tuple[float, float]] = {
    "v0": (28.0, 3.0),      # mean, sd
    "T": (1.2, 0.30),
    "s0": (2.0, 0.30),
    "a": (1.5, 0.35),
    "b": (2.0, 0.45),
    "delta": (4.0, 0.0),
}


@dataclass
class PlatoonData:
    t: np.ndarray
    #: trajectories[i] is a dict with x, v, s for vehicle i (0 = leader)
    trajectories: list[dict[str, np.ndarray]]
    params: list[dict[str, float]]
    L: float = 5.0

    @property
    def n_vehicles(self) -> int:
        return len(self.trajectories)

    def leader_trace(self) -> LeaderTrace:
        tr = self.trajectories[0]
        return LeaderTrace(t=self.t, x=tr["x"], v=tr["v"])

    def pair(self, i: int) -> tuple[LeaderTrace, np.ndarray, np.ndarray, np.ndarray]:
        """Return (leader_trace, t, gap, speed) for follower ``i`` (1-based index)."""
        lead = LeaderTrace(t=self.t, x=self.trajectories[i - 1]["x"], v=self.trajectories[i - 1]["v"])
        return lead, self.t, self.trajectories[i]["s"], self.trajectories[i]["v"]


def realistic_speed_profile(t: np.ndarray, v_mean: float = 25.0,
                            seed: int = 0) -> np.ndarray:
    """A realistic leader speed trace: slow drift + traffic waves + braking events."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    v = np.full_like(t, v_mean)
    # slow drift and mild oscillation (traffic waves)
    v += 3.0 * np.sin(2 * np.pi * t / 220.0 + rng.uniform(0, 6.28))
    v += 1.5 * np.sin(2 * np.pi * t / 37.0 + rng.uniform(0, 6.28))
    # a few braking/acceleration events
    n_events = max(1, int(t[-1] / 60))
    for _ in range(n_events):
        t0 = rng.uniform(0.1, 0.9) * t[-1]
        dur = rng.uniform(6.0, 14.0)
        depth = rng.uniform(4.0, 10.0)
        v = v - depth * np.exp(-0.5 * ((t - t0) / (dur / 2.0)) ** 2)
    return np.clip(v, 3.0, 40.0)


def generate_platoon(n_followers: int = 5, t_end: float = 300.0, dt: float = 0.1,
                     v_mean: float = 25.0, noise: float = 0.0, seed: int = 0,
                     heterogeneous: bool = True, L: float = 5.0) -> PlatoonData:
    """Generate a platoon of ``n_followers`` IDM vehicles behind a leader.

    ``noise`` is the standard deviation of additive Gaussian measurement noise
    applied to positions (and, proportionally, speeds) in the saved data — the
    noise-free trajectories are used internally for propagation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + 1e-9, dt)
    v_lead = realistic_speed_profile(t, v_mean=v_mean, seed=seed)
    trace = LeaderTrace.from_speed_profile(t, v_lead)

    params: list[dict[str, float]] = []
    trajectories: list[dict[str, np.ndarray]] = [{"x": trace.x, "v": trace.v, "s": np.full_like(t, np.nan)}]

    leader = trace
    for i in range(n_followers):
        if heterogeneous:
            p = {k: float(rng.normal(mu, sd)) for k, (mu, sd) in IDM_PRIORS.items()}
            p["v0"] = float(np.clip(p["v0"], 18.0, 38.0))
            p["T"] = float(np.clip(p["T"], 0.6, 2.5))
            p["s0"] = float(np.clip(p["s0"], 1.0, 4.0))
            p["a"] = float(np.clip(p["a"], 0.6, 3.0))
            p["b"] = float(np.clip(p["b"], 1.0, 3.5))
            p["delta"] = 4.0
        else:
            p = {"v0": 28.0, "T": 1.2, "s0": 2.0, "a": 1.5, "b": 2.0, "delta": 4.0}
        model = IDM(**p)
        fs = model.free_speed()
        v_init = min(float(leader.v[0]), 0.98 * fs)
        res = simulate_following(model, leader, v0=v_init, dt=dt, L=L, t_end=t_end)
        # measured gap: leader position minus follower position minus L
        s = leader.x - res.x - L
        x_save = res.x.copy(); v_save = res.v.copy()
        if noise > 0:
            x_save = x_save + rng.normal(0.0, noise, size=x_save.shape)
            v_save = v_save + rng.normal(0.0, noise / max(dt, 1e-3) * 0.05, size=v_save.shape)
        trajectories.append({"x": x_save, "v": v_save, "s": s})
        params.append(p)
        leader = LeaderTrace(t=res.t, x=res.x, v=res.v)

    return PlatoonData(t=t, trajectories=trajectories, params=params, L=L)
