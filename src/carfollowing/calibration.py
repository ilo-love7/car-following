"""Parameter calibration of car-following models against observed trajectories.

The calibration solves a bounded non-linear least-squares problem: given a
leader trajectory and the observed follower trajectory, adjust a chosen subset
of model parameters so the simulated follower reproduces the observed gap and
speed.  Multiple random restarts are used to mitigate local minima.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .models import CarFollowingModel, MODEL_REGISTRY, make_model
from .simulate import LeaderTrace, simulate_following

__all__ = ["CalibrationResult", "calibrate_following"]

#: default search bounds per parameter (used when none are supplied)
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (5.0, 45.0), "T": (0.3, 3.0), "s0": (0.5, 8.0),
    "a": (0.2, 4.0), "b": (0.5, 6.0), "delta": (1.0, 8.0),
    "v_max": (5.0, 45.0), "h": (5.0, 80.0), "alpha": (0.1, 5.0),
    "lam": (0.0, 2.0),
    "C1": (0.05, 3.0), "C2": (0.05, 3.0),
    "c": (0.05, 5.0), "m": (0.0, 3.0), "l": (0.5, 4.0),
    "v_des": (5.0, 45.0), "tau": (0.3, 2.5),
    "v_free": (5.0, 45.0),
}


@dataclass
class CalibrationResult:
    model_name: str
    fitted: dict[str, float]
    params: dict[str, float]
    cost: float
    rmse_gap: float
    rmse_speed: float
    n_obs: int
    success: bool
    nfev: int

    def sim_model(self) -> CarFollowingModel:
        return MODEL_REGISTRY[self.model_name](**self.params)


def _residuals(x: np.ndarray, model_name: str, base: dict[str, float],
               fit_names: list[str], leader: LeaderTrace,
               obs_t: np.ndarray, obs_s: np.ndarray, obs_v: np.ndarray,
               w_s: float, w_v: float, dt: float, L: float) -> np.ndarray:
    params = dict(base)
    for name, val in zip(fit_names, x):
        params[name] = float(val)
    model = MODEL_REGISTRY[model_name](**params)
    res = simulate_following(model, leader, v0=float(obs_v[0]),
                             s0=float(obs_s[0]), dt=dt, L=L,
                             t_end=float(obs_t[-1]))
    s_sim = np.interp(obs_t, res.t, res.s)
    v_sim = np.interp(obs_t, res.t, res.v)
    return np.concatenate([(s_sim - obs_s) / w_s, (v_sim - obs_v) / w_v])


def calibrate_following(model_name: str, leader: LeaderTrace,
                        obs_t: np.ndarray, obs_s: np.ndarray, obs_v: np.ndarray,
                        fit_params: list[str] | None = None,
                        base_params: dict[str, float] | None = None,
                        bounds: dict[str, tuple[float, float]] | None = None,
                        dt: float = 0.1, L: float = 5.0,
                        w_s: float | None = None, w_v: float | None = None,
                        n_restarts: int = 6, seed: int = 0) -> CalibrationResult:
    """Calibrate ``model_name`` to an observed follower trajectory.

    The leader trajectory is the boundary condition; ``obs_t/obs_s/obs_v`` are
    the observed follower time, gap and speed.  ``fit_params`` selects which
    parameters are optimised (defaults to every free parameter of the model).
    """
    model0 = make_model(model_name)
    all_names = list(model0.param_names)
    fit_names = list(fit_params) if fit_params else all_names
    base = dict(model0.get_params())
    if base_params:
        base.update(base_params)

    bnds = dict(DEFAULT_BOUNDS)
    if bounds:
        bnds.update(bounds)
    lo = np.array([bnds[n][0] for n in fit_names], float)
    hi = np.array([bnds[n][1] for n in fit_names], float)

    obs_t = np.asarray(obs_t, float)
    obs_s = np.asarray(obs_s, float)
    obs_v = np.asarray(obs_v, float)
    w_s = w_s if w_s is not None else max(np.std(obs_s), 1.0)
    w_v = w_v if w_v is not None else max(np.std(obs_v), 1.0)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        if k == 0:
            x0 = np.clip([base[n] for n in fit_names], lo, hi)
        else:
            x0 = lo + rng.random(len(fit_names)) * (hi - lo)
        try:
            sol = least_squares(
                _residuals, x0, bounds=(lo, hi),
                args=(model_name, base, fit_names, leader, obs_t, obs_s, obs_v,
                      w_s, w_v, dt, L),
                method="trf", max_nfev=300)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError(f"calibration of {model_name} failed for all restarts")

    params = dict(base)
    for name, val in zip(fit_names, best.x):
        params[name] = float(val)
    model = MODEL_REGISTRY[model_name](**params)
    res = simulate_following(model, leader, v0=float(obs_v[0]), s0=float(obs_s[0]),
                             dt=dt, L=L, t_end=float(obs_t[-1]))
    s_sim = np.interp(obs_t, res.t, res.s)
    v_sim = np.interp(obs_t, res.t, res.v)
    return CalibrationResult(
        model_name=model_name,
        fitted={n: float(v) for n, v in zip(fit_names, best.x)},
        params=params,
        cost=float(best.cost),
        rmse_gap=float(np.sqrt(np.mean((s_sim - obs_s) ** 2))),
        rmse_speed=float(np.sqrt(np.mean((v_sim - obs_v) ** 2))),
        n_obs=len(obs_t),
        success=bool(best.success),
        nfev=int(best.nfev),
    )
