"""Performance and safety metrics for car-following trajectories."""

from __future__ import annotations

import numpy as np

__all__ = [
    "rmse", "mae", "gap_errors", "speed_errors",
    "time_to_collision", "time_headway", "drac", "safety_report",
]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def gap_errors(s_sim: np.ndarray, s_ref: np.ndarray) -> dict[str, float]:
    s_sim = np.asarray(s_sim, float); s_ref = np.asarray(s_ref, float)
    err = s_sim - s_ref
    return {"rmse": rmse(s_sim, s_ref), "mae": mae(s_sim, s_ref),
            "bias": float(np.mean(err)), "max_abs": float(np.max(np.abs(err)))}


def speed_errors(v_sim: np.ndarray, v_ref: np.ndarray) -> dict[str, float]:
    v_sim = np.asarray(v_sim, float); v_ref = np.asarray(v_ref, float)
    return {"rmse": rmse(v_sim, v_ref), "mae": mae(v_sim, v_ref),
            "bias": float(np.mean(v_sim - v_ref))}


def time_to_collision(s: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """TTC in seconds; ``inf`` where the follower is not closing in (``dv <= 0``)."""
    s = np.asarray(s, float); dv = np.asarray(dv, float)
    ttc = np.full_like(s, np.inf)
    mask = dv > 1e-6
    ttc[mask] = s[mask] / dv[mask]
    return ttc


def time_headway(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Time headway in seconds; ``inf`` at standstill."""
    s = np.asarray(s, float); v = np.asarray(v, float)
    thw = np.full_like(s, np.inf)
    mask = v > 1e-3
    thw[mask] = s[mask] / v[mask]
    return thw


def drac(s: np.ndarray, v: np.ndarray, v_lead: np.ndarray) -> np.ndarray:
    """Deceleration Rate to Avoid a Crash (DRAC) in m/s^2."""
    s = np.asarray(s, float); v = np.asarray(v, float); vl = np.asarray(v_lead, float)
    d = v - vl
    out = np.zeros_like(s)
    mask = d > 0
    out[mask] = d[mask] ** 2 / (2.0 * np.maximum(s[mask], 1e-6))
    return out


def safety_report(s: np.ndarray, v: np.ndarray, v_lead: np.ndarray,
                  ttc_threshold: float = 1.5) -> dict[str, float]:
    """Aggregate safety indicators for a following event."""
    s = np.asarray(s, float); v = np.asarray(v, float); v_lead = np.asarray(v_lead, float)
    ttc = time_to_collision(s, v - v_lead)
    thw = time_headway(s, v)
    d = drac(s, v, v_lead)
    dt = 1.0
    return {
        "min_gap": float(np.min(s)),
        "collisions": int(np.sum(s <= 0.0)),
        "frac_ttc_below": float(np.mean(ttc < ttc_threshold)),
        "min_ttc": float(np.min(ttc)),
        "mean_thw": float(np.mean(thw[np.isfinite(thw)])) if np.any(np.isfinite(thw)) else float("inf"),
        "max_drac": float(np.max(d)),
        "frac_drac_above_3": float(np.mean(d > 3.0)),
    }
