"""Longitudinal electric-vehicle energy model driven by a speed trace.

The tractive force requirement is

    F = m a + 0.5 rho Cd A v^2 + m g Crr + m g sin(theta)

with the usual rolling-resistance, aerodynamic and road-grade terms.  Positive
wheel power is drawn from the battery through the drivetrain efficiency,
negative wheel power is recuperated with a regenerative efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

__all__ = ["EVParams", "energy_from_speed", "compare_styles"]

G = 9.81
RHO_AIR = 1.2


@dataclass
class EVParams:
    mass: float = 1500.0        # vehicle mass [kg]
    Cd: float = 0.30            # drag coefficient [-]
    area: float = 2.2           # frontal area [m^2]
    Crr: float = 0.010          # rolling resistance coefficient [-]
    eta_drive: float = 0.90     # drivetrain efficiency (traction) [-]
    eta_regen: float = 0.65     # regenerative efficiency [-]
    aux_power: float = 300.0    # constant auxiliary load [W]
    grade: float = 0.0          # road grade [rad]

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def energy_from_speed(t: np.ndarray, v: np.ndarray, p: EVParams | None = None,
                      a: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Instantaneous battery power and cumulative energy for a speed trace.

    ``a`` may be supplied; otherwise it is obtained by finite differences.
    """
    if p is None:
        p = EVParams()
    t = np.asarray(t, float); v = np.asarray(v, float)
    if a is None:
        a = np.gradient(v, t)
    a = np.asarray(a, float)
    F_roll = p.mass * G * p.Crr
    F_grade = p.mass * G * np.sin(p.grade)
    F_drag = 0.5 * RHO_AIR * p.Cd * p.area * v ** 2
    F_inertia = p.mass * a
    F = F_inertia + F_drag + F_roll + F_grade

    P_wheel = F * v
    P_batt = np.where(P_wheel >= 0.0, P_wheel / p.eta_drive, P_wheel * p.eta_regen)
    P_batt = P_batt + p.aux_power

    E = np.concatenate([[0.0], np.cumsum(0.5 * (P_batt[1:] + P_batt[:-1]) * np.diff(t))])
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    return {"F": F, "P_wheel": P_wheel, "P_batt": P_batt, "E": E,
            "distance": distance}


def compare_styles(t: np.ndarray, v_trace: np.ndarray,
                   p: EVParams | None = None) -> dict[str, float]:
    """Energy summary for a speed trace: total, per-km and per-100 km."""
    r = energy_from_speed(t, v_trace, p)
    E = r["E"][-1]
    dist = r["distance"][-1]              # [m]
    dist_km = dist / 1000.0
    return {
        "energy_J": float(E),
        "energy_kWh": float(E / 3.6e6),
        "distance_km": float(dist_km),
        "kWh_per_100km": float((E / 3.6e6) / max(dist_km, 1e-9) * 100.0),
    }
