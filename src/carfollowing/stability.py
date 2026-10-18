"""Stability analysis for car-following models.

Three complementary notions are provided:

1. **Local (plane-wave) linear stability** of a homogeneous stream.  For a model
   ``dv/dt = f(s, v, dv)`` the classical long-wavelength condition is derived in
   the technical report and implemented here as the sign of

       m = f_s - f_dv f_v - f_v^2 / 2

   with ``m < 0`` (strictly) meaning linearly stable.  The partial derivatives
   are evaluated at the equilibrium point ``(s_e(v), v, 0)``.

2. **Ring eigenvalue spectrum**: the exact linearised dynamics of ``N`` vehicles
   on a ring, whose largest real eigenvalue gives the growth rate of the most
   unstable mode (including finite-wavelength modes the asymptotic condition
   cannot see).

3. **String stability** (relevant to ACC): whether a velocity perturbation is
   amplified as it propagates.  For the linear :class:`~carfollowing.models.Helly` model
   this is available in closed form; for any model it is measured numerically by
   driving a platoon sinusoidally.
"""

from __future__ import annotations

import numpy as np

from .models import CarFollowingModel, Helly
from .simulate import LeaderTrace, simulate_following

__all__ = [
    "partial_derivatives",
    "linear_stability",
    "ring_eigenvalues",
    "max_ring_growth",
    "helly_string_stability",
    "numeric_string_stability",
]


def partial_derivatives(model: CarFollowingModel, s_eq: float, v_eq: float,
                        h: float = 1e-4) -> tuple[float, float, float]:
    """Central-difference estimates of ``f_s``, ``f_v`` and ``f_dv`` at ``(s_eq, v_eq, 0)``."""
    f = model.accel
    fs = (f(s_eq + h, v_eq, 0.0) - f(s_eq - h, v_eq, 0.0)) / (2 * h)
    fv = (f(s_eq, v_eq + h, 0.0) - f(s_eq, v_eq - h, 0.0)) / (2 * h)
    fdv = (f(s_eq, v_eq, h) - f(s_eq, v_eq, -h)) / (2 * h)
    return fs, fv, fdv


def linear_stability(model: CarFollowingModel, v_eq: float,
                     tol: float = 0.0) -> dict[str, float]:
    """Long-wavelength linear stability at equilibrium speed ``v_eq``.

    Returns a dict with the equilibrium gap, the partial derivatives, the
    stability margin ``m = f_s - f_dv f_v - f_v^2/2`` and a boolean ``stable``.
    """
    s_eq = model.equilibrium_gap(v_eq)
    if not np.isfinite(s_eq):
        nan = float("nan")
        return {"v_eq": v_eq, "s_eq": s_eq, "f_s": nan, "f_v": nan, "f_dv": nan,
                "margin": nan, "stable": False, "defined": False}
    fs, fv, fdv = partial_derivatives(model, s_eq, v_eq)
    m = fs - fdv * fv - fv ** 2 / 2.0
    return {"v_eq": v_eq, "s_eq": s_eq, "f_s": fs, "f_v": fv, "f_dv": fdv,
            "margin": m, "stable": bool(m < -tol), "defined": True}


def ring_eigenvalues(model: CarFollowingModel, n_vehicles: int, v_eq: float,
                     L: float = 5.0) -> np.ndarray:
    """Eigenvalues of the linearised ring dynamics (state = positions+speeds)."""
    from .equilibrium import ring_length_for

    s_eq = model.equilibrium_gap(v_eq)
    fs, fv, fdv = partial_derivatives(model, s_eq, v_eq)
    N = n_vehicles
    A = np.zeros((2 * N, 2 * N))
    # dx_i/dt = v_i
    for i in range(N):
        A[i, N + i] = 1.0
    # dv_i/dt = fs (x_{i+1} - x_i) + fv v_i + fdv (v_i - v_{i+1})
    # (vehicle i+1 is directly ahead of i)
    for i in range(N):
        j = (i + 1) % N
        A[N + i, j] = fs
        A[N + i, i] = -fs
        A[N + i, N + i] = fv + fdv
        A[N + i, N + j] = -fdv
    return np.linalg.eigvals(A)


def max_ring_growth(model: CarFollowingModel, n_vehicles: int, v_eq: float,
                    L: float = 5.0, neutral_tol: float = 1e-9) -> float:
    """Largest real part among the *non-neutral* linearised ring eigenvalues.

    The uniform-translation mode is an exact eigenvalue ``0`` and is excluded;
    the returned value is the growth rate of the most unstable relative mode.
    """
    ev = ring_eigenvalues(model, n_vehicles, v_eq, L=L)
    ev = ev[np.abs(ev) > neutral_tol]
    return float(np.max(ev.real)) if ev.size else 0.0


# ---------------------------------------------------------------------------
# String stability
# ---------------------------------------------------------------------------
def helly_string_stability(C1: float, C2: float, T: float,
                           omega: np.ndarray) -> np.ndarray:
    """|Gamma(i omega)| for the linear Helly model (leader -> follower speed).

    ``Gamma(s) = (C2 s + C1) / (s^2 + (C1 T + C2) s + C1)``.
    String stability holds iff ``|Gamma| <= 1`` for all ``omega``, which reduces
    to the closed-form condition ``C1 T^2 + 2 C2 T >= 2``.
    """
    omega = np.asarray(omega, dtype=float)
    s = 1j * omega
    g = (C2 * s + C1) / (s ** 2 + (C1 * T + C2) * s + C1)
    return np.abs(g)


def helly_string_stability_condition(C1: float, C2: float, T: float) -> bool:
    """Closed-form condition ``C1 T^2 + 2 C2 T >= 2``."""
    return (C1 * T ** 2 + 2.0 * C2 * T) >= 2.0


def numeric_string_stability(model: CarFollowingModel, omega: float, v_eq: float,
                             amplitude: float = 0.5, n_followers: int = 1,
                             L: float = 5.0, dt: float = 0.05,
                             t_end: float | None = None) -> float:
    """Measure the platoon amplification ratio ``|V_follower| / |V_leader|``.

    The leader is driven with ``v(t) = v_eq + amplitude * sin(omega t)``.  The
    follower chain is simulated and the steady-state amplitude at ``omega`` is
    recovered by a single-bin Fourier projection, giving ``Gamma(omega)``.
    """
    if t_end is None:
        t_end = max(400.0, 40.0 * 2 * np.pi / max(omega, 1e-3))
    n = int(round(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, n)
    v_lead_trace = v_eq + amplitude * np.sin(omega * t)
    trace = LeaderTrace.from_speed_profile(t, v_lead_trace)

    v = np.full(n_followers, v_eq)
    x = None
    res = None
    # chain the followers: each is a follower of the previous one
    leader = trace
    for _ in range(n_followers):
        res = simulate_following(model, leader, v0=v_eq, dt=dt, L=L, t_end=t_end)
        leader = LeaderTrace(t=res.t, x=res.x, v=res.v)
    follower_v = res.v

    # discard a transient (first half) and project onto sin/cos at omega
    n0 = len(t) // 2
    tt = t[n0:]
    vv = follower_v[n0:]
    vlead = v_lead_trace[n0:]
    proj = lambda y: np.sqrt((np.sum(y * np.sin(omega * tt)) ** 2 +  # noqa: E731
                              np.sum(y * np.cos(omega * tt)) ** 2)) * 2.0 / len(tt)
    amp_lead = proj(vlead - v_eq) + 1e-12
    amp_fol = proj(vv - v_eq)
    return float(amp_fol / amp_lead)
