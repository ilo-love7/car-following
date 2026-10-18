"""Case 3 - ring-road dynamics and string stability.

Part A: simulate an ``N``-vehicle ring road with a small perturbation for a
linearly *stable* and an *unstable* IDM parameter set; compare the observed
growth rate of the speed variance against the linearised ring prediction.

Part B: string stability of the Helly/ACC model - the analytic transfer
function ``|Gamma(i omega)|`` versus the numerically measured amplification of a
sinusoidal leader, for both a string-stable and a string-unstable gain set.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing import IDM, Helly
from carfollowing.equilibrium import ring_length_for
from carfollowing.simulate import simulate_ring
from carfollowing.stability import (max_ring_growth, helly_string_stability,
                            helly_string_stability_condition, numeric_string_stability)
from carfollowing.viz import new_fig, savefig, PALETTE


def _ring_std(res: dict) -> np.ndarray:
    return res["v"].std(axis=1)


def part_a() -> dict:
    banner("Case 3A - ring road: damping vs growth of a perturbation")
    N, L = 40, 5.0
    cases = {
        "stable":   dict(model=IDM(T=1.8, v0=30.0), v_eq=18.0, t_end=120.0, fit=(0.10, 0.40)),
        "unstable": dict(model=IDM(T=0.4, v0=30.0), v_eq=6.0,  t_end=120.0, fit=(0.10, 0.55)),
    }
    rng = np.random.default_rng(7)
    fig, axes = new_fig(2, 2, figsize=(12, 7))
    out = {}
    for col, (name, cfg) in enumerate(cases.items()):
        m = cfg["model"]; v_eq = cfg["v_eq"]
        ring_len = ring_length_for(v_eq, N, m, L)
        spacing = ring_len / N
        x0 = np.arange(N) * spacing + rng.normal(0.0, 0.3, N)
        v0 = np.full(N, v_eq) + rng.normal(0.0, 0.08, N)
        res = simulate_ring(m, N, ring_len, v_eq, dt=0.05, t_end=cfg["t_end"],
                            L=L, x0=x0, v0=v0)
        t = res["t"]; sigma = _ring_std(res)
        pred = max_ring_growth(m, N, v_eq, L=L)

        f0, f1 = cfg["fit"]
        i0, i1 = int(f0 * len(t)), int(f1 * len(t))
        mask = sigma[i0:i1] > 1e-7
        rate = float(np.polyfit(t[i0:i1][mask], np.log(sigma[i0:i1][mask]), 1)[0]) \
            if mask.sum() > 10 else float("nan")

        ax1, ax2 = axes[0, col], axes[1, col]
        ax1.semilogy(t, sigma, color=PALETTE[col])
        ax1.set_xlabel("time [s]"); ax1.set_ylabel("speed std [m/s]")
        ax1.set_title(f"{name}: T={m.T} s, v={v_eq} m/s\n"
                      f"predicted growth {pred:+.4f} 1/s, fitted {rate:+.4f} 1/s")
        ax2.plot(res["x"][:, :12] % ring_len, color=PALETTE[col], lw=0.8)
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("position mod L [m]")
        ax2.set_title("trajectories (12 of 40 vehicles)")
        out[name] = {"predicted_growth": pred, "fitted_growth": rate,
                     "sigma_initial": float(sigma[0]), "sigma_final": float(sigma[-1])}
        print(f"  {name:9s}: predicted {pred:+.5f} 1/s | fitted {rate:+.5f} 1/s | "
              f"sigma {sigma[0]:.3g} -> {sigma[-1]:.3g}")
    savefig(fig, FIGS / "case3_ring_dynamics.png")
    return out


def part_b() -> dict:
    banner("Case 3B - string stability (Helly/ACC): analytic vs numeric")
    # two gain sets: one string-stable, one not
    sets = {
        "stable":   dict(C1=0.4, C2=0.9, T=1.2),
        "unstable": dict(C1=0.2, C2=0.2, T=1.2),
    }
    omegas = np.array([0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0, 1.6, 2.5, 4.0])
    fig, axes = new_fig(1, 2, figsize=(12, 4.2))
    ax1, ax2 = axes
    table = {}
    for i, (name, g) in enumerate(sets.items()):
        model = Helly(**g)
        cond = helly_string_stability_condition(g["C1"], g["C2"], g["T"])
        ana = helly_string_stability(g["C1"], g["C2"], g["T"], omegas)
        num = np.array([numeric_string_stability(model, float(w), v_eq=20.0,
                                                  amplitude=0.4, dt=0.05,
                                                  t_end=max(300.0, 25 * 2 * np.pi / w))
                        for w in omegas])
        ax1.plot(omegas, ana, "-", color=PALETTE[i], label=f"{name} (analytic)")
        ax1.plot(omegas, num, "o", color=PALETTE[i], ms=4, mfc="w",
                 label=f"{name} (numeric)")
        table[name] = {"analytic": ana.tolist(), "numeric": num.tolist(),
                       "condition_satisfied": bool(cond)}
        print(f"  {name:9s}: closed-form condition satisfied = {cond}; "
              f"max|Gamma| analytic {ana.max():.3f}, numeric {num.max():.3f}")

    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.set_xscale("log"); ax1.set_ylim(0, max(2.0, ax1.get_ylim()[1]))
    ax1.set_xlabel("frequency omega [rad/s]"); ax1.set_ylabel("amplification |Gamma|")
    ax1.set_title("String stability: leader -> follower speed gain")
    ax1.legend(fontsize=8)

    # (C2, C1) plane showing the analytic boundary C1 T^2 + 2 C2 T = 2
    T = 1.2
    C1 = np.linspace(0.05, 1.0, 200)
    C2 = np.linspace(0.05, 1.5, 200)
    C1g, C2g = np.meshgrid(C1, C2)
    metric = C1g * T ** 2 + 2 * C2g * T
    cs = ax2.contourf(C1g, C2g, metric, levels=[0, 2, 10], colors=["#f7c6c6", "#c7e6c7"])
    ax2.contour(C1g, C2g, metric, levels=[2.0], colors="k", linewidths=1.5)
    for name, g in sets.items():
        ok = helly_string_stability_condition(g["C1"], g["C2"], g["T"])
        ax2.plot([g["C1"]], [g["C2"]], "o", color="#d62728" if not ok else "#2ca02c", ms=8)
        ax2.annotate(name, (g["C1"], g["C2"]), textcoords="offset points", xytext=(6, 6))
    ax2.set_xlabel("gap gain C1 [1/s^2]"); ax2.set_ylabel("speed gain C2 [1/s]")
    ax2.set_title(f"String-stability boundary (T={T} s)\ngreen = C1 T^2 + 2 C2 T >= 2")
    savefig(fig, FIGS / "case3_string_stability.png")
    return table


def main() -> dict:
    a = part_a()
    b = part_b()
    metrics = {"ring": a, "string_stability": b}
    write_metrics("case3", metrics)
    update_summary("case3", {
        "ring_fitted_growth": {k: round(v["fitted_growth"], 4) for k, v in a.items()},
        "ring_predicted_growth": {k: round(v["predicted_growth"], 4) for k, v in a.items()},
        "helly_string_stable": {k: v["condition_satisfied"] for k, v in b.items()},
    })
    print("\nCase 3 done.")
    return metrics


if __name__ == "__main__":
    main()
