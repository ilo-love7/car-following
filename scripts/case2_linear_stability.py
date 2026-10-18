"""Case 2 — linear (long-wavelength) stability and its numerical validation.

The analytic stability margin is ``m = f_s - f_dv f_v - f_v^2/2`` (stable iff
``m < 0``).  Here we

* map ``m`` over the ``(time-headway T, equilibrium speed v)`` plane for IDM,
* compare its sign with the exact growth rate of the linearised ring dynamics,
* plot ``m(v)`` for every model,
and report the fraction of the grid where the two notions agree.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing import make_model, IDM
from carfollowing.stability import linear_stability, max_ring_growth
from carfollowing.viz import new_fig, savefig, PALETTE

MODELS = ["IDM", "OVM", "FVD", "Helly"]
N_RING = 120


def main() -> dict:
    banner("Case 2 - Linear stability vs ring eigenvalue growth")

    # ---- (T, v) grid for IDM -------------------------------------------
    # v stays strictly below the IDM desired speed v0=30 (no equilibrium above it)
    Ts = np.linspace(0.4, 3.0, 40)
    vs = np.linspace(4.0, 28.0, 40)
    margin = np.zeros((len(Ts), len(vs)))
    growth = np.zeros_like(margin)
    s_eq = np.zeros_like(margin)
    for i, T in enumerate(Ts):
        m = IDM(T=float(T))
        for j, v in enumerate(vs):
            ls = linear_stability(m, float(v))
            margin[i, j] = ls["margin"]
            s_eq[i, j] = ls["s_eq"]
            growth[i, j] = max_ring_growth(m, N_RING, float(v))

    agree = np.mean((margin < 0) == (growth < 0))
    n_unstable = int(np.sum(margin > 0))
    print(f"grid {margin.shape}, analytic-unstable cells: {n_unstable} "
          f"({100*n_unstable/margin.size:.1f}%)")
    print(f"sign agreement analytic vs ring growth: {100*agree:.2f}%")

    write_csv("case2", "idm_stability_grid",
              ["T_s", "v_ms", "margin", "s_eq_m", "ring_growth"],
              [np.repeat(Ts, len(vs)), np.tile(vs, len(Ts)),
               margin.ravel(), s_eq.ravel(), growth.ravel()])

    # ---- margin(v) for each model --------------------------------------
    vfine = np.linspace(4.0, 31.0, 120)
    fig, axes = new_fig(1, 3, figsize=(15, 4.0))
    ax0, ax1, ax2 = axes
    imm = ax0.imshow(margin, origin="lower", aspect="auto", cmap="RdBu_r",
                     vmin=-np.abs(margin).max(), vmax=np.abs(margin).max(),
                     extent=[vs[0], vs[-1], Ts[0], Ts[-1]])
    ax0.contour(vs, Ts, margin, levels=[0.0], colors="k", linewidths=1.5)
    ax0.set_xlabel("equilibrium speed [m/s]"); ax0.set_ylabel("time headway T [s]")
    ax0.set_title("IDM stability margin m\n(blue=m<0 stable, red=m>0 unstable)")
    fig.colorbar(imm, ax=ax0, shrink=0.85)

    imm2 = ax1.imshow(growth, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-np.abs(growth).max(), vmax=np.abs(growth).max(),
                      extent=[vs[0], vs[-1], Ts[0], Ts[-1]])
    ax1.contour(vs, Ts, growth, levels=[0.0], colors="k", linewidths=1.5)
    ax1.set_xlabel("equilibrium speed [m/s]"); ax1.set_ylabel("time headway T [s]")
    ax1.set_title(f"Ring growth rate (N={N_RING})\nblack contour = stability boundary")
    fig.colorbar(imm2, ax=ax1, shrink=0.85)

    for i, name in enumerate(MODELS):
        m = make_model(name)
        vals = np.array([linear_stability(m, float(v))["margin"] for v in vfine])
        vals = np.where(np.abs(vals) < 1e6, vals, np.nan)
        ax2.plot(vfine, vals, color=PALETTE[i], label=name)
    ax2.axhline(0.0, color="k", lw=1)
    ax2.set_xlabel("equilibrium speed [m/s]"); ax2.set_ylabel("stability margin m")
    ax2.set_title("Stability margin vs speed"); ax2.legend()
    savefig(fig, FIGS / "case2_linear_stability.png")

    # ---- boundary speed per model (where m crosses 0) -------------------
    boundary = {}
    for name in MODELS:
        m = make_model(name)
        vals = np.array([linear_stability(m, float(v))["margin"] for v in vfine])
        finite = np.isfinite(vals)
        sign = np.sign(np.nan_to_num(vals, nan=-1.0))
        d = np.diff(sign)
        cross = np.where((d != 0) & finite[:-1] & finite[1:])[0]
        boundary[name] = float(vfine[cross[0]]) if len(cross) else None

    print(f"\n{'model':8s} {'stability margin @ 15 m/s':>26s} {'@ 25 m/s':>12s}")
    for name in MODELS:
        m = make_model(name)
        print(f"{name:8s} {linear_stability(m,15.0)['margin']:26.5f} "
              f"{linear_stability(m,25.0)['margin']:12.5f}")

    metrics = {
        "grid_shape": list(margin.shape),
        "n_ring": N_RING,
        "fraction_analytic_unstable": float(n_unstable / margin.size),
        "sign_agreement": float(agree),
        "boundary_speed_ms": boundary,
        "idm_margin_at": {str(v): float(linear_stability(IDM(), v)["margin"]) for v in (10, 15, 20, 25, 28)},
    }
    write_metrics("case2", metrics)
    update_summary("case2", {"sign_agreement": round(float(agree), 4),
                             "analytic_unstable_fraction": round(float(n_unstable / margin.size), 3)})
    print("\nCase 2 done.")
    return metrics


if __name__ == "__main__":
    main()
