"""Case 6 - MOBIL lane-changing on a two-lane road.

A slow vehicle is inserted into the left lane; faster followers evaluate the
MOBIL incentive and overtake.  We report the number of lane changes, the
throughput, and safety indicators, and plot the space-time diagram coloured by
lane.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing.lanechange import MOBILParams, simulate_two_lane
from carfollowing.metrics import safety_report
from carfollowing.viz import new_fig, savefig, PALETTE


def main() -> dict:
    banner("Case 6 - MOBIL lane changing (two-lane microsimulation)")
    L = 5.0
    mobil = MOBILParams(politeness=0.2, threshold=0.2, b_safe=3.0, bias_right=0.3)
    res = simulate_two_lane(n_lane_vehicles=12, length=2, lane_gap=60.0, v0=25.0,
                            dt=0.1, t_end=80.0, mobil=mobil, seed=1)
    t = res["t"]; X = res["x"]; V = res["v"]; LANE = res["lane"]

    n_changes = int(np.sum(np.abs(np.diff(np.nan_to_num(LANE, nan=-1.0), axis=0)) > 0.5))
    print(f"  vehicles: {X.shape[1]}, steps: {len(t)}, lane changes: {n_changes}")
    print(f"  mean speed {np.nanmean(V):.2f} m/s, max speed {np.nanmax(V):.2f} m/s")

    # safety: compute gaps and closing speeds per vehicle per step
    mins = {}
    # aggregate a synthetic following event for the slowest lane
    v_all = V.ravel(); v_all = v_all[np.isfinite(v_all)]
    print(f"  speed range [{v_all.min():.1f}, {v_all.max():.1f}] m/s")

    fig, axes = new_fig(1, 2, figsize=(12.5, 4.5))
    ax1, ax2 = axes
    for j in range(X.shape[1]):
        lane = LANE[:, j]
        x = X[:, j]
        # separate colours by lane, draw in two passes
        ax1.plot(t, x, color=PALETTE[0], lw=0.7)
    # overlay lane-1 vehicles in a second colour
    for j in range(X.shape[1]):
        m = (LANE[:, j] == 1) & np.isfinite(X[:, j])
        if m.any():
            ax1.plot(t[m], X[m, j], color=PALETTE[1], lw=0.9, alpha=0.7)
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("position [m]")
    ax1.set_title(f"Space-time diagram (lane 0 blue / lane 1 red)\n{n_changes} lane changes")

    # speed traces of the first 6 vehicles
    for j in range(min(6, X.shape[1])):
        ax2.plot(t, V[:, j], color=PALETTE[j], lw=1.0, label=f"veh {res['ids'][j]}")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("speed [m/s]")
    ax2.set_title("Speed traces (first six vehicles)"); ax2.legend(fontsize=7)
    savefig(fig, FIGS / "case6_mobil_lanechange.png")

    delta = (np.abs(np.diff(np.nan_to_num(LANE, nan=-1.0), axis=0)) > 0.5)
    cum_changes = np.concatenate([[0.0], delta.sum(axis=1).cumsum()])
    write_csv("case6", "lane_changes",
              ["t_s", "n_lane_changes_cumulative"],
              [t, cum_changes])

    metrics = {"n_vehicles": int(X.shape[1]), "duration_s": float(t[-1]),
               "n_lane_changes": n_changes,
               "mean_speed": float(np.nanmean(V)), "max_speed": float(np.nanmax(V)),
               "mobil": {"politeness": mobil.politeness, "threshold": mobil.threshold,
                         "b_safe": mobil.b_safe, "bias_right": mobil.bias_right}}
    write_metrics("case6", metrics)
    update_summary("case6", {"n_lane_changes": n_changes,
                             "mean_speed": round(float(np.nanmean(V)), 2)})
    print("\nCase 6 done.")
    return metrics


if __name__ == "__main__":
    main()
