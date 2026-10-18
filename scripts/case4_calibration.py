"""Case 4 - parameter calibration on the synthetic benchmark.

Part A: recover the (known) IDM parameters of a follower from its simulated
trajectory - a ground-truth identifiability test free of data noise.

Part B: model selection - calibrate five models to the same follower and rank by
gap/speed RMSE (reporting the number of free parameters).

Part C: noise robustness - inject increasing measurement noise and track the
parameter-recovery error.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing.datasets import generate_platoon
from carfollowing.calibration import calibrate_following
from carfollowing.simulate import LeaderTrace
from carfollowing.viz import new_fig, savefig, PALETTE

MODELS = ["IDM", "OVM", "FVD", "Helly", "Newell"]
IDM_FIT = ["v0", "T", "s0", "a", "b"]


def _data(seed: int = 0, noise: float = 0.0, t_end: float = 120.0, dt: float = 0.2):
    p = generate_platoon(n_followers=3, t_end=t_end, dt=dt, seed=seed, noise=noise)
    lead, t, s, v = p.pair(2)
    return lead, t, s, v, p.params[1]


def part_a() -> dict:
    banner("Case 4A - IDM parameter recovery (noise-free)")
    lead, t, s, v, true = _data(seed=0, noise=0.0)
    res = calibrate_following("IDM", lead, t, s, v, fit_params=IDM_FIT,
                              n_restarts=4, dt=0.2, seed=0)
    print(f"  RMSE gap {res.rmse_gap:.3f} m | RMSE speed {res.rmse_speed:.3f} m/s")
    print(f"  {'param':6s} {'true':>8s} {'fitted':>8s} {'err %':>8s}")
    errpct = {}
    for k in IDM_FIT:
        e = 100.0 * (res.fitted[k] - true[k]) / true[k]
        errpct[k] = float(e)
        print(f"  {k:6s} {true[k]:8.3f} {res.fitted[k]:8.3f} {e:8.1f}")
    return {"rmse_gap": res.rmse_gap, "rmse_speed": res.rmse_speed,
            "true": true, "fitted": res.fitted, "param_err_pct": errpct,
            "leader": lead, "t": t, "s": s, "v": v}


def part_b() -> dict:
    banner("Case 4B - model selection on the same follower")
    lead, t, s, v, true = _data(seed=0, noise=0.0)
    rows = []
    for name in MODELS:
        try:
            res = calibrate_following(name, lead, t, s, v, n_restarts=2, dt=0.2, seed=1)
            rows.append((name, len(res.fitted), res.rmse_gap, res.rmse_speed))
            print(f"  {name:8s} npar={len(res.fitted)}  RMSE_gap={res.rmse_gap:7.3f} m  "
                  f"RMSE_v={res.rmse_speed:6.3f} m/s")
        except Exception as exc:  # pragma: no cover
            print(f"  {name:8s} calibration failed: {exc}")
    rows.sort(key=lambda r: r[2])
    return {"table": [{"model": r[0], "n_params": r[1], "rmse_gap": r[2],
                       "rmse_speed": r[3]} for r in rows]}


def part_c() -> dict:
    banner("Case 4C - noise robustness of IDM recovery")
    noise_levels = [0.0, 0.1, 0.3, 0.6]
    table = []
    for nl in noise_levels:
        lead, t, s, v, true = _data(seed=0, noise=nl)
        # add noise to the leader trace as well (radar/GPS measurement)
        rng = np.random.default_rng(int(nl * 1000) + 5)
        lead_noisy = LeaderTrace(t=lead.t, x=lead.x + rng.normal(0, nl, lead.x.shape),
                                 v=lead.v + rng.normal(0, nl / 0.2 * 0.05, lead.v.shape))
        res = calibrate_following("IDM", lead_noisy, t, s, v, fit_params=IDM_FIT,
                                  n_restarts=3, dt=0.2, seed=2)
        err = float(np.mean([abs(res.fitted[k] - true[k]) / abs(true[k]) for k in IDM_FIT]))
        table.append({"noise_m": nl, "rmse_gap": res.rmse_gap, "mean_param_err": err})
        print(f"  noise={nl:4.1f} m  RMSE_gap={res.rmse_gap:6.3f} m  "
              f"mean param err={100*err:5.1f}%")
    return {"table": table}


def main() -> dict:
    a = part_a()
    b = part_b()
    c = part_c()

    # ---- figures --------------------------------------------------------
    fig, axes = new_fig(1, 3, figsize=(15, 4.0))
    # (1) fit quality from 4A
    lead, t, s, v = a["leader"], a["t"], a["s"], a["v"]
    from carfollowing.models import IDM as _IDM
    from carfollowing.simulate import simulate_following
    model = _IDM(**a["fitted"], delta=4.0)
    sim = simulate_following(model, lead, v0=float(v[0]), s0=float(s[0]),
                             dt=0.2, t_end=float(t[-1]))
    axes[0].plot(t, s, color="k", lw=1.2, label="observed")
    axes[0].plot(sim.t, sim.s, color=PALETTE[1], ls="--", label="calibrated IDM")
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("gap [m]")
    axes[0].set_title("Gap fit (case 4A)"); axes[0].legend()

    axes[1].plot(t, v, color="k", lw=1.2, label="observed")
    axes[1].plot(sim.t, sim.v, color=PALETTE[1], ls="--", label="calibrated IDM")
    axes[1].plot(lead.t, lead.v, color=PALETTE[2], alpha=0.6, label="leader")
    axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("speed [m/s]")
    axes[1].set_title("Speed fit (case 4A)"); axes[1].legend()

    # (2) model ranking
    names = [r["model"] for r in b["table"]]
    rmse = [r["rmse_gap"] for r in b["table"]]
    axes[2].bar(range(len(names)), rmse, color=[PALETTE[i % len(PALETTE)] for i in range(len(names))])
    axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names)
    axes[2].set_ylabel("gap RMSE [m]"); axes[2].set_title("Model selection (case 4B)")
    savefig(fig, FIGS / "case4_calibration.png")

    # noise figure
    fig2, ax = new_fig(1, 1, figsize=(5.5, 3.6))
    nl = [r["noise_m"] for r in c["table"]]
    mp = [100 * r["mean_param_err"] for r in c["table"]]
    ax.plot(nl, mp, "o-", color=PALETTE[0])
    ax.set_xlabel("measurement noise [m]"); ax.set_ylabel("mean |param error| [%]")
    ax.set_title("IDM recovery error vs noise (case 4C)")
    savefig(fig2, FIGS / "case4_noise_robustness.png")

    write_csv("case4", "model_selection",
              ["model", "n_params", "rmse_gap", "rmse_speed"],
              [[r["model"] for r in b["table"]],
               [r["n_params"] for r in b["table"]],
               [r["rmse_gap"] for r in b["table"]],
               [r["rmse_speed"] for r in b["table"]]])
    metrics = {"A_param_recovery": {"true": a["true"], "fitted": a["fitted"],
                                    "err_pct": a["param_err_pct"],
                                    "rmse_gap": a["rmse_gap"], "rmse_speed": a["rmse_speed"]},
               "B_model_selection": b["table"], "C_noise": c["table"]}
    write_metrics("case4", metrics)
    update_summary("case4", {"A_rmse_gap_m": round(a["rmse_gap"], 3),
                             "A_max_param_err_pct": round(max(abs(v) for v in a["param_err_pct"].values()), 1),
                             "B_best_model": b["table"][0]["model"],
                             "C_noise_table": c["table"]})
    print("\nCase 4 done.")
    return metrics


if __name__ == "__main__":
    main()
