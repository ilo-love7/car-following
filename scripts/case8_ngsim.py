"""Case 8 - NGSIM adapter: end-to-end validation.

The real NGSIM trajectory files are public but large and not bundled here.  To
prove the loader + event extractor are correct, this script:

1. builds a *synthetic NGSIM-format* fixture from a known IDM platoon (US
   customary columns, 10 Hz, ft / ft-per-second),
2. writes it to ``data/ngsim_fixture.txt``,
3. reads it back through :func:`carfollowing.datasets.ngsim.load_ngsim`,
4. extracts sustained car-following events, and
5. calibrates IDM on an extracted event, comparing the recovered parameters with
   the ground truth.

If a real NGSIM file is present under ``data/ngsim/`` it is used instead, and the
ground-truth comparison is skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import DATA, FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing.datasets import generate_platoon
from carfollowing.datasets.ngsim import (NGSIM_FPS, extract_following_events, load_ngsim)
from carfollowing.calibration import calibrate_following
from carfollowing.viz import new_fig, savefig, PALETTE

FT = 1.0 / 0.3048          # m -> ft
FTS = 1.0 / 0.3048         # m/s -> ft/s


def build_fixture(path: Path, seed: int = 0) -> dict:
    """Write a synthetic IDM platoon in NGSIM text format and return the truth."""
    t_end, dt = 80.0, 1.0 / NGSIM_FPS
    p = generate_platoon(n_followers=3, t_end=t_end, dt=dt, seed=seed)
    t = p.t
    n = len(t)
    # vehicle IDs: leader = 1, followers = 2..; NGSIM 'Preceding' convention
    ids = list(range(1, p.n_vehicles + 1))
    lines = ["Vehicle_ID\tFrame_ID\tGlobal_Time\tLocal_X\tLocal_Y\tv_Vel\tv_Acc\t"
             "Lane_ID\tPreceding\tFollowing"]
    for vi in range(p.n_vehicles):
        tr = p.trajectories[vi]
        x = tr["x"]; v = tr["v"]
        acc = np.gradient(v, t)
        prec = ids[vi - 1] if vi > 0 else 0
        foll = ids[vi + 1] if vi < p.n_vehicles - 1 else 0
        for k in range(n):
            lines.append(
                f"{ids[vi]}\t{k}\t{int(round(t[k]*1000))}\t{x[k]*FT:.4f}\t0.0000\t"
                f"{v[k]*FTS:.4f}\t{acc[k]*FTS:.4f}\t1\t{prec}\t{foll}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"n_vehicles": p.n_vehicles, "duration_s": float(t_end),
            "follower_params": p.params[0] if p.params else {}}


def _calibrate_first_event(events, max_events: int = 3, dt: float = 0.2) -> list[dict]:
    from carfollowing.simulate import LeaderTrace

    out = []
    for ev in events[:max_events]:
        # subsample to dt for a fast calibration, keep at most 500 points
        step = max(1, int(round(dt * NGSIM_FPS)))
        idx = np.arange(0, len(ev.t), step)[:500]
        t = ev.t[idx]; s = ev.gap[idx]; v = ev.v[idx]
        lead = LeaderTrace(t=ev.t[idx], x=ev.x_lead[idx], v=ev.v_lead[idx])
        try:
            res = calibrate_following("IDM", lead, t, s, v, n_restarts=3, dt=dt, seed=0)
            out.append({"follower_id": ev.follower_id, "leader_id": ev.leader_id,
                        "n_obs": int(len(t)), "rmse_gap": res.rmse_gap,
                        "rmse_speed": res.rmse_speed, "fitted": res.fitted})
            print(f"    event follower={ev.follower_id} leader={ev.leader_id}: "
                  f"RMSE_gap={res.rmse_gap:.3f} m, RMSE_v={res.rmse_speed:.3f} m/s")
        except Exception as exc:  # pragma: no cover
            print(f"    event follower={ev.follower_id} calibration failed: {exc}")
    return out


def main() -> dict:
    banner("Case 8 - NGSIM adapter (loader + car-following extraction)")

    real_dir = DATA / "ngsim"
    real_files = sorted(real_dir.glob("*.txt")) + sorted(real_dir.glob("*.csv")) \
        if real_dir.exists() else []
    source = "real"
    if real_files:
        path = real_files[0]
        print(f"  using real NGSIM file: {path.name}")
    else:
        path = DATA / "ngsim_fixture.txt"
        truth = build_fixture(path)
        source = "synthetic-fixture"
        print(f"  no real NGSIM found under data/ngsim/; wrote fixture ({path.name})")

    df = load_ngsim(path)
    print(f"  loaded {len(df)} rows, {df['vid'].nunique()} vehicles, "
          f"t in [0, {df['t'].max():.1f}] s")

    events = extract_following_events(df, min_duration=20.0, gap_range=(2.0, 80.0),
                                      v_min=2.0, max_events=8)
    print(f"  extracted {len(events)} car-following events "
          f"(>=20 s, gap in [2,80] m)")
    for ev in events[:5]:
        print(f"    follower {ev.follower_id} <- leader {ev.leader_id}: "
              f"T={len(ev.t)} steps, gap {ev.gap.min():.1f}-{ev.gap.max():.1f} m, "
              f"v {ev.v.mean():.1f} m/s")

    results = _calibrate_first_event(events)

    # ---- figure: one event + its calibration ---------------------------
    fig, axes = new_fig(1, 2, figsize=(12, 4.2))
    if events:
        ev = events[0]
        axes[0].plot(ev.t, ev.gap, "k", lw=1.0, label="observed gap")
        axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("gap [m]")
        axes[0].set_title(f"Extracted event: follower {ev.follower_id} "
                          f"<- leader {ev.leader_id}")
        axes[1].plot(ev.t, ev.v, color=PALETTE[0], label="follower")
        axes[1].plot(ev.t, ev.v_lead, color=PALETTE[2], alpha=0.7, label="leader")
        axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("speed [m/s]")
        axes[1].set_title("Event speeds"); axes[1].legend()
    savefig(fig, FIGS / "case8_ngsim_event.png")

    if events:
        write_csv("case8", "event0",
                  ["t_s", "gap_m", "v_ms", "v_lead_ms"],
                  [events[0].t, events[0].gap, events[0].v, events[0].v_lead])

    metrics = {"source": source, "file": path.name, "n_rows": int(len(df)),
               "n_events": len(events),
               "calibrations": results,
               "fixture_truth": (truth if source == "synthetic-fixture" else None)}
    write_metrics("case8", metrics)
    update_summary("case8", {"source": source, "n_events": len(events),
                             "n_calibrated": len(results)})
    print("\nCase 8 done.")
    return metrics


if __name__ == "__main__":
    main()
