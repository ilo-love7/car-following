"""Case 7 - electric-vehicle energy consumption from trajectories.

Three driving styles (eco / normal / aggressive) are generated as speed traces,
and the longitudinal EV energy model converts each into battery power and
cumulative energy.  We also compute the U-shaped constant-speed consumption
curve and quantify the energy penalty of an unstable (oscillatory) car-following
stream versus a smooth one.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing.energy import EVParams, energy_from_speed, compare_styles
from carfollowing.viz import new_fig, savefig, PALETTE


def _style_profile(t: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """A 60 s urban/highway trace whose 'aggressiveness' varies."""
    rng = np.random.default_rng(seed)
    phases = {
        "eco":        dict(base=16.0, osc=1.5, period=40.0, events=1, depth=3.0),
        "normal":     dict(base=18.0, osc=3.5, period=25.0, events=2, depth=6.0),
        "aggressive": dict(base=20.0, osc=6.0, period=14.0, events=4, depth=11.0),
    }
    p = phases[kind]
    v = p["base"] + p["osc"] * np.sin(2 * np.pi * t / p["period"])
    for _ in range(p["events"]):
        t0 = rng.uniform(0.1, 0.9) * t[-1]
        dur = rng.uniform(3.0, 7.0)
        v = v - p["depth"] * np.exp(-0.5 * ((t - t0) / (dur / 2.0)) ** 2)
    return np.clip(v, 2.0, 35.0)


def main() -> dict:
    banner("Case 7 - EV energy consumption across driving styles")
    p = EVParams(mass=1600.0, Cd=0.29, area=2.3, Crr=0.010,
                 eta_drive=0.90, eta_regen=0.65, aux_power=300.0)
    t = np.arange(0.0, 60.0, 0.1)
    styles = {k: _style_profile(t, k) for k in ("eco", "normal", "aggressive")}

    fig, axes = new_fig(1, 3, figsize=(15, 4.2))
    table = {}
    for i, (name, v) in enumerate(styles.items()):
        r = energy_from_speed(t, v, p)
        summary = compare_styles(t, v, p)
        table[name] = summary
        axes[0].plot(t, v, color=PALETTE[i], label=name)
        axes[1].plot(t, r["P_batt"] / 1000.0, color=PALETTE[i], label=name)
        axes[2].plot(r["distance"] / 1000.0, r["E"] / 3.6e6, color=PALETTE[i], label=name)
        print(f"  {name:11s}: {summary['distance_km']:5.2f} km  "
              f"{summary['energy_kWh']:6.3f} kWh  {summary['kWh_per_100km']:6.2f} kWh/100km")

    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("speed [m/s]")
    axes[0].set_title("Driving-style speed traces"); axes[0].legend()
    axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("battery power [kW]")
    axes[1].set_title("Battery power (sign = traction/charge)"); axes[1].legend()
    axes[2].set_xlabel("distance [km]"); axes[2].set_ylabel("cumulative energy [kWh]")
    axes[2].set_title("Cumulative battery energy"); axes[2].legend()
    savefig(fig, FIGS / "case7_energy_styles.png")

    # ---- constant-speed consumption curve (U-shape) --------------------
    speeds = np.linspace(5.0, 40.0, 60)
    cons = []
    for v in speeds:
        tt = np.arange(0.0, 100.0, 0.1)
        vv = np.full_like(tt, v)
        cons.append(compare_styles(tt, vv, p)["kWh_per_100km"])
    cons = np.array(cons)
    i_min = int(np.argmin(cons))
    fig2, ax = new_fig(1, 1, figsize=(6, 3.8))
    ax.plot(speeds * 3.6, cons, color=PALETTE[0], lw=1.6)
    ax.plot([speeds[i_min] * 3.6], [cons[i_min]], "o", color=PALETTE[1])
    ax.set_xlabel("constant speed [km/h]"); ax.set_ylabel("consumption [kWh/100km]")
    ax.set_title(f"Constant-speed consumption\\nminimum at {speeds[i_min]*3.6:.1f} km/h "
                 f"({cons[i_min]:.1f} kWh/100km)")
    savefig(fig2, FIGS / "case7_consumption_curve.png")

    # ---- congestion penalty via ring-road stop-and-go ------------------
    # Each stream starts uniform at the same target speed, on its own
    # equilibrium ring (same mean speed, different headway).  The
    # short-headway stream destabilises into stop-and-go waves.
    from carfollowing.models import IDM
    from carfollowing.equilibrium import ring_length_for
    from carfollowing.simulate import simulate_ring

    N, Lring, v_eq = 30, 5.0, 12.0
    for label, T in (("smooth stream (T=1.8)", 1.8),
                     ("stop-and-go (T=0.5)", 0.5)):
        m = IDM(T=T)
        rl = ring_length_for(v_eq, N, m, Lring)
        spacing = rl / N
        rng = np.random.default_rng(11)
        x0 = np.arange(N) * spacing + rng.normal(0.0, 0.3, N)
        v0 = np.full(N, v_eq) + rng.normal(0.0, 0.05, N)
        res = simulate_ring(m, N, rl, v_eq, dt=0.1, t_end=400.0, L=Lring,
                            x0=x0, v0=v0)
        tt = res["t"]
        sel = tt >= 200.0                      # developed state, last 200 s
        ts = tt[sel]
        energ = [compare_styles(ts, res["v"][sel, j], p)["kWh_per_100km"]
                 for j in range(N)]
        vstd = float(np.mean(res["v"][sel].std(axis=1)))
        table[label] = {"kWh_per_100km": float(np.mean(energ)),
                        "speed_std_ms": vstd,
                        "v_mean": float(np.mean(res["v"][sel]))}
        print(f"  {label:22s}: {np.mean(energ):6.2f} kWh/100km  "
              f"speed std {vstd:4.2f} m/s  (mean speed {np.mean(res['v'][sel]):4.1f} m/s)")

    # a small dedicated figure for the stop-and-go penalty
    fig3, ax = new_fig(1, 2, figsize=(12, 4.0))
    labels = [k for k in table if "speed_std_ms" in table[k]]
    vals = [table[k]["kWh_per_100km"] for k in labels]
    stds = [table[k]["speed_std_ms"] for k in labels]
    ax[0].bar(range(len(labels)), vals, color=[PALETTE[0], PALETTE[1]])
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("energy [kWh/100km]"); ax[0].set_title("Energy: stable vs stop-and-go")
    ax[1].bar(range(len(labels)), stds, color=[PALETTE[0], PALETTE[1]])
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("mean speed std [m/s]"); ax[1].set_title("Speed variability")
    savefig(fig3, FIGS / "case7_congestion_penalty.png")

    write_csv("case7", "styles",
              ["t_s", "eco_v", "normal_v", "aggressive_v"],
              [t, styles["eco"], styles["normal"], styles["aggressive"]])
    write_csv("case7", "consumption_curve",
              ["speed_kmh", "kwh_per_100km"], [speeds * 3.6, cons])

    metrics = {"EV_params": p.as_dict(), "styles": table,
               "constant_speed_min": {"speed_kmh": float(speeds[i_min] * 3.6),
                                      "kwh_per_100km": float(cons[i_min])}}
    write_metrics("case7", metrics)
    update_summary("case7", {
        "kwh_per_100km": {k: round(v["kWh_per_100km"], 2) for k, v in table.items()
                          if "kWh_per_100km" in v},
        "congestion_speed_std_ms": {k: round(v["speed_std_ms"], 2) for k, v in table.items()
                                    if "speed_std_ms" in v}})
    print("\nCase 7 done.")
    return metrics


if __name__ == "__main__":
    main()
