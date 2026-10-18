"""Case 1 — equilibrium relations and the microscopic fundamental diagram.

Builds the speed-gap equilibrium curve and the flow-density (fundamental)
diagram of five models, reports capacity points, and writes figures + CSVs.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing import IDM, OVM, FVD, Helly, Newell, make_model
from carfollowing.equilibrium import fundamental_diagram, capacity_point
from carfollowing.viz import new_fig, savefig, PALETTE

MODELS = ["IDM", "OVM", "FVD", "Helly", "Newell"]
L = 5.0


def main() -> dict:
    banner("Case 1 · Equilibrium & microscopic fundamental diagram")

    curves = {}
    caps = {}
    # 内容签名去重：FVD 在本库继承 OVM 的稳态关系（s_e(v) 逐字节相同），
    # 二者基本图完全等价，仅保留先写者，避免重复数据产物。
    written_sig: dict[str, str] = {}
    for name in MODELS:
        m = make_model(name)
        fd = fundamental_diagram(m, L=L)
        curves[name] = fd
        caps[name] = capacity_point(m, L=L)
        sig = "|".join(
            np.asarray(fd[k]).tobytes().hex() for k in
            ("v", "gap", "headway", "density", "flow")
        )
        if sig in written_sig:
            print(f"  skip case1/fd_{name}.csv — identical to fd_{written_sig[sig]}.csv")
            continue
        written_sig[sig] = name
        write_csv("case1", f"fd_{name}",
                  ["v_ms", "gap_m", "headway_m", "density_veh_km", "flow_veh_h"],
                  [fd["v"], fd["gap"], fd["headway"], fd["density"], fd["flow"]])

    # ---- figure: speed-gap + flow-density -------------------------------
    fig, axes = new_fig(1, 2, figsize=(11, 4.2))
    ax1, ax2 = axes
    for i, name in enumerate(MODELS):
        fd = curves[name]
        ax1.plot(fd["gap"], fd["v"], color=PALETTE[i], label=name)
        ax2.plot(fd["density"], fd["flow"], color=PALETTE[i], label=name)
        cp = caps[name]
        ax2.plot([cp["rho_crit"]], [cp["capacity"]], "o", color=PALETTE[i], ms=4)
    ax1.set_xlabel("equilibrium gap [m]"); ax1.set_ylabel("speed [m/s]")
    ax1.set_title("Equilibrium speed-gap relation"); ax1.legend()
    ax2.set_xlabel("density [veh/km]"); ax2.set_ylabel("flow [veh/h]")
    ax2.set_title("Fundamental diagram"); ax2.legend()
    savefig(fig, FIGS / "case1_fundamental_diagram.png")

    # ---- table ----------------------------------------------------------
    print(f"\n{'model':8s} {'v_free':>7s} {'capacity':>9s} {'v_crit':>7s} "
          f"{'rho_crit':>8s} {'rho_jam':>8s}")
    for name in MODELS:
        c = caps[name]
        print(f"{name:8s} {c['v_free']:7.2f} {c['capacity']:9.1f} {c['v_crit']:7.2f} "
              f"{c['rho_crit']:8.1f} {c['rho_jam']:8.1f}")

    metrics = {"L_vehicle": L,
               "capacity": {k: caps[k] for k in MODELS},
               "equilibrium_gap_at_20ms": {n: float(make_model(n).equilibrium_gap(20.0))
                                           for n in MODELS}}
    write_metrics("case1", metrics)
    update_summary("case1", {"capacity_veh_h": {k: round(caps[k]["capacity"], 1) for k in MODELS},
                             "v_crit_ms": {k: round(caps[k]["v_crit"], 2) for k in MODELS}})
    print("\nCase 1 done.")
    return metrics


if __name__ == "__main__":
    main()
