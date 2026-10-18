"""Run every case script in order and print a consolidated summary.

Usage:
    python scripts/run_all.py            # run all cases
    python scripts/run_all.py case1 case3   # run a subset by prefix
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CASES = [
    "case1_equilibrium",
    "case2_linear_stability",
    "case3_ring_string_stability",
    "case4_calibration",
    "case5_prediction",
    "case6_lanechange_mobil",
    "case7_energy",
    "case8_ngsim",
]


def main(argv: list[str]) -> int:
    selected = CASES
    if argv:
        selected = [c for c in CASES if any(c.startswith(a) for a in argv)]
    print(f"Running {len(selected)} case(s): {', '.join(selected)}\n")
    results = {}
    failures = []
    for name in selected:
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            mod.main()
            dt = time.time() - t0
            results[name] = {"status": "ok", "seconds": round(dt, 1)}
            print(f"[ok] {name}  ({dt:.1f} s)")
        except Exception as exc:  # pragma: no cover
            dt = time.time() - t0
            results[name] = {"status": "error", "error": str(exc), "seconds": round(dt, 1)}
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")

    out = HERE.parent / "data" / "run_all_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("All cases completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
