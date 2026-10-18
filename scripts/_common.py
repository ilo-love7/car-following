"""Shared helpers for the case scripts: paths, JSON metric I/O, printing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ROOT / "data"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
for _d in (DATA, FIGS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def write_metrics(case: str, metrics: dict) -> Path:
    path = DATA / f"metrics_{case}.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv(case: str, name: str, header: list[str], columns: list[np.ndarray]) -> Path:
    """Write a CSV with a header using the stdlib csv module (handles text columns)."""
    import csv

    path = DATA / f"{case}_{name}.csv"
    cols = [np.asarray(c) for c in columns]
    n = len(cols[0]) if cols else 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(n):
            writer.writerow([_fmt(c[i]) for c in cols])
    return path


def _fmt(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v



def update_summary(case: str, block: dict) -> Path:
    path = DATA / "summary.json"
    summary = {}
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    summary[case] = block
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
