"""NGSIM trajectory loader and car-following event extraction.

NGSIM (US-101 / Lankershim) text files are fully public (US FHWA, distributed via
data.gov) but large; they are **not** bundled here.  Point the loader at a local
copy and it will:

1. read the column-separated file (tab or ``;``) with tolerant header matching,
2. convert US customary units to SI (ft -> m, ft/s -> m/s),
3. extract clean *car-following events* — sustained runs where a follower keeps
   the same predecessor for at least ``min_duration`` seconds in the same lane.

The output is the same ``(leader_trace, t, gap, speed)`` shape used by the
synthetic generator, so every downstream experiment is data-source agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from ..simulate import LeaderTrace

__all__ = ["ngsim_available", "load_ngsim", "extract_following_events", "NGSIM_FPS"]

NGSIM_FPS = 10.0                 # NGSIM is recorded at 10 Hz
FT_TO_M = 0.3048
FTS_TO_MS = 0.3048

#: tolerant header aliases (lower-cased exact names used by different releases)
_COLS = {
    "vid": ["vehicle_id", "vehid"],
    "frame": ["frame_id", "frameid"],
    "time": ["global_time", "time"],
    "x": ["local_x", "localx"],
    "y": ["local_y", "localy"],
    "v": ["v_vel", "vvel", "velocity"],
    "a": ["v_acc", "vacc", "acceleration"],
    "lane": ["lane_id", "laneid", "lane"],
    "prec": ["preceding", "preceding_id"],
    "foll": ["following", "following_id"],
}


def ngsim_available(path: str | Path) -> bool:
    return Path(path).exists()


def _pick(columns: dict[str, str], key: str) -> str | None:
    for alias in _COLS[key]:
        if alias in columns:
            return columns[alias]
    return None


def load_ngsim(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load an NGSIM trajectory file into SI units.

    The returned frame has columns ``vid, frame, t, x, y, v, a, lane, prec``.
    Time ``t`` is seconds since the start of the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"NGSIM file not found: {path}.  Download a trajectory file from "
            f"data.gov (search 'Next Generation Simulation (NGSIM) Vehicle "
            f"Trajectories') and pass its path here."
        )
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else (";" if ";" in first else ",")
    df = pd.read_csv(path, sep=sep)
    cols = {c.strip().lower(): c for c in df.columns}

    vid = _pick(cols, "vid"); frame = _pick(cols, "frame"); time = _pick(cols, "time")
    x = _pick(cols, "x"); y = _pick(cols, "y"); v = _pick(cols, "v"); a = _pick(cols, "a")
    lane = _pick(cols, "lane"); prec = _pick(cols, "prec")
    if vid is None or x is None or v is None:
        raise ValueError(f"unexpected NGSIM header: {list(df.columns)}")

    out = pd.DataFrame()
    out["vid"] = df[vid].astype(int)
    out["frame"] = df[frame].astype(int) if frame is not None else np.arange(len(df))
    out["t"] = (df[time].astype(float) - df[time].min()) / 1000.0 if time is not None else out["frame"] / NGSIM_FPS
    out["x"] = df[x].astype(float) * FT_TO_M
    out["y"] = df[y].astype(float) * FT_TO_M if y is not None else 0.0
    out["v"] = df[v].astype(float) * FTS_TO_MS
    out["a"] = df[a].astype(float) * FTS_TO_MS if a is not None else 0.0
    out["lane"] = df[lane].astype(int) if lane is not None else 0
    out["prec"] = df[prec].astype(float).astype("Int64") if prec is not None else pd.NA
    return out.sort_values(["vid", "frame"]).reset_index(drop=True)


@dataclass
class FollowingEvent:
    follower_id: int
    leader_id: int
    lane: int
    t: np.ndarray
    gap: np.ndarray
    v: np.ndarray
    v_lead: np.ndarray
    x: np.ndarray
    x_lead: np.ndarray

    def leader_trace(self) -> LeaderTrace:
        return LeaderTrace(t=self.t, x=self.x_lead, v=self.v_lead)


def extract_following_events(df: pd.DataFrame, min_duration: float = 20.0,
                             gap_range: tuple[float, float] = (2.0, 80.0),
                             v_min: float = 2.0, max_events: int | None = None) -> list[FollowingEvent]:
    """Extract sustained car-following events from a loaded NGSIM frame."""
    by_id = {vid: g.sort_values("frame") for vid, g in df.groupby("vid")}
    events: list[FollowingEvent] = []
    for vid, g in by_id.items():
        if g["prec"].isna().all():
            continue
        prec = g["prec"].to_numpy()
        n = len(g)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and prec[j + 1] == prec[i]:
                j += 1
            seg = g.iloc[i:j + 1]
            leader_id = prec[i]
            i = j + 1
            if pd.isna(leader_id):
                continue
            leader_id = int(leader_id)
            if leader_id not in by_id:
                continue
            lg = by_id[leader_id]
            merged = pd.merge(seg[["frame", "t", "x", "v", "lane"]],
                              lg[["frame", "x", "v"]], on="frame", suffixes=("", "_lead"))
            if len(merged) < int(min_duration * NGSIM_FPS):
                continue
            gap = merged["x_lead"].to_numpy() - merged["x"].to_numpy() - 5.0
            if not (np.all(gap > gap_range[0]) and np.median(gap) < gap_range[1]):
                continue
            if np.mean(merged["v"].to_numpy()) < v_min:
                continue
            events.append(FollowingEvent(
                follower_id=int(vid), leader_id=leader_id, lane=int(merged["lane"].iloc[0]),
                t=merged["t"].to_numpy(), gap=gap,
                v=merged["v"].to_numpy(), v_lead=merged["v_lead"].to_numpy(),
                x=merged["x"].to_numpy(), x_lead=merged["x_lead"].to_numpy()))
            if max_events is not None and len(events) >= max_events:
                return events
    return events
