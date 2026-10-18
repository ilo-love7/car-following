"""Plotting helpers with a headless (Agg) backend and a shared style."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["new_fig", "savefig", "style", "PALETTE", "plot_trajectories", "plot_gap_speed"]

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
    })


def new_fig(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None,
            **kw):
    style()
    if figsize is None:
        figsize = (6.2 * ncols, 3.4 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


def savefig(fig, path: str | Path, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def plot_trajectories(ax, t: np.ndarray, xs: list[np.ndarray], labels=None, **kw):
    for i, x in enumerate(xs):
        ax.plot(t, x, color=PALETTE[i % len(PALETTE)], lw=1.3,
                label=(labels[i] if labels else None), **kw)
    ax.set_xlabel("time [s]"); ax.set_ylabel("position [m]")
    return ax


def plot_gap_speed(axes, t, s, v, v_lead=None, s_ref=None, v_ref=None):
    ax1, ax2 = axes
    ax1.plot(t, s, color=PALETTE[0], label="gap")
    if s_ref is not None:
        ax1.plot(t, s_ref, color=PALETTE[1], ls="--", label="reference")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("gap [m]"); ax1.legend()
    ax2.plot(t, v, color=PALETTE[0], label="follower")
    if v_lead is not None:
        ax2.plot(t, v_lead, color=PALETTE[2], label="leader")
    if v_ref is not None:
        ax2.plot(t, v_ref, color=PALETTE[1], ls="--", label="reference")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("speed [m/s]"); ax2.legend()
    return axes
