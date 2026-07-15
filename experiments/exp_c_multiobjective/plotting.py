"""Reusable plotting primitives for Experiment C figures.

Publication style copied from the Experiment B convention (colorblind cycle,
white background, grid).  Every figure is saved as both ``.png`` and ``.pdf``.
Regret curves that are strictly positive use a log y-axis.

This module holds only primitives; the experiment-specific figures are
assembled in ``analyze.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Canonical method colours / labels, consistent across every figure.
METHOD_STYLE: dict[str, dict[str, str]] = {
    "right":              {"label": "RIGHT (SoC)",          "color": "#0072B2"},
    "wrong":              {"label": "WRONG",                "color": "#D55E00"},
    "right_algorithm_1":  {"label": "RIGHT Alg. 1 (shared)", "color": "#0072B2"},
    "right_algorithm_2":  {"label": "RIGHT Alg. 2",          "color": "#009E73"},
    "wrong_algorithm_2":  {"label": "WRONG Alg. 2",          "color": "#D55E00"},
}


def style_for(method: str) -> dict[str, str]:
    return METHOD_STYLE.get(method, {"label": method, "color": None})


def apply_style() -> None:
    """Apply the shared publication theme (best-effort; falls back cleanly)."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "grid", "no-latex"])
    except Exception:
        try:
            import seaborn as sns
            sns.set_theme(context="notebook", style="whitegrid",
                          palette="colorblind", font_scale=1.0)
        except Exception:
            pass
    mpl.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm",
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "axes.titleweight": "semibold",
        "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
        "axes.axisbelow": True,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
        "#F0E442", "#000000",
    ])


def save_fig(fig, out_dir: Path, name: str) -> None:
    """Save ``fig`` as ``name.png`` and ``name.pdf`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def curve_with_band(
    ax,
    x: Sequence[float],
    curves: np.ndarray,
    *,
    color: Optional[str],
    label: Optional[str],
    statistic: str = "median",
    pct: tuple[float, float] = (25.0, 75.0),
) -> None:
    """Plot a central curve + inter-seed percentile band.

    Args:
        curves: array ``(n_seeds, n_steps)``.
    """
    curves = np.asarray(curves, dtype=float)
    center = (np.nanmedian(curves, axis=0) if statistic == "median"
              else np.nanmean(curves, axis=0))
    lo = np.nanpercentile(curves, pct[0], axis=0)
    hi = np.nanpercentile(curves, pct[1], axis=0)
    ax.plot(x, center, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


def objective_space(
    ax,
    *,
    front_raw: np.ndarray,
    evaluated_raw: np.ndarray,
    nondominated_raw: np.ndarray,
    title: str,
) -> None:
    """Raw objective-space scatter (ZDT minimization; ``lower is better``)."""
    ax.plot(front_raw[:, 0], front_raw[:, 1], color="#000000", lw=1.5,
            label="true Pareto front", zorder=1)
    ax.scatter(evaluated_raw[:, 0], evaluated_raw[:, 1], s=18, c="#56B4E9",
               alpha=0.6, label="evaluated", zorder=2)
    ax.scatter(nondominated_raw[:, 0], nondominated_raw[:, 1], s=36,
               c="#D55E00", marker="D", label="non-dominated", zorder=3)
    ax.set_xlabel(r"$f_1$ (lower is better)")
    ax.set_ylabel(r"$f_2$ (lower is better)")
    ax.set_title(title)
    ax.legend(loc="best")


def decision_space(
    ax,
    *,
    pareto_set: np.ndarray,
    evaluated_X: np.ndarray,
    nondominated_X: np.ndarray,
    title: str,
) -> None:
    """Decision-space scatter for ``input_dim == 2`` (true set at ``x_2 = 0``)."""
    ax.plot(pareto_set[:, 0], pareto_set[:, 1], color="#000000", lw=1.5,
            label=r"true Pareto set ($x_2=0$)", zorder=1)
    ax.scatter(evaluated_X[:, 0], evaluated_X[:, 1], s=18, c="#56B4E9",
               alpha=0.6, label="evaluated", zorder=2)
    ax.scatter(nondominated_X[:, 0], nondominated_X[:, 1], s=36, c="#D55E00",
               marker="D", label="non-dominated image", zorder=3)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.legend(loc="best")


__all__ = [
    "METHOD_STYLE",
    "style_for",
    "apply_style",
    "save_fig",
    "curve_with_band",
    "objective_space",
    "decision_space",
]
