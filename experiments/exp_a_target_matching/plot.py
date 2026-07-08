#!/usr/bin/env python
"""Experiment A — target matching: reproduce every analysis figure as a script.

This is the script-based counterpart of ``notebooks/Exp_A_analysis.ipynb``.  It
reads a processed run directory (``process.py`` + ``analyze.py`` must have run
first) and regenerates *all* the figures the notebook produces, saving each one
as ``.png`` and ``.pdf`` under::

    <run_dir>/figures/<benchmark>/

Usage::

    python -m experiments.exp_a_target_matching.plot \
        --run-dir results/exp_a_target_matching/runs/<run_id>

By default every benchmark found under ``processed/per_seed/`` is plotted; pass
``--benchmark`` (repeatable) to restrict the set.  The plotting logic is copied
verbatim from the notebook; only the run/benchmark selection and figure saving
are turned into a reusable function.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless: never require a display / GUI backend
import matplotlib as mpl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

from common.serialization import load_json
from common.acquisition_utils import BetaSchedule
from benchmarks import build_benchmark

logger = logging.getLogger("exp_a.plot")


def _long_path(path: Path) -> str:
    """Return a save path safe against Windows' 260-char MAX_PATH limit.

    On Windows the deep ``results/.../figures/<fig>.png`` paths can exceed
    MAX_PATH, which makes ``open()`` fail with a misleading
    ``FileNotFoundError`` even though the directory exists.  Prefixing the
    absolute path with the ``\\?\`` extended-length marker lifts the limit.
    No-op on non-Windows platforms.
    """
    if os.name != "nt":
        return str(path)
    abs_path = os.path.abspath(str(path))
    return abs_path if abs_path.startswith("\\\\?\\") else "\\\\?\\" + abs_path


# --- global plotting style (copied from the notebook) -----------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# --- centralized publication style (copied from the Exp-B notebook) ---------
def _apply_style() -> str:
    """Apply the notebook's figure theme; return the backend that was used."""
    backend = "matplotlib-fallback"
    try:
        import scienceplots  # noqa: F401  (registers the 'science' style sheets)
        plt.style.use(["science", "grid", "no-latex"])
        backend = "scienceplots"
    except Exception:
        try:
            import seaborn as sns
            sns.set_theme(context="notebook", style="whitegrid",
                          palette="colorblind", font_scale=1.0)
            backend = "seaborn"
        except Exception:
            pass

    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.transparent": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.titleweight": "semibold",
        "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.grid": True, "grid.color": "#B0B0B0",
        "grid.linewidth": 0.6, "grid.alpha": 0.35, "axes.axisbelow": True,
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC", "legend.fancybox": False,
    })
    colorblind_cycle = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                        "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colorblind_cycle)
    return backend


def _clear_stale_figures(out_dir: Path) -> None:
    """Remove pre-existing figure files so re-running plot.py overwrites
    in place instead of accumulating stale files left over from figures
    that a previous code revision produced under a different name."""
    for pattern in ("*.png", "*.pdf"):
        for f in out_dir.glob(pattern):
            f.unlink()


# Canonical colours / labels for the two compared methods.
METHOD_STYLE = {
    "right_target_matching_ucb": {"label": "RIGHT (GP on f)",     "color": "#1f77b4"},
    "wrong_gp_on_abs_lcb":       {"label": "WRONG (GP on |f-T|)", "color": "#d62728"},
}


def style_for(method: str) -> dict:
    """Style for a method, with a neutral fallback for unknown names."""
    return METHOD_STYLE.get(method, {"label": method, "color": None})


# --- single-benchmark plotting (notebook body, verbatim logic) --------------
def plot_benchmark(run_dir: Path, benchmark_name: str) -> int:
    """Reproduce and save every Exp-A figure for one benchmark.

    Returns the number of figures written.  Mirrors the notebook cell-by-cell;
    every ``plt.show()`` is dropped and each figure is closed after saving.
    """
    RUN_DIR = run_dir
    BENCHMARK = benchmark_name

    RAW_DIR = RUN_DIR / "raw" / BENCHMARK
    AGG_DIR = RUN_DIR / "processed" / "aggregate" / BENCHMARK
    PER_SEED_DIR = RUN_DIR / "processed" / "per_seed" / BENCHMARK
    FIG_DIR = RUN_DIR / "figures" / BENCHMARK
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Global figure registry: every plotting block does add_figure("<name>", fig).
    FIGURES: dict[str, plt.Figure] = {}

    def add_figure(name: str, fig: plt.Figure) -> plt.Figure:
        if name in FIGURES:
            logger.warning("overwriting already-registered figure '%s'", name)
        FIGURES[name] = fig
        return fig

    # ---- load config, metadata and aggregate files --------------------------
    config = load_json(RUN_DIR / "config.json")
    metadata = load_json(RUN_DIR / "metadata.json")  # noqa: F841 (kept for parity)

    AGG_FILES = [
        "convergence_curves",
        "calibration_curves",
        "final_performance",
        "calibration_summary",
        "statistical_tests",
    ]
    agg: dict[str, Optional[dict]] = {}
    for name in AGG_FILES:
        p = AGG_DIR / f"{name}.json"
        agg[name] = load_json(p) if p.exists() else None

    def load_all_per_seed(per_seed_dir: Path) -> list[dict]:
        files = sorted(per_seed_dir.glob("seed_*_processed.json"))
        if not files:
            raise FileNotFoundError(f"No processed per-seed files under {per_seed_dir}")
        return [load_json(p) for p in files]

    def method_names(seed_record: dict) -> list[str]:
        return list(seed_record["methods"].keys())

    all_seeds = load_all_per_seed(PER_SEED_DIR)
    seed0 = all_seeds[0]
    methods = method_names(seed0)

    _raw_files = sorted(RAW_DIR.glob("seed_*.json")) if RAW_DIR.exists() else []
    raw0 = load_json(_raw_files[0]) if _raw_files else None

    T = float(config["target"])

    benchmark = build_benchmark(config)
    DIM = int(benchmark.dim)

    # Scalar UCB coefficient used by the acquisition diagnostics.  Handles both
    # a constant `beta` and a `beta_schedule` (log / matern), mirroring run.py;
    # for a constant schedule this equals the former float(config["beta"]).
    BETA = float(BetaSchedule.from_config(config).beta_at(1, DIM))

    PREDICTIONS_AVAILABLE = any(
        c.get("predictions") for c in seed0["methods"][methods[0]]["calibrations"]
    )

    logger.info(
        "[%s] %d seeds | methods=%s | dim=%d | T=%g | beta=%g | predictions=%s",
        BENCHMARK, len(all_seeds), methods, DIM, T, BETA, PREDICTIONS_AVAILABLE,
    )

    # ---- sanity checks ------------------------------------------------------
    cfg_bench = config["benchmark"]
    cfg_bench_name = cfg_bench["name"] if isinstance(cfg_bench, dict) else cfg_bench
    assert cfg_bench_name == BENCHMARK, (
        f"config benchmark {cfg_bench_name!r} != selected BENCHMARK {BENCHMARK!r}"
    )
    for label, path in [("RUN_DIR", RUN_DIR), ("AGG_DIR", AGG_DIR), ("PER_SEED_DIR", PER_SEED_DIR)]:
        assert path.exists(), f"{label} does not exist: {path}"
    for s in all_seeds:
        bm = s.get("metadata", {}).get("benchmark", BENCHMARK)
        assert bm == BENCHMARK, f"per-seed record benchmark {bm!r} != {BENCHMARK!r}"
    n_seeds = len(all_seeds)

    def band_caption(payload: Optional[dict], n: int) -> str:
        if not payload:
            return f"aggregated across {n} seeds"
        stat = payload.get("statistic", "?")
        bt = payload.get("band_type", "?")
        conf = payload.get("confidence")
        conf_s = f"{int(round(100 * conf))}%" if conf is not None else "?"
        simul = " (simultaneous)" if bt == "uniform" else ""
        mean_note = "; dashed: mean" if payload.get("mean") is not None else ""
        return f"{stat} over {n} seeds, {conf_s} {bt} bootstrap band{simul}{mean_note}"

    # ========================================================================
    # 1. Geometry of the target-matching problem
    # ========================================================================
    def _eval_f(X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=torch.double)
        return benchmark.f(Xt).detach().cpu().numpy().reshape(-1)

    if DIM == 1:
        lb, ub = float(benchmark.bounds[0, 0]), float(benchmark.bounds[1, 0])
        x_grid = np.linspace(lb, ub, 1001)
        f_grid = _eval_f(x_grid.reshape(-1, 1))
        abs_grid = np.abs(f_grid - T)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
        ax = axes[0]
        ax.plot(x_grid, f_grid, color="black", lw=1.6, label=r"$f(x)$")
        ax.axhline(T, color="#2ca02c", ls="--", lw=1.3, label=fr"$T = {T:g}$")
        sign_change = np.where(np.diff(np.sign(f_grid - T)))[0]
        for k, idx in enumerate(sign_change):
            ax.axvline(x_grid[idx], color="#2ca02c", ls=":", lw=0.9, alpha=0.7,
                       label="target crossing" if k == 0 else None)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$f(x)$")
        ax.set_title("Raw function $f$ (modelled by RIGHT)")
        ax.legend(loc="best")
        ax = axes[1]
        ax.plot(x_grid, abs_grid, color="black", lw=1.6, label=r"$|f(x)-T|$")
        ax.axhline(0.0, color="#2ca02c", ls="--", lw=1.3, label="zero level")
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$|f(x)-T|$")
        ax.set_title(r"Transformed objective $|f-T|$ (modelled by WRONG)")
        ax.legend(loc="best")
        fig.suptitle(f"{BENCHMARK} geometry -- target $T = {T:g}$  "
                     f"({len(sign_change)} target crossing(s))")
        fig.tight_layout()
        add_figure("01_geometry", fig)

    elif DIM == 2:
        lb = benchmark.bounds[0].detach().cpu().numpy()
        ub = benchmark.bounds[1].detach().cpu().numpy()
        n = 200
        xs = np.linspace(lb[0], ub[0], n)
        ys = np.linspace(lb[1], ub[1], n)
        XX, YY = np.meshgrid(xs, ys)
        F = _eval_f(np.column_stack([XX.ravel(), YY.ravel()])).reshape(n, n)
        A = np.abs(F - T)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        ax = axes[0]; ax.grid(False)
        cf = ax.contourf(XX, YY, F, levels=30, cmap="viridis")
        ax.contour(XX, YY, F, levels=[T], colors="white", linewidths=1.8)
        fig.colorbar(cf, ax=ax, shrink=0.9)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title(f"Raw $f$ (modelled by RIGHT)\nwhite line: target set $f = {T:g}$")
        ax = axes[1]; ax.grid(False)
        cf = ax.contourf(XX, YY, A, levels=30, cmap="viridis")
        ax.contour(XX, YY, A, levels=[0.0], colors="white", linewidths=1.2)
        fig.colorbar(cf, ax=ax, shrink=0.9)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title(r"Transformed $|f-T|$ (modelled by WRONG)")
        fig.suptitle(f"{BENCHMARK} geometry -- target $T = {T:g}$ "
                     f"(target match is a curve in 2D)")
        fig.tight_layout()
        add_figure("01_geometry", fig)

    else:
        f_sample = _eval_f(benchmark.test_set(4096, seed=0).detach().cpu().numpy())
        logger.info("benchmark %s has dim=%d; geometry plot skipped "
                    "(f range on Sobol sample [%.4g, %.4g], T=%g).",
                    BENCHMARK, DIM, f_sample.min(), f_sample.max(), T)

    # ========================================================================
    # 2. Mechanism on one seed
    # ========================================================================
    POSTERIOR_SEED = 1
    POSTERIOR_ITERS_REQUESTED = [5, 10, 20, 40, 50]

    def seed_record_for(seed_idx: int) -> dict:
        for s in all_seeds:
            if s.get("metadata", {}).get("seed") == seed_idx:
                return s
        return all_seeds[seed_idx]

    def calibration_at(record: dict, method: str, iteration: int) -> Optional[dict]:
        for c in record["methods"][method]["calibrations"]:
            if c["iteration_index"] == iteration:
                return c
        return None

    def reconstruct_test_grid() -> tuple[np.ndarray, np.ndarray]:
        n_test = int(config["test_grid"]["n_test"])
        grid_seed = int(config["test_grid"].get("seed", 0))
        X_test = benchmark.test_set(n=n_test, seed=grid_seed)
        f_test = benchmark.f(X_test).detach().cpu().numpy().reshape(-1)
        return X_test.detach().cpu().numpy(), f_test

    def modelled_truth(method: str, f_vals: np.ndarray) -> np.ndarray:
        if "wrong" in method:
            return np.abs(f_vals - T)
        return f_vals

    def reconstruct_acquisition(method: str, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        if "wrong" in method:
            return -mu + np.sqrt(BETA) * sd
        return -np.abs(mu - T) + np.sqrt(BETA) * sd

    _rec = seed_record_for(POSTERIOR_SEED)
    CALIB_ITERS = [c["iteration_index"] for c in _rec["methods"][methods[0]]["calibrations"]]
    POSTERIOR_ITERS = [it for it in POSTERIOR_ITERS_REQUESTED if it in CALIB_ITERS]

    # ---- 2.1 posterior snapshots -------------------------------------------
    if DIM == 1 and PREDICTIONS_AVAILABLE and POSTERIOR_ITERS:
        record = seed_record_for(POSTERIOR_SEED)
        X_test, f_test = reconstruct_test_grid()
        x_test = X_test.reshape(-1)
        order = np.argsort(x_test)

        n_rows = len(methods) + 1
        fig, axes = plt.subplots(
            n_rows, len(POSTERIOR_ITERS),
            figsize=(2.6 * len(POSTERIOR_ITERS), 2.6 * n_rows),
            sharex=True,
        )
        axes = np.atleast_2d(axes)
        acq_by_iter: dict = {}

        for i, method in enumerate(methods):
            style = style_for(method)
            truth = modelled_truth(method, f_test)
            for j, it in enumerate(POSTERIOR_ITERS):
                ax = axes[i, j]
                cal = calibration_at(record, method, it)
                pred = cal.get("predictions") if cal else None
                if not pred:
                    ax.text(0.5, 0.5, f"no predictions\nat t={it}",
                            transform=ax.transAxes, ha="center", va="center")
                    ax.set_axis_off()
                    continue
                mu = np.asarray(pred["posterior_mean"]).reshape(-1)[order]
                sd = np.asarray(pred["posterior_std"]).reshape(-1)[order]
                xo = x_test[order]; yo = truth[order]
                ax.fill_between(xo, mu - 2 * sd, mu + 2 * sd, color=style["color"],
                                alpha=0.20, label=r"$\mu \pm 2\sigma$")
                ax.plot(xo, mu, color=style["color"], lw=1.4, label=r"$\mu(x)$")
                ax.plot(xo, yo, color="black", lw=1.0, ls="--", label="modelled truth")
                ax.axhline(T if "wrong" not in method else 0.0,
                           color="#2ca02c", ls=":", lw=1.0)
                acq_by_iter[(method, it)] = (xo, reconstruct_acquisition(method, mu, sd))
                if i == 0:
                    ax.set_title(f"$t = {it}$")
                if j == 0:
                    ax.set_ylabel(style["label"])

        acq_row = len(methods)
        for j, it in enumerate(POSTERIOR_ITERS):
            ax = axes[acq_row, j]
            for method in methods:
                if (method, it) not in acq_by_iter:
                    continue
                style = style_for(method)
                xo, acq = acq_by_iter[(method, it)]
                ax.plot(xo, acq, color=style["color"], lw=1.4, label=style["label"])
                k = int(np.argmax(acq))
                ax.scatter(xo[k], acq[k], color=style["color"], s=20, zorder=3)
            ax.axhline(0.0, color="black", lw=0.8, alpha=0.3)
            if j == 0:
                ax.set_ylabel("Acquisition")
            ax.set_xlabel(r"$x$")

        handles, labels = [], []
        for ax in axes.ravel():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h); labels.extend(l)
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc="lower center",
                   ncol=min(len(unique), 5), bbox_to_anchor=(0.5, -0.01), frameon=False)
        fig.suptitle(f"Posterior and acquisition evolution (seed {POSTERIOR_SEED})")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        add_figure("02_posterior_and_acquisition_evolution", fig)

    elif DIM == 2 and PREDICTIONS_AVAILABLE and POSTERIOR_ITERS and raw0 is not None:
        record = seed_record_for(POSTERIOR_SEED)
        X_test, f_test = reconstruct_test_grid()
        X_test = np.asarray(X_test)
        x1, x2 = X_test[:, 0], X_test[:, 1]
        f_flat = f_test.reshape(-1)
        near = np.abs(f_flat - T)
        eps = config["near_target_epsilon"]
        X0 = np.asarray(raw0["initial_design"]["X"])
        n0 = len(X0)
        plot_cols = ["init"] + list(POSTERIOR_ITERS)

        fig, axes = plt.subplots(len(methods), len(plot_cols),
                                 figsize=(3.0 * len(plot_cols), 3.0 * len(methods)),
                                 sharex=True, sharey=True)
        axes = np.atleast_2d(axes)

        for i, method in enumerate(methods):
            style = style_for(method)
            Xq = np.asarray([step["x"] for step in raw0["methods"][method]["trajectory"]])
            X_hist = np.vstack([X0, Xq])

            for j, col in enumerate(plot_cols):
                ax = axes[i, j]
                ax.grid(False)

                if col == "init":
                    ax.tricontourf(x1, x2, near, levels=30)
                    ax.tricontourf(x1, x2, near, levels=[0.0, eps], colors=["white"], alpha=0.18)
                    ax.tricontour(x1, x2, near, levels=[eps], colors="white",
                                  linewidths=1.0, linestyles="--")
                    ax.tricontour(x1, x2, f_flat, levels=[T], colors="white", linewidths=1.2)
                    ax.scatter(X0[:, 0], X0[:, 1], facecolors="white", edgecolors="black",
                               linewidth=1.0, s=42, zorder=6)
                    for q, x in enumerate(X0):
                        ax.text(x[0], x[1], f"I{q}", fontsize=6, ha="center", va="center",
                                color="black", zorder=7)
                    if i == 0:
                        ax.set_title("init")
                    if j == 0:
                        ax.set_ylabel(style["label"])
                    ax.set_xlabel(r"$x_1$")
                    continue

                it = col
                cal = calibration_at(record, method, it)
                pred = cal.get("predictions") if cal else None
                if not pred:
                    ax.text(0.5, 0.5, f"no predictions\nat t={it}",
                            transform=ax.transAxes, ha="center", va="center")
                    ax.set_axis_off()
                    continue

                mu = np.asarray(pred["posterior_mean"]).reshape(-1)
                sd = np.asarray(pred["posterior_std"]).reshape(-1)
                acq = reconstruct_acquisition(method, mu, sd)
                cf = ax.tricontourf(x1, x2, acq, levels=30)
                ax.tricontourf(x1, x2, near, levels=[0.0, eps], colors=["white"], alpha=0.18)
                ax.tricontour(x1, x2, near, levels=[eps], colors="white",
                              linewidths=1.0, linestyles="--")
                ax.tricontour(x1, x2, f_flat, levels=[T], colors="white", linewidths=1.2)

                X_obs = X_hist[:n0 + it]
                ax.plot(X_obs[:, 0], X_obs[:, 1], color=style["color"],
                        alpha=0.35, lw=0.8, zorder=4)
                arrow_every = max(1, len(X_obs) // 8)
                for a, b in zip(X_obs[:-1:arrow_every], X_obs[1::arrow_every]):
                    ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                                arrowprops=dict(arrowstyle="->", color=style["color"],
                                                lw=0.8, alpha=0.65),
                                zorder=5)

                ax.scatter(X_obs[n0:, 0], X_obs[n0:, 1], color=style["color"],
                           edgecolor="white", linewidth=0.4, s=18, alpha=0.9, zorder=6)
                ax.scatter(X_obs[:n0, 0], X_obs[:n0, 1], facecolors="white",
                           edgecolors="black", linewidth=1.0, s=42, zorder=7)
                ax.scatter(X_obs[-1, 0], X_obs[-1, 1], color=style["color"],
                           edgecolor="black", linewidth=0.9, s=55, zorder=8)

                label_steps = set(range(n0)) | {n0 + k for k in [1, 5, 10, 20, 40, 50] if n0 + k < len(X_obs)}
                for q, x in enumerate(X_obs):
                    if q in label_steps:
                        label = f"I{q}" if q < n0 else str(q - n0)
                        ax.text(x[0], x[1], label, fontsize=6, ha="center", va="center",
                                color="black", bbox=dict(facecolor="white", alpha=0.65, pad=0.1),
                                zorder=9)

                k = int(np.argmax(acq))
                ax.scatter(x1[k], x2[k], color="black", marker="*", s=45, zorder=10)

                if i == 0:
                    ax.set_title(f"$t = {it}$")
                if j == 0:
                    ax.set_ylabel(style["label"])
                ax.set_xlabel(r"$x_1$")
                if j == len(plot_cols) - 1:
                    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
                    cbar.set_label(r"$\alpha(x)$")

        fig.suptitle(f"Acquisition evolution on test grid (seed {POSTERIOR_SEED})")
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        add_figure("02_acquisition_evolution_2d", fig)

    else:
        logger.info(
            "single-seed mechanism plot skipped "
            "(DIM=%d, predictions=%s, iters=%s, raw=%s).",
            DIM, PREDICTIONS_AVAILABLE, POSTERIOR_ITERS, raw0 is not None,
        )

    # ========================================================================
    # 3. Optimization performance across seeds
    # ========================================================================
    conv = agg["convergence_curves"]
    assert conv is not None, "convergence_curves.json missing — cannot plot convergence."
    conv_iters = np.asarray(conv["iteration"])
    CONV_N = conv.get("n_seeds", n_seeds)

    def plot_convergence_metric(ax, metric_key: str, logy: bool = False) -> Optional[dict]:
        sample = None
        for method in methods:
            payload = conv["methods"].get(method, {})
            if metric_key not in payload:
                logger.warning("%s: convergence metric '%s' missing", method, metric_key)
                continue
            cur = payload[metric_key]
            sample = sample or cur
            style = style_for(method)
            ax.plot(conv_iters, np.asarray(cur["center"]), color=style["color"], lw=1.8,
                    label=style["label"])
            ax.fill_between(conv_iters, np.asarray(cur["lower"]), np.asarray(cur["upper"]),
                            color=style["color"], alpha=0.20)
            # Secondary dashed curve: pointwise mean across seeds (median is the
            # solid center).  Absent on aggregate files written before analyze.py
            # stored it, in which case we simply skip it.
            mean_curve = cur.get("mean")
            if mean_curve is not None:
                ax.plot(conv_iters, np.asarray(mean_curve), color=style["color"],
                        lw=1.1, ls="--", alpha=0.85)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("iteration $t$")
        return sample

    # 3.2 best-so-far target distance
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sample = plot_convergence_metric(ax, "running_best_target_distance", logy=True)
    ax.set_ylabel(r"$\min_{s \leq t}\,|f(x_s) - T|$")
    ft = config.get("failure_threshold")
    if ft is not None:
        ax.axhline(ft, color="grey", ls=":", lw=1.0, label=f"failure threshold = {ft:g}")
    ax.set_title("Best-so-far target distance -- " + band_caption(sample, CONV_N))
    ax.legend(loc="best")
    fig.tight_layout()
    add_figure("03_running_best_target_distance", fig)

    # 3.3 instantaneous target distance
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sample = plot_convergence_metric(ax, "instantaneous_target_distance", logy=True)
    ax.set_ylabel(r"$|f(x_t) - T|$")
    ax.set_title("Instantaneous target distance -- " + band_caption(sample, CONV_N))
    ax.legend(loc="best")
    fig.tight_layout()
    add_figure("03_instantaneous_target_distance", fig)

    # 3.4 cumulative regret
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    s1 = plot_convergence_metric(axes[0], "cumulative_inst_target_regret")
    axes[0].set_ylabel(r"$\sum_{s \leq t} |f(x_s)-T|$")
    axes[0].set_title("Cumulative instantaneous regret")
    axes[0].legend(loc="best")
    s2 = plot_convergence_metric(axes[1], "cumulative_bsf_target_regret")
    axes[1].set_ylabel(r"$\sum_{s \leq t} D_s$")
    axes[1].set_title("Cumulative best-so-far regret")
    axes[1].legend(loc="best")
    fig.suptitle("Cumulative target regret -- " + band_caption(s1 or s2, CONV_N))
    fig.tight_layout()
    add_figure("03_cumulative_regret", fig)

    # 3.5 final performance distribution
    def stack_final(metric: str) -> dict[str, np.ndarray]:
        return {m: np.array([s["methods"][m]["final"][metric] for s in all_seeds], float)
                for m in methods}

    DIST_METRICS = ["cumulative_inst_target_regret", "cumulative_bsf_target_regret"]
    fig, axes = plt.subplots(1, len(DIST_METRICS), figsize=(5.6 * len(DIST_METRICS), 4.2))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, DIST_METRICS):
        data = stack_final(metric)
        positions = np.arange(len(methods))
        vals = [np.clip(data[m], 1e-12, None) for m in methods]
        bp = ax.boxplot(vals, positions=positions, widths=0.5, showfliers=False,
                        patch_artist=True, whis=(5, 95),
                        showmeans=True, meanline=True,
                        medianprops=dict(color="black", lw=1.4),
                        meanprops=dict(color="black", ls="--", lw=1.4))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(style_for(m)["color"]); patch.set_alpha(0.30)
        for k, m in enumerate(methods):
            jitter = (np.random.default_rng(k).random(len(vals[k])) - 0.5) * 0.18
            ax.scatter(np.full_like(vals[k], k) + jitter, vals[k], s=10,
                       color=style_for(m)["color"], alpha=0.5, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([style_for(m)["label"] for m in methods], rotation=10)
        ax.set_title(metric)
    # box: 25-75%, whiskers: 5th-95th percentile; solid line = median, dashed = mean
    from matplotlib.lines import Line2D
    _dist_handles = [Line2D([0], [0], color="black", lw=1.4, label="median"),
                     Line2D([0], [0], color="black", lw=1.4, ls="--", label="mean")]
    axes[0].legend(handles=_dist_handles, loc="best", fontsize=8)
    fig.suptitle(f"Final performance distribution across {n_seeds} seeds (log scale)\n"
                 "box: 25-75%, whiskers: 5-95%, solid: median, dashed: mean")
    fig.tight_layout()
    add_figure("03_final_distribution", fig)

    # 3.6 paired RIGHT-vs-WRONG comparison
    PAIR_METRIC = "final_best_target_distance"
    if len(methods) == 2:
        right_m, wrong_m = methods[0], methods[1]
        if "right" in methods[1] and "wrong" in methods[0]:
            right_m, wrong_m = methods[1], methods[0]
        data = stack_final(PAIR_METRIC)
        r = np.clip(data[right_m], 1e-15, None)
        w = np.clip(data[wrong_m], 1e-15, None)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        ax = axes[0]
        lim_lo = min(r.min(), w.min()) * 0.5
        lim_hi = max(r.max(), w.max()) * 2.0
        ax.scatter(r, w, s=18, color="#555555", alpha=0.6, zorder=3)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="grey", ls=":", lw=1.0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("RIGHT  $D_N$"); ax.set_ylabel("WRONG  $D_N$")
        ax.set_aspect("equal")
        ax.set_title("Paired final target distance\n(points above the diagonal: RIGHT better)")

        ax = axes[1]
        diff = data[wrong_m] - data[right_m]
        ax.hist(diff, bins=30, color="#1f77b4", alpha=0.75)
        ax.axvline(0.0, color="black", lw=1.0, ls=":")
        ax.set_xlabel(r"$D_N^{\mathrm{WRONG}} - D_N^{\mathrm{RIGHT}}$  (>0 favours RIGHT)")
        ax.set_ylabel("count of seeds")
        ax.set_title("Paired difference")

        st = agg["statistical_tests"]
        if st is not None:
            comp = st["comparisons"][0]
            payload = comp["metrics"].get(PAIR_METRIC)
            if payload is not None:
                wil = payload.get("wilcoxon_paired", {})
                txt = (f"Wilcoxon p = {wil.get('p_value'):.3g}\n"
                       f"median(W-R) = {wil.get('median_difference_wrong_minus_right'):.3g}\n"
                       f"rank-biserial = {payload.get('rank_biserial'):.3g}")
                ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        fig.suptitle(f"Paired comparison on {PAIR_METRIC}")
        fig.tight_layout()
        add_figure("03_paired_comparison", fig)
    else:
        logger.info("paired comparison expects exactly 2 methods; found %s.", methods)

    # 3.7 failure-rate analysis
    ft = config.get("failure_threshold")
    fp = agg["final_performance"]
    if ft is not None and fp is not None:
        rates = {style_for(m)["label"]: fp["methods"][m].get("failure_rate", np.nan)
                 for m in methods}
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        labels = list(rates.keys())
        colors = [style_for(m)["color"] for m in methods]
        ax.bar(labels, [rates[l] for l in labels], color=colors, alpha=0.75)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("failure rate")
        ax.set_title(f"Overall failure rate (threshold = {ft:g})")
        for i, l in enumerate(labels):
            ax.text(i, rates[l] + 0.02, f"{rates[l]:.2f}", ha="center")
        fig.tight_layout()
        add_figure("03_failure_rate", fig)
    else:
        logger.info("failure_threshold not set or final_performance missing — skipping failure rate.")

    # ========================================================================
    # 4. Probabilistic model diagnostics
    # ========================================================================
    cal = agg["calibration_curves"]
    assert cal is not None, "calibration_curves.json missing — cannot plot calibration."
    CAL_N = cal.get("n_seeds", n_seeds)

    def cal_iters(method: str) -> np.ndarray:
        return np.asarray(cal["methods"][method]["iteration"])

    def has_cal(method: str, region: str, key: str) -> bool:
        return key in cal["methods"].get(method, {}).get(region, {})

    def plot_cal_metric(ax, region: str, key: str, *, logy: bool = False,
                        nominal: Optional[float] = None) -> Optional[dict]:
        sample = None
        for method in methods:
            block = cal["methods"].get(method, {}).get(region, {})
            if key not in block:
                continue
            payload = block[key]
            sample = sample or payload
            style = style_for(method)
            it = cal_iters(method)
            ax.plot(it, np.asarray(payload["center"]), color=style["color"], lw=1.8,
                    label=style["label"])
            ax.fill_between(it, np.asarray(payload["lower"]), np.asarray(payload["upper"]),
                            color=style["color"], alpha=0.20)
        if nominal is not None:
            ax.axhline(nominal, color="grey", ls=":", lw=1.0, label=f"nominal {nominal:g}")
        if logy:
            ax.set_yscale("symlog")
        ax.set_xlabel("iteration $t$")
        return sample

    # 4.2 NLPD
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    s1 = plot_cal_metric(axes[0], "global_calibration", "nlpd", logy=True)
    axes[0].set_ylabel("NLPD (test grid)")
    axes[0].set_title("NLPD on full test grid")
    axes[0].legend(loc="best")
    s2 = plot_cal_metric(axes[1], "near_target", "nlpd", logy=True)
    axes[1].set_ylabel("NLPD (near target)")
    axes[1].set_title("NLPD near the target set")
    axes[1].legend(loc="best")
    fig.suptitle("NLPD -- " + band_caption(s1 or s2, CAL_N))
    fig.tight_layout()
    add_figure("04_nlpd", fig)

    # 4.3 coverage / violation
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    s1 = plot_cal_metric(axes[0], "global_calibration", "coverage", nominal=0.95)
    axes[0].set_ylim(0.0, 1.05); axes[0].set_ylabel("empirical coverage @95%")
    axes[0].set_title("Coverage (full test grid)")
    axes[0].legend(loc="best")
    s2 = plot_cal_metric(axes[1], "global_calibration", "confidence_violation_rate")
    axes[1].set_ylabel("confidence violation rate")
    axes[1].set_title("Confidence-bound violation rate")
    axes[1].legend(loc="best")
    fig.suptitle("Coverage / violation -- " + band_caption(s1 or s2, CAL_N))
    fig.tight_layout()
    add_figure("04_coverage_violation", fig)

    # 4.4 standardized residuals
    res_keys = ["mean_standardized_residual", "std_standardized_residual",
                "mean_abs_standardized_residual"]
    ref_lines = {"mean_standardized_residual": 0.0, "std_standardized_residual": 1.0,
                 "mean_abs_standardized_residual": float(np.sqrt(2 / np.pi))}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    sample = None
    for ax, key in zip(axes, res_keys):
        s = plot_cal_metric(ax, "global_calibration", key, nominal=ref_lines[key])
        sample = sample or s
        ax.set_ylabel(key); ax.set_title(key.replace("_", " "))
        ax.legend(loc="best")
    fig.suptitle("Standardized residuals -- " + band_caption(sample, CAL_N))
    fig.tight_layout()
    add_figure("04_standardized_residuals", fig)

    # 4.5 PIT moments
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    s1 = plot_cal_metric(axes[0], "global_calibration", "pit_mean", nominal=0.5)
    axes[0].set_ylabel("PIT mean"); axes[0].set_title("PIT mean trajectory")
    axes[0].legend(loc="best")
    s2 = plot_cal_metric(axes[1], "global_calibration", "pit_std",
                         nominal=float(1 / np.sqrt(12)))
    axes[1].set_ylabel("PIT std"); axes[1].set_title("PIT std trajectory")
    axes[1].legend(loc="best")
    fig.suptitle("PIT moments -- " + band_caption(s1 or s2, CAL_N))
    fig.tight_layout()
    add_figure("04_pit_moments", fig)

    # 4.5b pooled PIT histogram
    PIT_ITERATION = int(cal_iters(methods[0])[-1])

    def pooled_standardized_residuals(method: str, iteration: int,
                                      f_test: np.ndarray) -> np.ndarray:
        truth = modelled_truth(method, f_test)
        zs = []
        for s in all_seeds:
            c = next((c for c in s["methods"][method]["calibrations"]
                      if c["iteration_index"] == iteration), None)
            pred = c.get("predictions") if c else None
            if not pred:
                continue
            mu = np.asarray(pred["posterior_mean"])
            sd = np.clip(np.asarray(pred["posterior_std"]), 1e-12, None)
            zs.append((truth - mu) / sd)
        return np.concatenate(zs) if zs else np.array([])

    if PREDICTIONS_AVAILABLE:
        _, f_test_full = reconstruct_test_grid()
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        bins = np.linspace(0.0, 1.0, 21)
        for method in methods:
            z = pooled_standardized_residuals(method, PIT_ITERATION, f_test_full)
            if z.size == 0:
                continue
            ax.hist(norm.cdf(z), bins=bins, density=True, histtype="step", lw=1.6,
                    color=style_for(method)["color"], label=style_for(method)["label"])
        ax.axhline(1.0, color="grey", ls=":", lw=1.0, label="uniform")
        ax.set_xlabel("PIT value"); ax.set_ylabel("density")
        ax.set_title(f"Pooled PIT histogram at $t = {PIT_ITERATION}$ (across {n_seeds} seeds)")
        ax.legend(loc="best")
        fig.tight_layout()
        add_figure("04_pit_histogram", fig)
    else:
        logger.info("pooled PIT histogram skipped (no stored posterior predictions).")

    # 4.6 QQ diagnostics
    if PREDICTIONS_AVAILABLE:
        _, f_test_full = reconstruct_test_grid()
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        qq_extreme = 0.0
        for method in methods:
            z = pooled_standardized_residuals(method, PIT_ITERATION, f_test_full)
            if z.size == 0:
                continue
            z_sorted = np.sort(z)
            if z_sorted.size > 4000:
                idx = np.linspace(0, z_sorted.size - 1, 4000).astype(int)
                z_sorted = z_sorted[idx]
            probs = (np.arange(1, z_sorted.size + 1) - 0.5) / z_sorted.size
            theo = norm.ppf(probs)
            ax.plot(theo, z_sorted, lw=1.2, color=style_for(method)["color"],
                    label=style_for(method)["label"])
            qq_extreme = max(qq_extreme, float(np.max(np.abs(z_sorted))), float(np.max(np.abs(theo))))
        lim = max(qq_extreme, 4.0)
        ax.plot([-lim, lim], [-lim, lim], color="grey", ls=":", lw=1.0, label="diagonal")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(r"theoretical $\mathcal{N}(0,1)$ quantile")
        ax.set_ylabel(r"empirical quantile of $z = (y-\mu)/\sigma$")
        ax.set_title(f"QQ plot of standardized residuals at $t = {PIT_ITERATION}$")
        ax.legend(loc="best")
        fig.tight_layout()
        add_figure("04_qq", fig)
    else:
        logger.info("QQ plot skipped (no stored posterior predictions).")

    # 4.7 near-target calibration
    if any(has_cal(m, "near_target", "coverage") for m in methods):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        s1 = plot_cal_metric(axes[0], "near_target", "coverage", nominal=0.95)
        axes[0].set_ylim(0.0, 1.05); axes[0].set_ylabel("coverage @95% (near target)")
        axes[0].set_title("Near-target coverage")
        axes[0].legend(loc="best")
        s2 = plot_cal_metric(axes[1], "near_target", "confidence_violation_rate")
        axes[1].set_ylabel("violation rate (near target)")
        axes[1].set_title("Near-target violation rate")
        axes[1].legend(loc="best")
        eps = config.get("near_target_epsilon")
        fig.suptitle(f"Near-target calibration (epsilon = {eps:g}) -- " + band_caption(s1 or s2, CAL_N))
        fig.tight_layout()
        add_figure("04_near_target_calibration", fig)
    else:
        logger.info("near-target calibration curves not available — skipping.")

    # 4.8 joint posterior diagnostics
    joint_keys = ["nlpd_joint", "mahalanobis_sq", "logdet"]
    joint_near_target_keys = ["joint_nlpd", "joint_mahalanobis_sq", "joint_logdet"]

    present = [k for k in joint_keys if any(has_cal(m, "joint_posterior", k) for m in methods)]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.2))
        axes = np.atleast_1d(axes)
        sample = None
        m_joint = config.get("test_grid", {}).get("m_test_joint_distrib")
        for ax, key in zip(axes, present):
            nominal = float(m_joint) if (key == "mahalanobis_sq" and m_joint) else None
            s = plot_cal_metric(ax, "joint_posterior", key, logy=(key != "logdet"), nominal=nominal)
            sample = sample or s
            ax.set_ylabel(key); ax.set_title(key.replace("_", " "))
            ax.legend(loc="best")
        fig.suptitle("Joint posterior diagnostics -- " + band_caption(sample, CAL_N))
        fig.tight_layout()
        add_figure("04_joint_posterior", fig)
    else:
        logger.info("joint posterior diagnostics not available — skipping.")

    present = [k for k in joint_near_target_keys if any(has_cal(m, "near_target", k) for m in methods)]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.2))
        axes = np.atleast_1d(axes)
        sample = None
        m_joint = config.get("test_grid", {}).get("m_test_joint_near_target")
        for ax, key in zip(axes, present):
            nominal = float(m_joint) if (key == "mahalanobis_sq" and m_joint) else None
            s = plot_cal_metric(ax, "near_target", key, logy=(key != "logdet"), nominal=nominal)
            sample = sample or s
            ax.set_ylabel(key); ax.set_title(key.replace("_", " "))
            ax.legend(loc="best")
        fig.suptitle("Near-target joint posterior diagnostics -- " + band_caption(sample, CAL_N))
        fig.tight_layout()
        add_figure("04b_near_target_joint_posterior", fig)
    else:
        logger.info("near-target joint posterior diagnostics not available — skipping.")

    # ========================================================================
    # Save all figures
    # ========================================================================
    _clear_stale_figures(FIG_DIR)
    saved = []
    for name, fig in FIGURES.items():
        png = FIG_DIR / f"{name}.png"
        pdf = FIG_DIR / f"{name}.pdf"
        fig.savefig(_long_path(png), format="png")
        fig.savefig(_long_path(pdf), format="pdf")
        plt.close(fig)
        saved.append(name)

    if agg["statistical_tests"] is not None:
        comp = agg["statistical_tests"]["comparisons"][0]
        rows = []
        for metric, payload in comp["metrics"].items():
            rows.append({
                "metric": metric,
                "wilcoxon_p": payload.get("wilcoxon_paired", {}).get("p_value"),
                "mann_whitney_p": payload.get("mann_whitney", {}).get("p_value"),
                "rank_biserial": payload.get("rank_biserial"),
                "cohens_d_right_better": payload.get("cohens_d_right_better"),
                "right_failure_rate": payload.get("right_failure_rate"),
                "wrong_failure_rate": payload.get("wrong_failure_rate"),
            })
        out_csv = FIG_DIR / "final_performance_comparison.csv"
        pd.DataFrame(rows).set_index("metric").to_csv(out_csv)
        logger.info("saved comparison table: %s", out_csv.relative_to(RUN_DIR))

    logger.info("[%s] saved %d figures to %s", BENCHMARK, len(saved), FIG_DIR)
    return len(saved)


# --- run-dir orchestration --------------------------------------------------
def _discover_processed(run_dir: Path) -> list[str]:
    """Benchmark names that have processed per-seed files under `run_dir`."""
    root = run_dir / "processed" / "per_seed"
    if not root.exists():
        raise FileNotFoundError(
            f"No processed/per_seed/ under {run_dir}.  Run process.py + analyze.py first."
        )
    out = []
    for bench_dir in sorted(root.iterdir()):
        if bench_dir.is_dir() and any(bench_dir.glob("seed_*_processed.json")):
            out.append(bench_dir.name)
    return out


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", default=None,
                        help="Filter (repeatable).  Defaults to all benchmarks.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    backend = _apply_style()
    logger.info("[style] backend = %s (usetex off, colour-blind palette active)", backend)

    args = _parse_args(argv)
    run_dir: Path = args.run_dir.resolve()

    benchmarks = _discover_processed(run_dir)
    if args.benchmark is not None:
        benchmarks = [b for b in benchmarks if b in args.benchmark]
    if not benchmarks:
        logger.warning("No processed benchmarks matched the filters; nothing to do.")
        return

    total = 0
    for benchmark in benchmarks:
        total += plot_benchmark(run_dir, benchmark)
    logger.info("Done: %d figure(s) across %d benchmark(s) in %s",
                total, len(benchmarks), run_dir)


if __name__ == "__main__":
    main()
