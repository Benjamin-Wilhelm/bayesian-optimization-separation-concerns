#!/usr/bin/env python
"""Experiment B — prior-guided BO: reproduce every analysis figure as a script.

This is the script-based counterpart of ``notebooks/Exp_B_analysis.ipynb``.  It
reads a processed run directory (``process.py`` + ``analyze.py`` must have run
first) and regenerates *all* the figures the notebook produces, saving each one
as ``.png`` and ``.pdf`` under::

    <run_dir>/figures/<benchmark>/

Usage::

    python -m experiments.exp_b_prior_guided.plot \
        --run-dir results/exp_b_prior_guided/runs/<run_id>

By default every benchmark found under ``processed/per_seed/`` is plotted; pass
``--benchmark`` (repeatable) to restrict the set.  The plotting logic is copied
from the notebook; only the run/benchmark selection and figure saving are turned
into a reusable function.  The notebook's validation-only cells (sanity checks)
and its display-only summary tables are omitted since they produce no figure.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless: never require a display / GUI backend

import numpy as np
import pandas as pd  # noqa: F401 (kept for parity with the notebook environment)
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize
from scipy.stats import norm as _norm

from common.serialization import load_json
from common.priors import GaussianPrior, PriorMeanModule
from common.statistics import bootstrap_ci_curves
from benchmarks.oriented import build_oriented_benchmark
from data.raw_prior import RawPriorSeedData
from data.processed_prior import ProcessedPriorSeedData
from experiments.exp_b_prior_guided.run import _METHOD_SPECS
from experiments.exp_b_prior_guided.process import (
    _reconstruction_config, _reconstruct_X_Y, _rebuild_gp_at_snapshot,
)
from acquisitions.prior_guided import VanillaUCB

logger = logging.getLogger("exp_b.plot")

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.double)

# --- global plotting style (copied from the notebook) -----------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Canonical colours / labels (consistent across every figure).
METHOD_STYLE = {
    "vanilla_ucb":            {"label": "vanilla UCB",            "color": "#4D4D4D"},
    "alpha_pi_ucb":           {"label": r"$\alpha\pi$-UCB (SoC)",  "color": "#0072B2"},
    "lambda_pi_fixed_ucb":    {"label": r"$\lambda\pi$ fixed",     "color": "#E69F00"},
    "lambda_pi_adaptive_ucb": {"label": r"$\lambda\pi$ adaptive",  "color": "#D55E00"},
    "lambda_pi_fixed_standardized_ucb":    {"label": r"$\lambda\pi$ fixed (std)",    "color": "#CC79A7"},
    "lambda_pi_adaptive_standardized_ucb": {"label": r"$\lambda\pi$ adaptive (std)", "color": "#009E73"},
}


def style_for(method: str) -> dict:
    return METHOD_STYLE.get(method, {"label": method, "color": None})


# --- centralized publication style (copied from the notebook) ---------------
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


def seed_band(ax, x, Y, *, color, label=None, pct=(25, 75),
              show_seeds=False, show_mean=False, lw=2.2, alpha_band=0.15):
    """Median curve + inter-seed percentile band (NaN-robust).  ``Y`` is (n_seeds, n_x)."""
    Y = np.asarray(Y, dtype=float)
    med = np.nanmedian(Y, axis=0)
    lo = np.nanpercentile(Y, pct[0], axis=0)
    hi = np.nanpercentile(Y, pct[1], axis=0)
    if show_seeds:
        for row in Y:
            ax.plot(x, row, color=color, lw=0.6, alpha=0.18, zorder=1)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, lw=0, zorder=2)
    line, = ax.plot(x, med, color=color, lw=lw, label=label, zorder=3)
    if show_mean:
        ax.plot(x, np.nanmean(Y, axis=0), color=color, lw=1.3, ls="--",
                alpha=0.9, zorder=3)
    return line


# --- single-benchmark plotting (notebook body, verbatim logic) --------------
def plot_benchmark(run_dir: Path, benchmark_name: str) -> int:
    """Reproduce and save every Exp-B figure for one benchmark.

    Returns the number of figures written.  Mirrors the notebook; every
    ``plt.show()`` is dropped and each figure is closed after saving.
    """
    RUN_DIR = run_dir
    BENCH_NAME = benchmark_name

    FIGURES: dict[str, plt.Figure] = {}

    def add_figure(name: str, fig: plt.Figure) -> plt.Figure:
        if name in FIGURES:
            logger.warning("overwriting already-registered figure '%s'", name)
        FIGURES[name] = fig
        return fig

    # ---- load config, metadata, processed seeds and aggregates --------------
    config = load_json(RUN_DIR / "config.json")

    _raw_root = RUN_DIR / "raw"
    _cfg_bench = config.get("benchmark")
    BENCH_NAME_CONFIG = _cfg_bench["name"] if isinstance(_cfg_bench, dict) else _cfg_bench  # noqa: F841
    _raw_files = sorted((_raw_root / BENCH_NAME).glob("seed_*.json"))
    SEEDS = [int(p.stem.split("_")[1]) for p in _raw_files]
    if not SEEDS:
        raise FileNotFoundError(f"No raw seed files under {_raw_root / BENCH_NAME}")

    _RAW_CACHE: dict[int, RawPriorSeedData] = {}

    def load_raw(seed: int) -> RawPriorSeedData:
        if seed not in _RAW_CACHE:
            _RAW_CACHE[seed] = RawPriorSeedData.from_dict(
                load_json(_raw_root / BENCH_NAME / f"seed_{seed:03d}.json"))
        return _RAW_CACHE[seed]

    # Representative seed: the notebook uses SEEDS[4]; fall back if fewer seeds.
    _rep_idx = min(4, len(SEEDS) - 1)
    raw0 = load_raw(SEEDS[_rep_idx])

    _per_seed_dir = RUN_DIR / "processed" / "per_seed" / BENCH_NAME
    PROCESSED = [ProcessedPriorSeedData.from_dict(load_json(p))
                 for p in sorted(_per_seed_dir.glob("seed_*_processed.json"))]

    _agg_dir = RUN_DIR / "processed" / "aggregate" / BENCH_NAME

    def _maybe_load(name: str):
        p = _agg_dir / name
        return load_json(p) if p.exists() else None

    AGG = {
        "final":        _maybe_load("final_performance.json"),
        "convergence":  _maybe_load("convergence_curves.json"),
        "calib_sum":    _maybe_load("calibration_summary.json"),
        "calib_curves": _maybe_load("calibration_curves.json"),
        "tests":        _maybe_load("statistical_tests.json"),
    }

    problem = build_oriented_benchmark(_reconstruction_config(config, raw0))
    prior = GaussianPrior.from_config(raw0.metadata.prior, dim=problem.dim)

    DIM = problem.dim
    BUDGET = raw0.metadata.budget
    N_INIT = raw0.metadata.initial_design_size
    N_EVAL = N_INIT + BUDGET
    ALPHA = float(raw0.metadata.alpha if raw0.metadata.alpha is not None else config.get("alpha", 0.5))
    BETA = float(raw0.metadata.beta if raw0.metadata.beta is not None else config["beta"])
    F_OPT = float(raw0.metadata.f_opt)
    EPSILON = PROCESSED[0].epsilon if PROCESSED else config.get("epsilon")
    PRIOR_LABEL = (raw0.metadata.prior or {}).get("label", "?")
    PI_THRESHOLD = (config.get("calibration", {}) or {}).get("prior_region_threshold", 0.5)

    METHODS = [m for m in METHOD_STYLE if m in raw0.methods] + \
              [m for m in raw0.methods if m not in METHOD_STYLE]

    BOUNDS = problem.bounds.numpy()
    LB, UB = BOUNDS[0], BOUNDS[1]

    logger.info(
        "[%s] dim=%d | methods=%s | seeds=%d | budget=%d | n_init=%d | "
        "prior='%s' | alpha=%g beta=%g f_opt=%g epsilon=%s",
        BENCH_NAME, DIM, METHODS, len(SEEDS), BUDGET, N_INIT,
        PRIOR_LABEL, ALPHA, BETA, F_OPT, EPSILON,
    )

    # ---- numerical location of the optimum x* -------------------------------
    def _f_np(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return problem.f(torch.as_tensor(np.atleast_2d(X), dtype=torch.double)).squeeze(-1).numpy()

    def _pi_np(X: np.ndarray) -> np.ndarray:
        return prior.evaluate(torch.as_tensor(np.atleast_2d(X), dtype=torch.double)).numpy()

    def locate_optimum(n_sobol: int = 2 ** (13 if DIM <= 3 else 15), n_refine: int = 12) -> np.ndarray:
        Xs = problem.test_set(n=n_sobol, seed=7).numpy()
        f = _f_np(Xs)
        starts = Xs[np.argsort(-f)[:n_refine]]
        best_x, best_f = None, -np.inf
        for x0 in starts:
            res = minimize(lambda z: -_f_np(z[None, :])[0], x0,
                           method="L-BFGS-B", bounds=list(zip(LB, UB)))
            if -res.fun > best_f:
                best_f, best_x = -res.fun, res.x
        return np.asarray(best_x)

    X_STAR = locate_optimum()
    X_PI = prior.center.numpy()

    # ---- GP / acquisition reconstruction helpers ----------------------------
    def gp_at(raw: RawPriorSeedData, method: str, iteration: int):
        traj = raw.methods[method]
        snap = next(s for s in traj.model_snapshots if s.iteration == iteration)
        prefix = traj.trajectory[: iteration - 1]
        train_X, train_Y = _reconstruct_X_Y(raw.initial_design, prefix, DEVICE)
        model = _rebuild_gp_at_snapshot(snap, traj, config, problem, prior, train_X, train_Y)
        return model, train_X, snap

    def acq_for(method: str, model, beta):
        return _METHOD_SPECS[method]["acq_builder"](model, prior, ALPHA, beta)

    def eval_acq(acq, X_np: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(np.atleast_2d(X_np), dtype=torch.double).unsqueeze(1)
        with torch.no_grad():
            return acq(X).cpu().numpy()

    def posterior_np(model, X_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = torch.as_tensor(np.atleast_2d(X_np), dtype=torch.double)
        with torch.no_grad():
            post = model.posterior(X)
            mu = post.mean.squeeze(-1).cpu().numpy()
            sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return mu, sd

    def lambda_at(raw: RawPriorSeedData, method: str, iteration: int) -> Optional[float]:
        step = raw.methods[method].trajectory[iteration - 1]
        return step.lambda_value

    def beta_at_iter(raw: RawPriorSeedData, method: str, iteration: int) -> float:
        """Per-iteration beta_t actually used to select the candidate at `iteration`
        (1-based).  With an adaptive beta_schedule beta_t varies across iterations;
        falls back to the scalar BETA for legacy runs without a stored beta_value."""
        step = raw.methods[method].trajectory[iteration - 1]
        b = getattr(step, "beta_value", None)
        return float(b) if b is not None else BETA

    def queries_np(raw: RawPriorSeedData, method: str) -> np.ndarray:
        return np.asarray([s.x for s in raw.methods[method].trajectory], dtype=float)

    X_INIT = np.asarray(raw0.initial_design.X, dtype=float)

    # ---- grids / slices ------------------------------------------------------
    def grid_1d(n: int = 800) -> np.ndarray:
        return np.linspace(LB[0], UB[0], n)[:, None]

    def grid_2d(n: int = 160):
        g1 = np.linspace(LB[0], UB[0], n)
        g2 = np.linspace(LB[1], UB[1], n)
        G1, G2 = np.meshgrid(g1, g2)
        XY = np.stack([G1.ravel(), G2.ravel()], axis=-1)
        return G1, G2, XY

    def slice_line(n: int = 400, extend: float = 0.15):
        s = np.linspace(-extend, 1.0 + extend, n)
        X = (1.0 - s)[:, None] * X_STAR[None, :] + s[:, None] * X_PI[None, :]
        X = np.clip(X, LB, UB)
        return s, X

    # ========================================================================
    # 1. Benchmark and prior visualization
    # ========================================================================
    if DIM == 1:
        xs = grid_1d()
        f_vals, pi_vals = _f_np(xs), _pi_np(xs)
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        axp = ax.twinx()
        lf, = ax.plot(xs[:, 0], f_vals, color="black", lw=1.6, label=r"objective $f(x)$ (raw)")
        lp, = axp.plot(xs[:, 0], pi_vals, color="#2ca02c", lw=1.4, ls="--", label=r"prior $\pi(x)$")
        mask = pi_vals >= PI_THRESHOLD
        axp.fill_between(xs[:, 0], 0, 1, where=mask, color="#2ca02c", alpha=0.08,
                         label=rf"prior region $\pi \geq {PI_THRESHOLD:g}$")
        ax.axvline(X_STAR[0], color="black", ls=":", lw=1.2)
        ls, = ax.plot(X_STAR[0], F_OPT, marker="*", ms=16, color="gold", mec="black", ls="none",
                      label=r"optimum $x^\star$", zorder=5)
        lc, = ax.plot(X_PI[0], _f_np(X_PI[None])[0], marker="v", ms=10, color="#2ca02c",
                      mec="black", ls="none", label=r"prior center $x_\pi$", zorder=5)
        li, = ax.plot(X_INIT[:, 0], _f_np(X_INIT), marker="o", ms=6, color="white", mec="black",
                      ls="none", label=f"initial design (seed {SEEDS[0]})")
        ax.set_xlabel("x"); ax.set_ylabel(r"objective $f$ (raw scale)")
        axp.set_ylabel(r"prior $\pi(x) \in [0, 1]$"); axp.set_ylim(-0.05, 1.05)
        ax.set_title("Objective and prior")
        handles = [lf, lp, ls, lc, li]
        ax.legend(handles, [h.get_label() for h in handles], loc="best", ncol=2)
        fig.tight_layout(); add_figure("01_benchmark_and_prior", fig)

    elif DIM == 2:
        G1, G2, XY = grid_2d()
        F = _f_np(XY).reshape(G1.shape)
        PI = _pi_np(XY).reshape(G1.shape)
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        im = ax.contourf(G1, G2, F, levels=30, cmap="viridis")
        fig.colorbar(im, ax=ax, label=r"$f(x)$ (raw)")
        cs = ax.contour(G1, G2, PI, levels=[0.1, PI_THRESHOLD, 0.9], colors="white",
                        linewidths=1.0, linestyles="--")
        ax.clabel(cs, fmt=lambda v: rf"$\pi$={v:g}", fontsize=8)
        ax.plot(*X_STAR, marker="*", ms=18, color="gold", mec="black", ls="none",
                label=r"$x^\star$", zorder=5)
        ax.plot(*X_PI, marker="v", ms=11, color="#2ca02c", mec="black", ls="none",
                label=r"$x_\pi$", zorder=5)
        ax.plot(X_INIT[:, 0], X_INIT[:, 1], marker="o", ms=6, color="white", mec="black",
                ls="none", label=f"initial design (seed {SEEDS[0]})")
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.grid(False)
        ax.set_title("Objective and prior")
        ax.legend(loc="best")
        fig.tight_layout(); add_figure("01_benchmark_and_prior", fig)

    else:
        s, Xs = slice_line()
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
        ax = axes[0]; axp = ax.twinx()
        lf, = ax.plot(s, _f_np(Xs), color="black", lw=1.6, label=r"$f(x(s))$ (raw)")
        lp, = axp.plot(s, _pi_np(Xs), color="#2ca02c", lw=1.4, ls="--", label=r"$\pi(x(s))$")
        ax.axvline(0, color="black", ls=":", lw=1.0); ax.axvline(1, color="#2ca02c", ls=":", lw=1.0)
        ax.set_xlabel(r"$s$  with  $x(s) = (1-s)\,x^\star + s\,x_\pi$")
        ax.set_ylabel(r"$f$ (raw)"); axp.set_ylabel(r"$\pi \in [0, 1]$"); axp.set_ylim(-0.05, 1.05)
        ax.set_title("Optimum-to-prior slice")
        ax.legend([lf, lp], [lf.get_label(), lp.get_label()])
        ax = axes[1]
        g = np.linspace(0, 1, 250)
        for j in range(DIM):
            Xj = np.repeat(X_STAR[None, :], len(g), axis=0)
            Xj[:, j] = LB[j] + g * (UB[j] - LB[j])
            ax.plot(LB[j] + g * (UB[j] - LB[j]), _f_np(Xj), lw=1.1, label=rf"$x_{{{j+1}}}$")
        ax.set_xlabel("coordinate value"); ax.set_title(r"Coordinate slices of $f$ through $x^\star$")
        ax.legend(ncol=2, fontsize=8)
        fig.suptitle("Benchmark and prior", y=1.02)
        fig.tight_layout(); add_figure("01_benchmark_and_prior", fig)

    # ========================================================================
    # 2. Representative-seed qualitative analysis
    # ========================================================================
    REPRESENTATIVE_SEED = SEEDS[_rep_idx]
    POSTERIOR_ITERS = [1, 2, 5, 6, 7, 8, 9]
    POSTERIOR_ITERS = sorted({min(t, BUDGET) for t in POSTERIOR_ITERS if t >= 1})
    raw_rep = load_raw(REPRESENTATIVE_SEED)

    # ---- 2.1 posterior and acquisition evolution ----------------------------
    def _plot_posterior_acq_curves(raw, method, iters, X_eval, x_axis, x_axis_label,
                                   x_star_coord, x_pi_coord, queries_proj):
        st = style_for(method)
        f_curve = _f_np(X_eval)
        pi_curve = _pi_np(X_eval)
        n_c = len(iters)
        fig, axes = plt.subplots(2, n_c, figsize=(3.1 * n_c, 5.6),
                                 sharex=True, squeeze=False)
        for k, t in enumerate(iters):
            model, train_X, snap = gp_at(raw, method, t)
            mu, sd = posterior_np(model, X_eval)
            beta_t = beta_at_iter(raw, method, t)
            acq = acq_for(method, model, beta_t)
            a_vals = eval_acq(acq, X_eval)
            x_next = np.asarray(raw.methods[method].trajectory[t - 1].x)
            lam = lambda_at(raw, method, t)

            ax = axes[0, k]
            ax.plot(x_axis, f_curve, color="black", lw=1.0, alpha=0.7, label=r"$f$")
            ax.plot(x_axis, mu, color=st["color"], lw=1.5, label=r"$\mu_t$")
            ax.fill_between(x_axis, mu - math.sqrt(beta_t) * sd, mu + math.sqrt(beta_t) * sd,
                            color=st["color"], alpha=0.18, lw=0,
                            label=r"$\mu_t \pm \sqrt{\beta_t}\sigma_t$")
            if lam is not None:
                ax.plot(x_axis, lam * pi_curve, color="#2ca02c", ls="--", lw=1.2,
                        label=rf"$\hat\lambda_t\,\pi$ ($\hat\lambda$={lam:.2f})")
            if queries_proj is not None:
                past = queries_proj(train_X.numpy())
                ax.plot(past, _f_np(train_X.numpy()), "o", ms=4, color="white",
                        mec="black", label="queried")
            ax.axvline(x_star_coord, color="gold", ls=":", lw=1.2)
            nxt = queries_proj(x_next[None, :])[0] if queries_proj is not None else None
            if nxt is not None:
                ax.axvline(nxt, color=st["color"], ls="-", lw=0.8, alpha=0.6)
                ax.plot(nxt, _f_np(x_next[None])[0], "*", ms=13, color=st["color"],
                        mec="black", label="next query")
            ax.set_title(f"t = {t}", fontsize=10)
            if k == 0:
                ax.set_ylabel("posterior")
                ax.legend(fontsize=7, loc="best")

            ax = axes[1, k]
            ax.plot(x_axis, a_vals, color=st["color"], lw=1.5, label="acquisition")
            if method == "alpha_pi_ucb":
                ref = eval_acq(VanillaUCB(model, beta=beta_t), X_eval)
                ax.plot(x_axis, ref, color="grey", ls="--", lw=1.0, label="vanilla UCB (same GP)")
            if nxt is not None:
                ax.axvline(nxt, color=st["color"], ls="-", lw=0.8, alpha=0.6)
            ax.axvline(x_pi_coord, color="#2ca02c", ls=":", lw=1.0)
            ax.set_xlabel(x_axis_label)
            if k == 0:
                ax.set_ylabel("acquisition")
                ax.legend(fontsize=7, loc="best")
        fig.suptitle(st["label"], y=1.0)
        fig.tight_layout()
        return fig

    if DIM == 1:
        X_eval = grid_1d(500)
        for method in METHODS:
            fig = _plot_posterior_acq_curves(
                raw_rep, method, POSTERIOR_ITERS, X_eval, X_eval[:, 0], "x",
                X_STAR[0], X_PI[0], queries_proj=lambda X: X[:, 0])
            add_figure(f"02_posterior_acq_{method}", fig)
    elif DIM > 2:
        s, X_eval = slice_line(300)
        for method in METHODS:
            fig = _plot_posterior_acq_curves(
                raw_rep, method, POSTERIOR_ITERS, X_eval, s,
                r"$s$  ($x(s) = (1-s)x^\star + s\,x_\pi$)", 0.0, 1.0,
                queries_proj=None)
            add_figure(f"02_posterior_acq_slice_{method}", fig)
    else:
        logger.info("d = 2: 1D posterior curves skipped (see spatial views 2.2/2.3).")

    # ---- 2.2 query trajectory (spatial view, d = 2) -------------------------
    if DIM == 2:
        G1, G2, XY = grid_2d()
        F = _f_np(XY).reshape(G1.shape)
        PI = _pi_np(XY).reshape(G1.shape)
        n_m = len(METHODS)
        ncols = 2; nrows = (n_m + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.0 * nrows), squeeze=False)
        for i, method in enumerate(METHODS):
            ax = axes[i // ncols, i % ncols]
            st = style_for(method)
            ax.contourf(G1, G2, F, levels=30, cmap="viridis")
            ax.contour(G1, G2, PI, levels=[0.1, PI_THRESHOLD, 0.9], colors="white",
                       linewidths=0.8, linestyles="--")
            Q = queries_np(raw_rep, method)
            sc = ax.scatter(Q[:, 0], Q[:, 1], c=np.arange(1, len(Q) + 1), cmap="autumn",
                            s=34, edgecolors="black", linewidths=0.5, zorder=4)
            ax.plot(X_INIT[:, 0], X_INIT[:, 1], "o", ms=6, color="white", mec="black", zorder=3)
            ax.plot(*X_STAR, marker="*", ms=16, color="gold", mec="black", zorder=5)
            ax.plot(*X_PI, marker="v", ms=10, color="#2ca02c", mec="black", zorder=5)
            ax.annotate("1", Q[0], textcoords="offset points", xytext=(5, 5), color="white", fontsize=8)
            ax.annotate(str(len(Q)), Q[-1], textcoords="offset points", xytext=(5, 5),
                        color="white", fontsize=8)
            ax.set_title(st["label"]); ax.grid(False)
            ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
            fig.colorbar(sc, ax=ax, label="iteration")
        for j in range(n_m, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        fig.suptitle("Query trajectories", y=1.0)
        fig.tight_layout(); add_figure("02_query_trajectories_2d", fig)

    # ---- 2.3 posterior / acquisition snapshots (d = 2) ----------------------
    if DIM == 2:
        SNAPSHOT_ITERS_2D = sorted({POSTERIOR_ITERS[0],
                                    POSTERIOR_ITERS[len(POSTERIOR_ITERS) // 2],
                                    POSTERIOR_ITERS[-1]})
        G1s, G2s, XYs = grid_2d(90)
        for method in METHODS:
            st = style_for(method)
            n_r = len(SNAPSHOT_ITERS_2D)
            fig, axes = plt.subplots(n_r, 4, figsize=(16.0, 3.6 * n_r), squeeze=False)
            for r, t in enumerate(SNAPSHOT_ITERS_2D):
                model, train_X, _ = gp_at(raw_rep, method, t)
                mu, sd = posterior_np(model, XYs)
                beta_t = beta_at_iter(raw_rep, method, t)
                ucb = mu + math.sqrt(beta_t) * sd
                a_vals = eval_acq(acq_for(method, model, beta_t), XYs)
                x_next = np.asarray(raw_rep.methods[method].trajectory[t - 1].x)
                panels = [(mu, r"$\mu_t$", "viridis"), (sd, r"$\sigma_t$", "magma"),
                          (ucb, r"UCB $\mu_t+\sqrt{\beta_t}\sigma_t$", "cividis"),
                          (a_vals, "actual acquisition", "cividis")]
                for c, (Z, title, cmap) in enumerate(panels):
                    ax = axes[r, c]
                    im = ax.contourf(G1s, G2s, Z.reshape(G1s.shape), levels=25, cmap=cmap)
                    fig.colorbar(im, ax=ax)
                    Xq = train_X.numpy()
                    ax.plot(Xq[:, 0], Xq[:, 1], "o", ms=3.5, color="white", mec="black", lw=0.4)
                    ax.plot(*x_next, marker="*", ms=12, color="red", mec="black")
                    ax.plot(*X_STAR, marker="*", ms=10, color="gold", mec="black")
                    ax.plot(*X_PI, marker="v", ms=8, color="#2ca02c", mec="black")
                    ax.set_title(f"t={t}: {title}", fontsize=10); ax.grid(False)
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(st["label"], y=1.0)
            fig.tight_layout(); add_figure(f"02_snapshots_2d_{method}", fig)

    # ---- 2.4 query diagnostics in high dimension (d > 2) --------------------
    if DIM > 2:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
        for method in METHODS:
            st = style_for(method)
            Q = queries_np(raw_rep, method)
            t_ax = np.arange(1, len(Q) + 1)
            axes[0].plot(t_ax, np.linalg.norm(Q - X_STAR, axis=1), color=st["color"],
                         lw=1.4, label=st["label"])
            axes[1].plot(t_ax, np.linalg.norm(Q - X_PI, axis=1), color=st["color"], lw=1.4)
        axes[0].set_ylabel(r"$\Vert x_t - x^\star \Vert$"); axes[0].set_xlabel("BO iteration $t$")
        axes[0].set_title("Distance to optimum"); axes[0].legend(fontsize=8)
        axes[1].set_ylabel(r"$\Vert x_t - x_\pi \Vert$"); axes[1].set_xlabel("BO iteration $t$")
        axes[1].set_title("Distance to prior center")
        fig.suptitle("Query distances", y=1.02)
        fig.tight_layout(); add_figure("02_query_distances_seed", fig)

        pooled = np.concatenate([queries_np(raw_rep, m) for m in METHODS], axis=0)
        mu_p = pooled.mean(axis=0)
        _, _, Vt = np.linalg.svd(pooled - mu_p, full_matrices=False)
        P = Vt[:2].T
        fig, axes = plt.subplots(1, len(METHODS), figsize=(3.4 * len(METHODS), 3.6),
                                 sharex=True, sharey=True, squeeze=False)
        for i, method in enumerate(METHODS):
            ax = axes[0, i]
            Z = (queries_np(raw_rep, method) - mu_p) @ P
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(1, len(Z) + 1), cmap="autumn",
                            s=26, edgecolors="black", linewidths=0.4)
            ax.plot(*((X_STAR - mu_p) @ P), marker="*", ms=14, color="gold", mec="black")
            ax.plot(*((X_PI - mu_p) @ P), marker="v", ms=9, color="#2ca02c", mec="black")
            ax.set_title(style_for(method)["label"], fontsize=10)
            ax.set_xlabel("PC 1")
            if i == 0:
                ax.set_ylabel("PC 2")
        fig.colorbar(sc, ax=axes[0, -1], label="iteration")
        fig.suptitle("PCA projection of queries", y=1.04)
        fig.tight_layout(); add_figure("02_query_pca_seed", fig)

    # ========================================================================
    # 3. Aggregate optimization performance
    # ========================================================================
    CONV = AGG["convergence"]

    # Convergence bands: bootstrap CI of the median (publication-style, and
    # consistent with Exp A).  Catastrophic / trapped-seed information is not
    # lost: it is carried by the dashed mean line (which diverges from the
    # median under bimodality), the final-regret distributions (3.4) and the
    # hitting-time / failure-rate figure (3.5).
    _CI_CONF  = float(CONV.get("confidence", 0.95)) if isinstance(CONV, dict) else 0.95
    _CI_NBOOT = int(CONV.get("n_bootstrap", 10000)) if isinstance(CONV, dict) else 10000
    _LOG_FLOOR = 1e-7

    CONV_KEYS = {
        "simple_regret":            r"$r_t^{\mathrm{best}} = f(x^\star) - \max_{s\leq t} f(x_s)$",
        "instantaneous_regret":     r"$r_t = f(x^\star) - f(x_t)$",
        "cumulative_inst_regret":   r"$R_t = \sum_{s\leq t}\,[f(x^\star) - f(x_s)]$",
        "running_best_value":       r"$\max_{s\leq t} f(x_s)$",
        "prior_value_at_candidate": r"$\pi(x_t)$ (BO-iteration axis)",
        "lambda_per_iteration":     r"$\hat\lambda_t$ (BO-iteration axis)",
    }
    _EVAL_KEYS = {"simple_regret", "instantaneous_regret",
                  "cumulative_inst_regret", "running_best_value"}

    if not PROCESSED:
        logger.warning("[%s] no processed per-seed files — section 3+ skipped.", BENCH_NAME)
    else:
        EVAL_AXIS = np.asarray(PROCESSED[0].methods[METHODS[0]].convergence.iteration, float)
        BO_AXIS = np.asarray(PROCESSED[0].methods[METHODS[0]].convergence.bo_iteration, float)

        def seed_curves(method: str, key: str):
            if key not in CONV_KEYS:
                raise KeyError(f"'{key}' is not a known curve key.  Known: {sorted(CONV_KEYS)}")
            rows = [getattr(p.methods[method].convergence, key) for p in PROCESSED
                    if getattr(p.methods[method].convergence, key) is not None]
            return np.asarray(rows, dtype=float) if rows else None

        def plot_conv_metric(ax, key: str, logy: bool = False):
            x = EVAL_AXIS if key in _EVAL_KEYS else BO_AXIS
            for method in METHODS:
                A = seed_curves(method, key)
                if A is None:
                    continue
                med, mean = np.median(A, 0), np.mean(A, 0)
                ci = bootstrap_ci_curves(A, statistic="median",
                                         confidence=_CI_CONF, n_bootstrap=_CI_NBOOT,
                                         random_state=0)
                lo, hi = ci["lower"], ci["upper"]
                if logy:
                    med, mean = np.maximum(med, _LOG_FLOOR), np.maximum(mean, _LOG_FLOOR)
                    lo, hi = np.maximum(lo, _LOG_FLOOR), np.maximum(hi, _LOG_FLOOR)
                st = style_for(method)
                ax.plot(x, med, color=st["color"], lw=2.4, label=st["label"])
                ax.plot(x, mean, color=st["color"], lw=1.3, ls="--", alpha=0.85)
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14, lw=0)
            if logy:
                ax.set_yscale("log")
            ax.axvline(N_INIT + 0.5, color="grey", ls=":", lw=1.0)
            ax.set_xlabel("evaluation $t$ (initial design + BO)")
            ax.set_ylabel(CONV_KEYS[key])
            return ax

        def conv_band(method: str, key: str):
            entry = (CONV or {}).get("methods", {}).get(method, {}).get(key)
            if entry is None:
                return None
            if "pointwise" in entry:
                entry = entry["pointwise"]
            return entry

        # 3.1 best-so-far regret
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        plot_conv_metric(ax, "simple_regret", logy=True)
        if EPSILON is not None:
            ax.axhline(EPSILON, color="grey", ls="--", lw=1.0,
                       label=rf"hitting tolerance $\varepsilon = {EPSILON:g}$")
        ax.set_title("Best-so-far regret")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); add_figure("03_best_so_far_regret", fig)

        # 3.2 instantaneous regret
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        plot_conv_metric(ax, "instantaneous_regret", logy=False)
        ax.set_title("Instantaneous regret")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); add_figure("03_instantaneous_regret", fig)

        # 3.3 cumulative regret
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        plot_conv_metric(ax, "cumulative_inst_regret", logy=False)
        ax.set_title("Cumulative regret")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); add_figure("03_cumulative_regret", fig)

        # 3.4 final-regret distributions across seeds (severity of outcomes;
        # complements the hitting-time / failure-rate view below, which shows
        # only frequency).  Boxplot + per-seed points, one panel per regret.
        _final_specs = [
            ("simple_regret",          "best-so-far regret", r"$r_N^{\mathrm{best}}$"),
            ("cumulative_inst_regret", "cumulative regret",   r"$R_N$"),
        ]
        if any(seed_curves(m, k) is not None
               for m in METHODS for k, _, _ in _final_specs):
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
            _rng = np.random.default_rng(0)
            for ax, (key, title, ylab) in zip(axes, _final_specs):
                data, labels, colors = [], [], []
                for m in METHODS:
                    A = seed_curves(m, key)
                    if A is None:
                        continue
                    finals = A[:, -1]
                    finals = finals[np.isfinite(finals)]
                    data.append(finals)
                    labels.append(style_for(m)["label"]); colors.append(style_for(m)["color"])
                if not data:
                    continue
                bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                                showfliers=False, whis=(5, 95),
                                showmeans=True, meanline=True,
                                medianprops=dict(color="black", lw=1.4),
                                meanprops=dict(color="black", ls="--", lw=1.4))
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c); patch.set_alpha(0.4)
                for i, d in enumerate(data):
                    ax.plot(np.full(len(d), i + 1) + _rng.uniform(-0.08, 0.08, len(d)),
                            d, "o", ms=4, color=colors[i], alpha=0.6)
                ax.set_ylabel(ylab)
                ax.set_title(f"Final {title}")
                ax.tick_params(axis="x", rotation=12)
            # box: 25-75%, whiskers: 5th-95th percentile; solid = median, dashed = mean
            from matplotlib.lines import Line2D
            _dist_handles = [Line2D([0], [0], color="black", lw=1.4, label="median"),
                             Line2D([0], [0], color="black", lw=1.4, ls="--", label="mean")]
            axes[0].legend(handles=_dist_handles, loc="best", fontsize=8)
            fig.suptitle("Final-regret distributions across seeds\n"
                         "box: 25-75%, whiskers: 5-95%, solid: median, dashed: mean")
            fig.tight_layout(); add_figure("03_final_regret_distributions", fig)

        # 3.5 hitting time and failure rate
        if EPSILON is None:
            logger.info("no epsilon configured — hitting times not computed.")
        else:
            rows = []
            for psd in PROCESSED:
                for m in METHODS:
                    ht = psd.methods[m].final.hitting_time
                    if ht is not None:
                        rows.append({"method": m, "seed": psd.metadata.get("seed"),
                                     "hit": ht <= N_EVAL, "hitting_time": ht})
            ht_df = pd.DataFrame(rows)
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
            data, labels, colors = [], [], []
            for m in METHODS:
                sub = ht_df[(ht_df.method == m) & ht_df.hit]
                data.append(sub.hitting_time.values)
                labels.append(style_for(m)["label"]); colors.append(style_for(m)["color"])
            bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.4)
            for i, d in enumerate(data):
                axes[0].plot(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)),
                             d, "o", ms=4, color=colors[i], alpha=0.6)
            axes[0].set_ylabel(rf"$\tau_\varepsilon$  ($\varepsilon = {EPSILON:g}$)")
            axes[0].set_title("Hitting time")
            axes[0].tick_params(axis="x", rotation=12)
            fail = [1.0 - ht_df[ht_df.method == m].hit.mean() for m in METHODS]
            axes[1].bar(range(len(METHODS)), fail, color=colors, alpha=0.7, tick_label=labels)
            axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("failure rate")
            axes[1].set_title("Failure rate")
            axes[1].tick_params(axis="x", rotation=12)
            for i, fr in enumerate(fail):
                axes[1].text(i, fr + 0.02, f"{fr:.0%}", ha="center", fontsize=9)
            fig.tight_layout(); add_figure("03_hitting_time_failure", fig)

    # ========================================================================
    # 4. Mechanism diagnostics
    # ========================================================================
    # 4.1 adaptive lambda_t trajectory
    _lambda_methods = [m for m in METHODS
                       if any(psd.methods[m].convergence.lambda_per_iteration is not None
                              for psd in PROCESSED)] if PROCESSED else []
    if _lambda_methods:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        bo_x = np.asarray(PROCESSED[0].methods[_lambda_methods[0]].convergence.bo_iteration, float)
        for m in _lambda_methods:
            st = style_for(m)
            for psd in PROCESSED:
                lam = psd.methods[m].convergence.lambda_per_iteration
                if lam is not None:
                    ax.plot(bo_x, lam, color=st["color"], lw=0.7, alpha=0.25)
            entry = conv_band(m, "lambda_per_iteration") if CONV is not None else None
            if entry is not None:
                ax.plot(bo_x, entry["center"], color=st["color"], lw=2.2, label=st["label"] + " (median)")
                if entry.get("lower") is not None and entry["lower"][0] is not None:
                    ax.fill_between(bo_x, entry["lower"], entry["upper"],
                                    color=st["color"], alpha=0.15, lw=0)
        ax.axhline(1.0, color="grey", ls="--", lw=1.0, label=r"$\lambda = 1$ (fixed method)")
        ax.axhline(0.0, color="grey", ls=":", lw=1.0, label=r"$\lambda = 0$ (prior rejected)")
        ax.set_xlabel("BO iteration $t$"); ax.set_ylabel(r"$\hat\lambda_t$")
        ax.set_title("Adaptive prior-mean coefficient")
        ax.legend(loc="best")
        fig.tight_layout(); add_figure("04_lambda_trajectory", fig)
    else:
        logger.info("no prior-mean method with stored lambda trajectories in this run.")

    # 4.2 self-confirmation diagnostics for standardized prior-mean methods
    N_REF = 12
    MAX_DIAG_SEEDS = min(15, len(SEEDS))
    ITER_STRIDE = 1
    N_GRID = 256
    _JIT = 1e-8

    def _is_std_prior_mean(m: str) -> bool:
        spec = _METHOD_SPECS.get(m, {})
        return spec.get("learn_lambda") is not None and bool(spec.get("standardize_prior_mean", False))

    _diag_methods = [m for m in METHODS if _is_std_prior_mean(m)]

    def _solve_quad(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * torch.linalg.solve(C, b.unsqueeze(-1)).squeeze(-1)).sum())

    def selfconf_diag_at(raw, method: str, t: int, Z_list, G_native):
        model, train_X, _ = gp_at(raw, method, t)
        mean = model.mean_module
        if not isinstance(mean, PriorMeanModule):
            return None
        ot = model.outcome_transform
        s_t = float(ot.stdvs.detach().flatten()[0])
        m_t = float(ot.means.detach().flatten()[0])
        if not np.isfinite(s_t) or s_t <= 0:
            return None
        var_std = float(model.likelihood.noise.detach().mean())
        it = getattr(model, "input_transform", None)

        def _mats(S):
            St = torch.as_tensor(np.atleast_2d(S), dtype=torch.double)
            Sn = it.transform(St) if it is not None else St
            with torch.no_grad():
                K = model.covar_module(Sn).to_dense()
                p_std = mean._f_prior(Sn)
                p_raw = mean.baseline + mean.amplitude * mean._pi(Sn)
            n = K.shape[0]
            jit = _JIT * float(K.diagonal().abs().mean()) + 1e-12
            C = K + (var_std + jit) * torch.eye(n, dtype=torch.double)
            f_raw = torch.as_tensor(_f_np(np.atleast_2d(S)), dtype=torch.double)
            return C, p_std, p_raw, f_raw

        Cx, px_s, px_r, frx = _mats(train_X.cpu().numpy())
        Cx_raw = s_t ** 2 * Cx
        fsx = (frx - m_t) / s_t
        d_s = _solve_quad(Cx, px_s, px_s)
        lam_s = _solve_quad(Cx, px_s, fsx) / d_s if d_s > 0 else float("nan")
        d_r = _solve_quad(Cx_raw, px_r, px_r)
        lam_r = _solve_quad(Cx_raw, px_r, frx) / d_r if d_r > 0 else float("nan")
        q_s_X = _solve_quad(Cx, fsx - lam_s * px_s, fsx - lam_s * px_s)
        q_r_X = _solve_quad(Cx_raw, frx - lam_r * px_r, frx - lam_r * px_r)
        n = frx.numel()

        q_s_Z, q_r_Z = [], []
        for Z in Z_list:
            Cz, pz_s, pz_r, frz = _mats(Z)
            Cz_raw = s_t ** 2 * Cz
            fsz = (frz - m_t) / s_t
            q_s_Z.append(_solve_quad(Cz, fsz - lam_s * pz_s, fsz - lam_s * pz_s))
            q_r_Z.append(_solve_quad(Cz_raw, frz - lam_r * pz_r, frz - lam_r * pz_r))
        med_s_Z = float(np.median(q_s_Z)) if q_s_Z else float("nan")
        med_r_Z = float(np.median(q_r_Z)) if q_r_Z else float("nan")

        Cg, _pg_s, pg_r, frg = _mats(G_native)
        q_r_G = _solve_quad(s_t ** 2 * Cg, frg - lam_r * pg_r, frg - lam_r * pg_r)
        ng = frg.numel()

        return dict(
            n=n, lam_std=lam_s, lam_raw=lam_r,
            rho_std=q_s_X / med_s_Z if med_s_Z > 0 else float("nan"),
            rho_raw=q_r_X / med_r_Z if med_r_Z > 0 else float("nan"),
            q_raw_X_pn=q_r_X / n, q_raw_Z_pn=med_r_Z / n, q_raw_G_pn=q_r_G / ng,
        )

    if not _diag_methods:
        logger.info("no standardized prior-mean method — self-confirmation diagnostic skipped.")
    else:
        from matplotlib.lines import Line2D
        _diag_seeds = SEEDS[:MAX_DIAG_SEEDS]
        _snap_iters = sorted({s.iteration for s in raw0.methods[_diag_methods[0]].model_snapshots})
        DIAG_ITERS = _snap_iters[::ITER_STRIDE]
        _G = problem.test_set(n=N_GRID, seed=2024).cpu().numpy()
        _refs = {t: [problem.test_set(n=N_INIT + (t - 1), seed=10_000 + j).cpu().numpy()
                     for j in range(N_REF)] for t in DIAG_ITERS}
        _KEYS = ("n", "lam_std", "lam_raw", "rho_std", "rho_raw",
                 "q_raw_X_pn", "q_raw_Z_pn", "q_raw_G_pn")
        diag = {m: {k: np.full((len(_diag_seeds), len(DIAG_ITERS)), np.nan) for k in _KEYS}
                for m in _diag_methods}
        for si, seed in enumerate(_diag_seeds):
            rr = load_raw(seed)
            for m in _diag_methods:
                for ti, t in enumerate(DIAG_ITERS):
                    try:
                        d = selfconf_diag_at(rr, m, int(t), _refs[t], _G)
                    except (StopIteration, RuntimeError, ValueError):
                        d = None
                    if d is None:
                        continue
                    for k in _KEYS:
                        diag[m][k][si, ti] = d[k]
        logger.info("self-confirmation diagnostic: methods=%s, %d seeds x %d iters, N_REF=%d, |G|=%d.",
                    _diag_methods, len(_diag_seeds), len(DIAG_ITERS), N_REF, N_GRID)

        tt = np.asarray(DIAG_ITERS, dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
        for m in _diag_methods:
            st = style_for(m)
            D = diag[m]
            seed_band(axes[0, 0], tt, D["rho_std"], color=st["color"], label=st["label"])
            lam_model = np.asarray(
                [[(lambda_at(load_raw(s), m, int(t)) if lambda_at(load_raw(s), m, int(t)) is not None else np.nan)
                  for t in DIAG_ITERS] for s in _diag_seeds], dtype=float)
            seed_band(axes[0, 1], tt, lam_model, color=st["color"], label=st["label"])
            axes[0, 1].plot(tt, np.nanmedian(D["lam_std"], axis=0), color=st["color"],
                            lw=1.4, ls="--", alpha=0.85)
            seed_band(axes[1, 0], tt, D["q_raw_X_pn"], color=st["color"], label=st["label"])
            axes[1, 0].plot(tt, np.nanmedian(D["q_raw_Z_pn"], axis=0), color=st["color"],
                            lw=1.3, ls="--", alpha=0.85)
            axes[1, 0].plot(tt, np.nanmedian(D["q_raw_G_pn"], axis=0), color=st["color"],
                            lw=1.3, ls=":", alpha=0.85)
            seed_band(axes[1, 1], tt, D["rho_raw"], color=st["color"], label=st["label"])

        axes[0, 0].axhline(1.0, color="grey", ls="--", lw=1.0)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylabel(r"$\tilde\rho_t^{\mathrm{res}}$")
        axes[0, 0].set_title(r"A - internal standardized residual ratio")
        axes[0, 1].axhline(1.0, color="grey", ls=":", lw=1.0)
        axes[0, 1].axhline(0.0, color="grey", ls=":", lw=1.0)
        axes[0, 1].set_ylabel(r"$\lambda_t$")
        axes[0, 1].set_title(r"A - learned $\hat\lambda_t$ (solid) vs GLS $\tilde\lambda_{X_t}$ (dashed)")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_ylabel(r"$q^{\mathrm{raw}}(S)/|S|$")
        axes[1, 0].set_title(r"B - raw ex-post residual energy per point")
        axes[1, 1].axhline(1.0, color="grey", ls="--", lw=1.2)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_ylabel(r"$\rho_t^{\mathrm{raw}}$")
        axes[1, 1].set_title(r"B - raw ex-post residual ratio")

        _proxies = [Line2D([0], [0], color="grey", ls=s) for s in ("-", "--", ":")]
        axes[1, 0].legend(_proxies, [r"queried $X_t$", r"ref. median $Z_t$", r"dense grid $G$"],
                          fontsize=7, loc="best")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
            ax.legend(loc="best", fontsize=8)
        for ax in axes.ravel():
            ax.set_xlabel(r"BO iteration $t$")
        fig.suptitle("Self-confirmation diagnostics (standardized prior-mean methods)", y=1.005)
        fig.tight_layout()
        add_figure("04_selfconfirmation_std", fig)

    # 4.3 GP hyperparameter trajectories
    MAX_RAW_SEEDS = 30
    _hp_seeds = SEEDS[:MAX_RAW_SEEDS]

    def _hp_value(hp: dict, key: str) -> float:
        v = hp.get(key)
        if v is None:
            return np.nan
        if key == "lengthscale" and isinstance(v, list):
            arr = np.asarray(v, dtype=float).ravel()
            return float(np.exp(np.mean(np.log(np.maximum(arr, 1e-12)))))
        return float(v)

    HP_KEYS = ["lengthscale", "outputscale", "noise_variance_standardized", "lambda"]
    hp_data: dict[tuple, np.ndarray] = {}
    for m in METHODS:
        per_key = {k: [] for k in HP_KEYS}
        for seed in _hp_seeds:
            raw = load_raw(seed)
            snaps = raw.methods[m].model_snapshots
            for k in HP_KEYS:
                per_key[k].append([_hp_value(s.hyperparameters, k) for s in snaps])
        for k in HP_KEYS:
            hp_data[(m, k)] = np.asarray(per_key[k], dtype=float)

    _panels = []
    for k, label in [("lengthscale", "lengthscale (geo-mean over dims)"),
                     ("outputscale", "outputscale"),
                     ("noise_variance_standardized", "noise variance"),
                     ("lambda", r"$\hat\lambda_t$")]:
        arrs = [hp_data[(m, k)] for m in METHODS]
        finite = [a for a in arrs if np.isfinite(a).any()]
        if not finite:
            continue
        if k == "noise_variance_standardized":
            allv = np.concatenate([a[np.isfinite(a)] for a in finite])
            if allv.size and np.allclose(allv, allv.flat[0]):
                logger.info("noise variance fixed at %.2e — panel skipped", allv.flat[0])
                continue
        _panels.append((k, label))

    if _panels:
        ncols = 2; nrows = (len(_panels) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 3.8 * nrows), squeeze=False)
        bo_x = np.arange(1, BUDGET + 1)
        for i, (k, label) in enumerate(_panels):
            ax = axes[i // ncols, i % ncols]
            for m in METHODS:
                A = hp_data[(m, k)]
                if not np.isfinite(A).any():
                    continue
                st = style_for(m)
                med = np.nanmedian(A, axis=0)
                q25, q75 = np.nanpercentile(A, 25, axis=0), np.nanpercentile(A, 75, axis=0)
                ax.plot(bo_x, med, color=st["color"], lw=1.6, label=st["label"])
                ax.fill_between(bo_x, q25, q75, color=st["color"], alpha=0.13, lw=0)
            ax.set_xlabel("BO iteration $t$"); ax.set_ylabel(label)
            if k in ("lengthscale", "outputscale"):
                ax.set_yscale("log")
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(len(_panels), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        fig.suptitle("GP hyperparameters", y=1.0)
        fig.tight_layout(); add_figure("04_hyperparameters", fig)

    # 4.4 query locations: distances and prior values across seeds
    if PROCESSED:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
        bo_x = np.arange(1, BUDGET + 1)
        for m in METHODS:
            st = style_for(m)
            D_star = np.asarray([np.linalg.norm(queries_np(load_raw(s), m) - X_STAR, axis=1)
                                 for s in _hp_seeds])
            D_pi = np.asarray([np.linalg.norm(queries_np(load_raw(s), m) - X_PI, axis=1)
                               for s in _hp_seeds])
            for ax, D in zip(axes[:2], (D_star, D_pi)):
                ax.plot(bo_x, np.median(D, axis=0), color=st["color"], lw=1.6, label=st["label"])
                ax.fill_between(bo_x, np.percentile(D, 25, axis=0), np.percentile(D, 75, axis=0),
                                color=st["color"], alpha=0.12, lw=0)
            entry = conv_band(m, "prior_value_at_candidate") if CONV is not None else None
            if entry is not None:
                x_bo = np.asarray(CONV["bo_iteration"], float)
                axes[2].plot(x_bo, entry["center"], color=st["color"], lw=1.6)
                if entry.get("lower") is not None and entry["lower"][0] is not None:
                    axes[2].fill_between(x_bo, entry["lower"], entry["upper"],
                                         color=st["color"], alpha=0.12, lw=0)
        axes[0].set_ylabel(r"$\Vert x_t - x^\star \Vert$"); axes[0].set_title("Distance to optimum")
        axes[1].set_ylabel(r"$\Vert x_t - x_\pi \Vert$"); axes[1].set_title("Distance to prior center")
        axes[2].set_ylabel(r"$\pi(x_t)$"); axes[2].set_title("Prior value at candidates")
        axes[2].axhline(PI_THRESHOLD, color="grey", ls=":", lw=1.0)
        for ax in axes:
            ax.set_xlabel("BO iteration $t$")
        axes[0].legend(fontsize=8)
        fig.suptitle("Query distances and prior value", y=1.02)
        fig.tight_layout(); add_figure("04_query_distances", fig)

    # ========================================================================
    # 5. Calibration
    # ========================================================================
    _CAL = AGG["calib_curves"]
    _CAL_KEYS = [("nlpd", "NLPD", None),
                 ("coverage", "95% coverage", 0.95),
                 ("confidence_violation_rate", r"$\beta$-band violation rate",
                  float(2 * (1 - _norm.cdf(math.sqrt(BETA))))),
                 ("mean_abs_standardized_residual", r"mean $|z|$", float(math.sqrt(2 / math.pi)))]

    def plot_calibration_level(bucket: str, title: str, figname: str):
        if _CAL is None:
            logger.info("calibration_curves.json missing — '%s' skipped.", figname)
            return
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), squeeze=False)
        plotted = False
        for i, (key, label, ref) in enumerate(_CAL_KEYS):
            ax = axes[i // 2, i % 2]
            for m in METHODS:
                mdat = _CAL["methods"].get(m, {})
                entry = mdat.get(bucket, {}).get(key)
                if entry is None:
                    continue
                if "pointwise" in entry:
                    entry = entry["pointwise"]
                x = np.asarray(mdat["iteration"], float)
                c = np.asarray(entry["center"], float)
                if not np.isfinite(c).any():
                    continue
                st = style_for(m)
                ax.plot(x, c, color=st["color"], lw=1.6, marker="o", ms=3.5, label=st["label"])
                lo, up = entry.get("lower"), entry.get("upper")
                if lo is not None and up is not None and lo[0] is not None:
                    ax.fill_between(x, np.asarray(lo, float), np.asarray(up, float),
                                    color=st["color"], alpha=0.13, lw=0)
                plotted = True
            if ref is not None:
                ax.axhline(ref, color="grey", ls="--", lw=1.0, label=f"ideal = {ref:.3f}")
            ax.set_xlabel("BO iteration (checkpoint)"); ax.set_ylabel(label)
            if i == 0:
                ax.legend(fontsize=8)
        if plotted:
            fig.suptitle(title, y=1.0)
            fig.tight_layout(); add_figure(figname, fig)
        else:
            plt.close(fig)
            logger.info("no data for calibration bucket '%s'.", bucket)

    plot_calibration_level("global_calibration", "Full-grid calibration",
                           "05_calibration_global")
    plot_calibration_level("prior_region", "Prior-region calibration",
                           "05_calibration_prior_region")

    # ========================================================================
    # 6. Optional detailed diagnostics — joint-posterior NLPD
    # ========================================================================
    if _CAL is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
        found = False
        for ax, (bucket, key, title) in zip(axes, [
                ("joint_posterior", "nlpd_joint", "Full grid"),
                ("prior_region", "joint_nlpd", "Prior region")]):
            for m in METHODS:
                mdat = _CAL["methods"].get(m, {})
                entry = mdat.get(bucket, {}).get(key)
                if entry is None:
                    continue
                if "pointwise" in entry:
                    entry = entry["pointwise"]
                c = np.asarray(entry["center"], float)
                if not np.isfinite(c).any():
                    continue
                st = style_for(m)
                ax.plot(np.asarray(mdat["iteration"], float), c, color=st["color"],
                        lw=1.6, marker="o", ms=3.5, label=st["label"])
                found = True
            ax.set_xlabel("BO iteration (checkpoint)"); ax.set_ylabel("joint NLPD")
            ax.set_title(title)
        if found:
            axes[0].legend(fontsize=8)
            fig.tight_layout(); add_figure("06_joint_nlpd", fig)
        else:
            plt.close(fig)
            logger.info("joint-posterior diagnostics not present in this run.")

    # ========================================================================
    # Save all figures
    # ========================================================================
    out_dir = RUN_DIR / "figures" / BENCH_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    export_formats = ("png", "pdf")
    for name, fig in FIGURES.items():
        for ext in export_formats:
            fig.savefig(out_dir / f"{name}.{ext}")
        plt.close(fig)
    logger.info("[%s] saved %d figures (x %d formats) to %s",
                BENCH_NAME, len(FIGURES), len(export_formats), out_dir)
    return len(FIGURES)


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
