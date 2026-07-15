"""Experiment C — across-seed aggregation and figures.

Reads the per-seed processed files written by ``process.py`` (never rebuilds a
model), aggregates them across seeds with inter-seed percentile bands, writes
an aggregate summary JSON, and produces the figures under ``figures/``.

Figures (all saved as PNG + PDF; strictly-positive regrets on a log axis):

``fixed_scalarization``
    per (scalarization, angle): best-so-far, simple regret, cumulative
    instantaneous regret — RIGHT vs WRONG; plus a per-scalarization aggregate
    over angles.

``pareto_front``
    hypervolume, hypervolume regret, empirical Bayes regret vs the **global**
    objective budget (all methods overlaid); objective-space (raw ZDT,
    ``lower is better``) and decision-space (``input_dim == 2``) scatters for
    one seed.  Passing several ``--run-dir`` produces the L-ablation overlay.

CLI::

    python -m experiments.exp_c_multiobjective.analyze \\
        --run-dir results/exp_c_multiobjective/runs/<run_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from common.serialization import load_json, save_json
from common.statistics import summarize_sample, wilcoxon_paired_comparison
from benchmarks import multi_objective as mo_benchmarks
from experiments.exp_c_multiobjective import plotting
from data.raw_multiobjective import ParetoSeedData
from common.hypervolume import non_dominated_mask
from experiments.exp_c_multiobjective.process import assert_compatible_metadata

logger = logging.getLogger("analyze_exp_c")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_processed(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "processed" / "per_seed"
    if not root.exists():
        raise FileNotFoundError(
            f"No processed/per_seed under {run_dir}; run process.py first."
        )
    out = []
    for bench_dir in sorted(root.iterdir()):
        if not bench_dir.is_dir():
            continue
        for path in sorted(bench_dir.glob("seed_*_processed.json")):
            out.append(load_json(path))
    return out


def _stack(curves: list[list[float]]) -> np.ndarray:
    """Stack per-seed curves into ``(n_seeds, n_steps)`` (truncate to min len)."""
    min_len = min(len(c) for c in curves)
    return np.asarray([c[:min_len] for c in curves], dtype=float)


# ---------------------------------------------------------------------------
# Fixed scalarization
# ---------------------------------------------------------------------------

def _analyze_fixed(processed: list[dict[str, Any]], fig_dir: Path) -> dict[str, Any]:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    # (scalarization, angle) -> method -> metric -> list of per-seed curves
    store: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for pdict in processed:
        for case in pdict["cases"]:
            key = (case["scalarization"], float(case["lambda"]["angle_degrees"]))
            for mname, m in case["methods"].items():
                for metric in ("best_so_far", "simple_regret",
                               "instantaneous_regret", "cumulative_regret"):
                    store[key][mname][metric].append(m[metric])

    summary: dict[str, Any] = {"mode": "fixed_scalarization", "cases": []}
    scalarizations = sorted({k[0] for k in store})

    for scal in scalarizations:
        angles = sorted(a for (s, a) in store if s == scal)
        # Per-angle detailed comparison figures.
        for angle in angles:
            key = (scal, angle)
            methods = store[key]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            for mname, metrics in methods.items():
                st = plotting.style_for(mname)
                x = list(range(1, _stack(metrics["best_so_far"]).shape[1] + 1))
                plotting.curve_with_band(axes[0], x, _stack(metrics["best_so_far"]),
                                         color=st["color"], label=st["label"])
                plotting.curve_with_band(axes[1], x, _stack(metrics["simple_regret"]),
                                         color=st["color"], label=st["label"])
                plotting.curve_with_band(axes[2], x, _stack(metrics["cumulative_regret"]),
                                         color=st["color"], label=st["label"])
            axes[0].set_title("best-so-far scalarized"); axes[0].set_xlabel("evaluations")
            axes[1].set_title("simple regret"); axes[1].set_xlabel("evaluations")
            axes[1].set_yscale("log")
            axes[2].set_title("cumulative inst. regret"); axes[2].set_xlabel("evaluations")
            for ax in axes:
                ax.legend(loc="best")
            fig.suptitle(f"fixed scalarization — {scal}, angle {angle:g} deg")
            plotting.save_fig(fig, fig_dir, f"fixed_{scal}_angle{int(angle)}")

            # Per-(scal, angle) summary + paired test on final simple regret.
            case_summary: dict[str, Any] = {"scalarization": scal, "angle": angle,
                                            "methods": {}}
            finals: dict[str, np.ndarray] = {}
            for mname, metrics in methods.items():
                arr = _stack(metrics["simple_regret"])
                finals[mname] = arr[:, -1]
                case_summary["methods"][mname] = summarize_sample(arr[:, -1])
            if "right" in finals and "wrong" in finals:
                case_summary["wilcoxon_right_vs_wrong"] = wilcoxon_paired_comparison(
                    finals["right"], finals["wrong"], alternative="greater"
                )
            summary["cases"].append(case_summary)

        # Per-scalarization aggregate over angles: mean across angles of the
        # per-seed-median simple regret, RIGHT vs WRONG.
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for mname in sorted({m for a in angles for m in store[(scal, a)]}):
            per_angle_medians = []
            for angle in angles:
                arr = _stack(store[(scal, angle)][mname]["simple_regret"])
                per_angle_medians.append(np.nanmedian(arr, axis=0))
            agg = np.asarray(per_angle_medians)   # (n_angles, n_steps)
            st = plotting.style_for(mname)
            x = list(range(1, agg.shape[1] + 1))
            ax.plot(x, agg.mean(axis=0), color=st["color"], label=st["label"])
            ax.fill_between(x, agg.min(axis=0), agg.max(axis=0),
                            color=st["color"], alpha=0.15, linewidth=0)
        ax.set_yscale("log"); ax.set_xlabel("evaluations")
        ax.set_ylabel("simple regret (median, mean over angles)")
        ax.set_title(f"fixed scalarization — {scal}: aggregate over angles")
        ax.legend(loc="best")
        plotting.save_fig(fig, fig_dir, f"fixed_{scal}_aggregate_simple_regret")

    return summary


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def _analyze_pareto(
    processed: list[dict[str, Any]],
    run_dir: Path,
    fig_dir: Path,
) -> dict[str, Any]:
    plotting.apply_style()
    import matplotlib.pyplot as plt

    # method -> metric -> list of per-seed curves
    store: dict = defaultdict(lambda: defaultdict(list))
    for pdict in processed:
        for mname, m in pdict["methods"].items():
            for metric in ("hypervolume", "hypervolume_regret",
                           "empirical_bayes_regret"):
                store[mname][metric].append(m[metric])
    hv_star = float(processed[0]["hv_star"])

    # Time-series figures (x-axis = global objective budget).
    specs = [
        ("hypervolume", "hypervolume", False, "exact hypervolume"),
        ("hypervolume_regret", "hv_regret", True, "hypervolume regret"),
        ("empirical_bayes_regret", "bayes_regret", True, "empirical Bayes regret"),
    ]
    summary: dict[str, Any] = {"mode": "pareto_front", "hv_star": hv_star,
                               "methods": {}}
    for metric, fname, logy, title in specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for mname in sorted(store):
            arr = _stack(store[mname][metric])
            st = plotting.style_for(mname)
            x = list(range(1, arr.shape[1] + 1))
            plotting.curve_with_band(ax, x, arr, color=st["color"], label=st["label"])
        if logy:
            ax.set_yscale("log")
        if metric == "hypervolume":
            ax.axhline(hv_star, color="#000000", ls="--", lw=1.0, label="true HV*")
        ax.set_xlabel("global objective evaluations")
        ax.set_ylabel(title)
        ax.set_title(f"pareto front — {title}")
        ax.legend(loc="best")
        plotting.save_fig(fig, fig_dir, f"pareto_{fname}")

    for mname in sorted(store):
        summary["methods"][mname] = {
            "final_hypervolume": summarize_sample(_stack(store[mname]["hypervolume"])[:, -1]),
            "final_hv_regret": summarize_sample(_stack(store[mname]["hypervolume_regret"])[:, -1]),
            "final_bayes_regret": summarize_sample(_stack(store[mname]["empirical_bayes_regret"])[:, -1]),
        }

    # Objective- and decision-space scatters for the first available seed.
    _pareto_space_figures(run_dir, fig_dir)
    return summary


def _pareto_space_figures(run_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    raw_root = run_dir / "raw"
    bench_dirs = [d for d in sorted(raw_root.iterdir()) if d.is_dir()]
    if not bench_dirs:
        return
    seed_files = sorted(bench_dirs[0].glob("seed_*.json"))
    if not seed_files:
        return
    raw = ParetoSeedData.from_dict(load_json(seed_files[0]))
    seed = int(raw.metadata["seed"])
    benchmark = mo_benchmarks.BENCHMARK_REGISTRY[raw.metadata["benchmark"]](
        input_dim=int(raw.metadata["input_dim"]),
        reference_point=torch.tensor(raw.reference_point, dtype=torch.double),
    )
    front_raw = benchmark.true_pareto_front_raw(400).numpy()
    pset = benchmark.true_pareto_set(400).numpy()

    for mname, method in raw.methods.items():
        X = torch.tensor([e.x for e in method.global_evaluations], dtype=torch.double)
        raw_y = benchmark.evaluate_raw(X)
        oriented = -raw_y
        nd = non_dominated_mask(oriented)
        raw_np = raw_y.numpy()

        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        plotting.objective_space(
            ax, front_raw=front_raw, evaluated_raw=raw_np,
            nondominated_raw=raw_np[nd.numpy()],
            title=f"objective space — {mname} (seed {seed})",
        )
        plotting.save_fig(fig, fig_dir, f"pareto_objective_space_{mname}_seed{seed:03d}")

        if benchmark.input_dim == 2:
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            plotting.decision_space(
                ax, pareto_set=pset, evaluated_X=X.numpy(),
                nondominated_X=X.numpy()[nd.numpy()],
                title=f"decision space — {mname} (seed {seed})",
            )
            plotting.save_fig(fig, fig_dir, f"pareto_decision_space_{mname}_seed{seed:03d}")


def _l_ablation(run_dirs: list[Path], fig_dir: Path) -> None:
    """Overlay Algorithm-2 Bayes/HV regret across runs differing in ``L``."""
    import matplotlib.pyplot as plt
    series: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for rd in run_dirs:
        processed = _load_processed(rd)
        if not processed or processed[0].get("mode") != "pareto_front":
            continue
        L = int(processed[0]["metadata"]["n_scalarizations_algorithm_2"])
        for mname in ("right_algorithm_2", "wrong_algorithm_2"):
            curves = [p["methods"][mname]["empirical_bayes_regret"]
                      for p in processed if mname in p["methods"]]
            if curves:
                series[mname].append((L, _stack(curves)))
    if not series:
        return
    for mname, runs in series.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for L, arr in sorted(runs):
            x = list(range(1, arr.shape[1] + 1))
            ax.plot(x, np.nanmedian(arr, axis=0), label=f"L={L}")
        ax.set_yscale("log"); ax.set_xlabel("global objective evaluations")
        ax.set_ylabel("empirical Bayes regret (median)")
        ax.set_title(f"L ablation — {mname}")
        ax.legend(loc="best")
        plotting.save_fig(fig, fig_dir, f"pareto_L_ablation_{mname}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Run directory (repeatable; several enables L ablation).")
    args = parser.parse_args(argv)

    run_dirs = [rd.resolve() for rd in args.run_dir]
    primary = run_dirs[0]
    processed = _load_processed(primary)
    if not processed:
        logger.warning("No processed files found; run process.py first.")
        return

    assert_compatible_metadata([p["metadata"] for p in processed])
    mode = processed[0]["mode"]
    fig_dir = primary / "figures"

    if mode == "fixed_scalarization":
        summary = _analyze_fixed(processed, fig_dir)
    elif mode == "pareto_front":
        summary = _analyze_pareto(processed, primary, fig_dir)
        if len(run_dirs) > 1:
            _l_ablation(run_dirs, fig_dir)
    else:
        raise ValueError(f"Unknown processed mode '{mode}'.")

    summary["n_seeds"] = len(processed)
    out = primary / "processed" / "aggregate_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, out)
    logger.info(f"Wrote figures to {fig_dir} and summary to {out}")


if __name__ == "__main__":
    main()
