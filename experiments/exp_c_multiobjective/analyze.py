"""Experiment C — across-seed aggregation and figures.

Reads the per-seed processed files written by ``process.py`` (never rebuilds a
model), aggregates them across seeds with inter-seed percentile bands, writes
an aggregate summary JSON, and produces the figures under ``figures/``.

Figures (all saved as PNG + PDF; strictly-positive regrets on a log axis):

``fixed_scalarization``
    per (scalarization, angle): simple and cumulative instantaneous regret —
    RIGHT vs WRONG; plus per-scalarization aggregates over angles and query maps.

``pareto_front``
    hypervolume regret, cumulative dominance regret, dominated query fraction,
    IGD, query maps and final fronts. Secondary views go to ``supplementary/``.
    Passing several ``--run-dir`` produces the L-ablation overlay.

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
from experiments.exp_c_multiobjective.trajectory_figures import fixed_trajectory_figures
from experiments.exp_c_multiobjective.final_comparisons import final_comparisons
from experiments.exp_c_multiobjective.figure_selection import organize_existing_figures
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
    """Stack equal-budget per-seed curves into ``(n_seeds, n_steps)``."""
    if not curves or len({len(c) for c in curves}) != 1:
        raise ValueError("Curves must share identical evaluation budgets")
    return np.asarray(curves, dtype=float)


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
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for mname, metrics in methods.items():
                st = plotting.style_for(mname)
                x = list(range(1, _stack(metrics["best_so_far"]).shape[1] + 1))
                plotting.curve_with_band(axes[0], x, _stack(metrics["simple_regret"]),
                                         color=st["color"], label=st["label"])
                plotting.curve_with_band(axes[1], x, _stack(metrics["cumulative_regret"]),
                                         color=st["color"], label=st["label"])
            axes[0].set_title("simple regret"); axes[0].set_xlabel("evaluations")
            axes[0].set_yscale("log")
            axes[1].set_title("cumulative inst. regret"); axes[1].set_xlabel("evaluations")
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

        # Same angle aggregation convention for both regrets; the band is the
        # angle min–max range, not a confidence interval or inter-seed IQR.
        for metric in ("simple_regret", "cumulative_regret"):
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for mname in sorted({m for a in angles for m in store[(scal, a)]}):
                arrays = [_stack(store[(scal, a)][mname][metric]) for a in angles]
                agg = np.asarray([np.nanmedian(arr, axis=0) for arr in arrays])
                st = plotting.style_for(mname)
                x = list(range(1, agg.shape[1] + 1))
                ax.plot(x, agg.mean(axis=0), color=st["color"], label=st["label"])
                ax.plot(x, np.mean([arr.mean(axis=0) for arr in arrays], axis=0),
                        color=st["color"], ls="--", alpha=.55,
                        label=f"{st['label']} (mean over seeds and angles)")
                ax.fill_between(x, agg.min(axis=0), agg.max(axis=0),
                                color=st["color"], alpha=0.15, linewidth=0)
            if metric == "simple_regret":
                ax.set_yscale("log")
            label = metric.replace("_", " ")
            ax.set_xlabel("evaluations")
            ax.set_ylabel(f"{label} (median, mean over angles)")
            ax.set_title(f"{scal}: aggregate over angles (band: angle min–max)")
            ax.legend(loc="best")
            plotting.save_fig(fig, fig_dir, f"fixed_{scal}_aggregate_{metric}")

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
            for metric in ("cumulative_dominance_regret", "average_dominance_regret",
                           "dominated_query_fraction", "igd"):
                if metric not in m.get("diagnostics", {}):
                    raise ValueError("New diagnostics missing; rerun process.py first")
                store[mname][metric].append(m["diagnostics"][metric])
    hv_star = float(processed[0]["hv_star"])

    # Time-series figures (x-axis = global objective budget).
    specs = [
        ("hypervolume", "hypervolume", False, "exact hypervolume"),
        ("hypervolume_regret", "hv_regret", True, "hypervolume regret"),
        ("empirical_bayes_regret", "bayes_regret", True, "empirical Bayes regret"),
        ("cumulative_dominance_regret", "cumulative_dominance_regret", False, "cumulative dominance regret"),
        ("average_dominance_regret", "average_dominance_regret", False, "average dominance regret"),
        ("dominated_query_fraction", "dominated_query_fraction", False, "dominated query fraction (tolerance 1e-8)"),
        ("igd", "igd", False, "IGD (normalized arc-length reference)"),
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
            **{f"final_{metric}": summarize_sample(_stack(store[mname][metric])[:, -1])
               for metric in ("cumulative_dominance_regret", "average_dominance_regret", "dominated_query_fraction", "igd")},
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

    space_data = []
    for mname, method in raw.methods.items():
        X = torch.tensor([e.x for e in method.global_evaluations], dtype=torch.double)
        raw_y = benchmark.evaluate_raw(X)
        oriented = -raw_y
        nd = non_dominated_mask(oriented)
        raw_np = raw_y.numpy()
        space_data.append((mname, raw_np, raw_np[nd.numpy()]))

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

    # Compare methods with shared limits and a common global-budget colour scale.
    # Alg. 2's round-robin order is a budget convention, not physical chronology;
    # do not connect points from independent subruns with trajectory segments.
    from matplotlib.colors import Normalize
    norm = Normalize(1, max(len(y) for _, y, _ in space_data))
    for query_map in (True, False):
        fig, axes = plt.subplots(1, len(space_data), figsize=(5*len(space_data), 4.5),
                                 sharex=True, sharey=True, squeeze=False,
                                 layout="constrained")
        for ax, (mname, queries, archive) in zip(axes[0], space_data):
            ax.plot(front_raw[:, 0], front_raw[:, 1], 'k-', lw=1,
                    label="true Pareto front")
            if query_map:
                scatter = ax.scatter(queries[:, 0], queries[:, 1],
                                     c=np.arange(1, len(queries)+1), cmap="viridis",
                                     norm=norm, s=18, alpha=.75)
            else:
                ax.scatter(archive[:, 0], archive[:, 1], s=25,
                           color=plotting.style_for(mname)["color"],
                           label="final nondominated archive")
            ax.set_title(plotting.style_for(mname)["label"])
            ax.set_xlabel(r"$f_1$ (minimize)")
            ax.set_ylabel(r"$f_2$ (minimize)")
            ax.legend(fontsize=7)
        if query_map:
            fig.colorbar(scatter, ax=list(axes[0]), label="global objective budget")
            fig.suptitle(f"All queries — seed {seed} (Alg. 2: round-robin budget order)")
            name = "query_map"
        else:
            fig.suptitle(f"Final Pareto archives — seed {seed}")
            name = "final_front"
        plotting.save_fig(fig, fig_dir, f"pareto_{name}_seed{seed:03d}")


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
    parser.add_argument("--refresh-selection", action="store_true",
                        help="Refresh the figure layout using existing query maps and paired "
                             "comparisons; requires previously generated figures and statistics.")
    args = parser.parse_args(argv)

    run_dirs = [rd.resolve() for rd in args.run_dir]
    primary = run_dirs[0]
    processed = _load_processed(primary)
    if not processed:
        logger.warning("No processed files found; run process.py first.")
        return

    assert_compatible_metadata([p["metadata"] for p in processed])
    for p in processed:
        budget = int(p["metadata"]["budget"])
        entries = ([m for case in p["cases"] for m in case["methods"].values()]
                   if p["mode"] == "fixed_scalarization" else list(p["methods"].values()))
        key = "iteration" if p["mode"] == "fixed_scalarization" else "global_iteration"
        if any(len(m[key]) != budget for m in entries):
            raise ValueError("All methods must use the same full evaluation budget")
    mode = processed[0]["mode"]
    fig_dir = primary / "figures"
    comparisons_path = primary / "processed" / "paired_bootstrap_comparisons.json"
    if args.refresh_selection:
        # Validate reuse before moving any files. Bootstrap definitions do not
        # change in this presentation-only refresh.
        comparisons = load_json(comparisons_path)
        expected_seeds = sorted(int(p["metadata"]["seed"]) for p in processed)
        if (not comparisons["comparisons"] or any(
                sorted(c["seed_ids"]) != expected_seeds for c in comparisons["comparisons"])):
            raise ValueError("Saved paired comparisons do not match processed seeds; run full analyze.py")
        required = [f"{mode}_paired_bootstrap_{c['metric']}"
                    for c in comparisons["comparisons"]]
        if mode == "fixed_scalarization":
            first = processed[0]
            seed = int(first["metadata"]["seed"])
            required.extend(f"fixed_{c['scalarization']}_angle{c['lambda']['angle_degrees']:g}"
                            f"_queries_seed{seed:03d}" for c in first["cases"])
            required.extend(f"fixed_{c['scalarization']}_preference_map_all_seeds"
                            for c in first["cases"])
        if any(not (fig_dir / f"{name}.{ext}").exists()
               and not (fig_dir / "supplementary" / f"{name}.{ext}").exists()
               for name in required for ext in ("png", "pdf")):
            raise ValueError("Missing reusable figures; run full analyze.py without --refresh-selection")
    organize_existing_figures(fig_dir)

    if mode == "fixed_scalarization":
        summary = _analyze_fixed(processed, fig_dir)
        if not args.refresh_selection:
            fixed_trajectory_figures(primary, fig_dir)
    elif mode == "pareto_front":
        summary = _analyze_pareto(processed, primary, fig_dir)
        if len(run_dirs) > 1:
            _l_ablation(run_dirs, fig_dir)
    else:
        raise ValueError(f"Unknown processed mode '{mode}'.")

    if not args.refresh_selection:
        comparisons = final_comparisons(processed, fig_dir)
        save_json(comparisons, comparisons_path)
    summary["n_seeds"] = len(processed)
    out = primary / "processed" / "aggregate_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, out)
    logger.info(f"Wrote figures to {fig_dir} and summary to {out}")


if __name__ == "__main__":
    main()
