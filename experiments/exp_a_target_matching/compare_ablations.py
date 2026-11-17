"""Overlay matched lengthscale/kernel runs after process.py and analyze.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from benchmarks import build_benchmark
from common.serialization import load_json

_ARD_STYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1, 1, 3)))


def _diagnostic_legend(ax, *, lengthscales: bool) -> None:
    """Separate run colours from dimension styles when ARD would crowd the plot."""
    if not lengthscales or len(ax.lines) <= 8:
        ax.legend(fontsize=7)
        return
    runs, dimensions = {}, set()
    for line in ax.lines:
        label, separator, dimension = line.get_label().rpartition(", dim ")
        if not separator:
            ax.legend(fontsize=7)
            return
        runs.setdefault(label, line.get_color())
        dimensions.add(int(dimension))
    handles = [Line2D([], [], color=color, label=label) for label, color in runs.items()]
    handles.extend(Line2D([], [], color="0.3", label=f"dim {d}",
                          ls=_ARD_STYLES[(d-1) % len(_ARD_STYLES)])
                   for d in sorted(dimensions))
    ax.legend(handles=handles, fontsize=7, ncol=2)


def _plot_diagnostic(ax, data: dict, key: str, label: str, color: str) -> None:
    """Plot saved bands; retain every ARD dimension rather than averaging them."""
    keys = (sorted((k for k in data["curves"] if k.startswith("lengthscale_")),
                   key=lambda k: int(k.rsplit("_", 1)[1]))
            if key == "lengthscale" else [key])
    for dimension, metric in enumerate(keys):
        band = data["curves"].get(metric)
        if band is None:
            continue
        legend = (f"{label}, dim {int(metric.rsplit('_', 1)[1])+1}"
                  if key == "lengthscale" and len(keys) > 1 else label)
        ax.plot(data["iteration"], band["center"], label=legend, color=color,
                ls=_ARD_STYLES[dimension % len(_ARD_STYLES)])
        if all(v is not None for v in band["lower"] + band["upper"]):
            ax.fill_between(data["iteration"], band["lower"], band["upper"],
                            color=color, alpha=.12)


def plot_rkhs_diagnostics(series: dict[str, list[tuple[str, dict]]]) -> plt.Figure | None:
    """Snapshot diagnostics, separated by method; tolerate legacy aggregates.

    Each series is a run's seed-mean curve and pointwise bootstrap band.
    No norm values are pooled across kernels or lengthscale policies.
    """
    keys = ("lengthscale", "effective_rkhs_norm", "effective_rkhs_norm_unit_outputscale",
            "target_distance_rmse", "target_distance_rmse_near_target")
    if not any(data["curves"].get("effective_rkhs_norm") is not None
               for runs in series.values() for _, data in runs):
        return None
    fig, axes = plt.subplots(len(series), len(keys), figsize=(23, 4 * len(series)), squeeze=False)
    for row, (method, runs) in enumerate(series.items()):
        for index, (label, data) in enumerate(runs):
            for col, key in enumerate(keys):
                _plot_diagnostic(axes[row, col], data, key, label, f"C{index % 10}")
        for col, title in enumerate(("Lengthscales (input-transform units)",
                                     "Effective RKHS norm (fitted kernel)",
                                     "Centered raw mean norm\n(unit-outputscale kernel)",
                                     "Common target-distance RMSE",
                                     "Near-target distance RMSE")):
            ax = axes[row, col]
            ax.set_title(f"{method}\n{title}", fontsize=10)
            ax.set_xlabel("BO iteration (before query)")
            if ax.lines:
                _diagnostic_legend(ax, lengthscales=(col == 0))
            else:
                ax.text(.5, .5, "Unavailable: reprocess snapshots\nor choose a nonempty near-target subset",
                        transform=ax.transAxes, ha="center", fontsize=8)
        local = [data.get("near_target") for _, data in runs]
        if local and local[0] is not None and all(v == local[0] for v in local):
            axes[row, -1].set_ylabel(f"ε={local[0]['epsilon']}, grid points={local[0]['n_test']}")
    fig.suptitle("GP complexity and target-distance error — seed means and pointwise bootstrap CI\n"
                 "Fitted-kernel norms use different kernels. Unit-outputscale norms are comparable only "
                 "at fixed kernel shape, lengthscale and input coordinates.\n"
                 "Both norms exclude the prior mean; adaptive trajectories show associations, not causality.",
                 fontsize=11)
    fig.tight_layout()
    return fig


def compare_runs(run_dirs: list[Path], output_dir: Path) -> Path:
    """Require equal protocols, seed IDs and initial noisy observations.

    Only kernel and fixed_lengthscale may differ. Bands describe uncertainty
    on the mean over seeds, not the distribution of individual trajectories.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    configurations = [load_json(rd / "config.json") for rd in run_dirs]
    protocols = []
    for config in configurations:
        protocol = dict(config)
        protocol.pop("tag", None)
        model = dict(protocol.get("model", {}))
        model.pop("kernel", None)
        model.pop("fixed_lengthscale", None)
        protocol["model"] = model
        protocols.append(protocol)
    if any(p != protocols[0] for p in protocols[1:]):
        raise ValueError("Ablations must share all settings except kernel and fixed_lengthscale")
    config = configurations[0]
    problem = build_benchmark(config)
    benchmark = problem.name
    initial = None
    for rd in run_dirs:
        designs = []
        for seed in config["seeds"]:
            raw = load_json(rd / "raw" / benchmark / f"seed_{seed:03d}.json")
            designs.append(raw["initial_design"])
        if initial is not None and designs != initial:
            raise ValueError("Ablations must share initial inputs and noisy observations")
        initial = designs
    methods = config["methods"]
    rkhs_series = {method: [] for method in methods}
    fig, axes = plt.subplots(len(methods), 4, figsize=(18, 4 * len(methods)), squeeze=False)
    for index, (rd, cfg) in enumerate(zip(run_dirs, configurations)):
        aggregate = rd / "processed" / "aggregate" / benchmark
        diagnostics = load_json(aggregate / "generalization_curves.json")
        model = cfg.get("model", {})
        label = f"{model.get('kernel', 'matern52')}, "
        fixed = model.get("fixed_lengthscale")
        label += ("fixed ℓ (ARD)" if isinstance(fixed, list) else
                  f"fixed ℓ={fixed}" if fixed is not None else "MLL ℓ")
        records = [load_json(rd / "processed" / "per_seed" / benchmark /
                             f"seed_{seed:03d}_processed.json") for seed in config["seeds"]]
        for row, method in enumerate(methods):
            data = diagnostics["methods"][method]
            rkhs_series[method].append((label, data))
            for col, key in enumerate(("lengthscale", "rmse", "mean_posterior_variance")):
                _plot_diagnostic(axes[row, col], data, key, label, f"C{index % 10}")
            distances = np.asarray([r["methods"][method]["convergence"]["running_best_target_distance"]
                                    for r in records])
            axes[row, 3].plot(np.arange(1, distances.shape[1] + 1), distances.mean(0),
                              label=label, color=f"C{index % 10}")
    for row, method in enumerate(methods):
        for col, title in enumerate(("Lengthscales (input-transform units)", "Native-function RMSE",
                                     "Mean latent variance", "Best target distance")):
            axes[row, col].set_title(f"{method}\n{title}")
            axes[row, col].set_xlabel("Total evaluations" if col == 3 else "BO iteration (before query)")
            _diagnostic_legend(axes[row, col], lengthscales=(col == 0))
    confidence_pct = 100 * diagnostics["confidence"]
    fig.suptitle(f"Matched ablations — means over seeds; diagnostic bands: {confidence_pct:g}% bootstrap CI")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"lengthscale_ablations.{ext}", bbox_inches="tight")
    plt.close(fig)
    rkhs_fig = plot_rkhs_diagnostics(rkhs_series)
    if rkhs_fig is not None:
        for ext in ("png", "pdf"):
            rkhs_fig.savefig(output_dir / f"rkhs_complexity_ablations.{ext}", bbox_inches="tight")
        plt.close(rkhs_fig)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare_runs(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
