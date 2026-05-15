"""Experiment A — across-seed aggregator.

Loads every ``processed/per_seed/<benchmark>/seed_NNN_processed.json`` file
in a run directory and writes four aggregate JSON files per benchmark.

This is the **only** stage that touches multiple seeds.  It uses utilities
from :mod:`common.statistics`; it never reloads raw trajectories.

CLI::

    python -m experiments.exp_a_target_matching.analyze \\
        --run-dir results/exp_a_target_matching/runs/<run_id>

Outputs (under ``<run_dir>/processed/aggregate/<benchmark>/``):

- ``final_performance.json``     — per-method descriptive stats on headline scalars.
- ``convergence_curves.json``    — per-method median + bootstrap CI on time series.
- ``calibration_summary.json``   — per-method descriptive stats on calibration scalars.
- ``statistical_tests.json``     — pairwise Mann-Whitney, rank-biserial, Cohen's d.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.logging import RunLogger
from common.serialization import load_json, save_json
from common.statistics import (
    aggregate_convergence_curves,
    cohens_d_right_better,
    failure_rate,
    mann_whitney_comparison,
    rank_biserial_correlation,
    summarize_sample,
)
from data.processed import ProcessedSeedData

logger = logging.getLogger("analyze")


# Headline scalars summarised in final_performance.json and compared in
# statistical_tests.json.  Keep this list small and meaningful.
_FINAL_SCALAR_KEYS: list[str] = [
    "final_best_target_distance",
    "final_target_simple_regret",
    "mean_inst_target_distance",
    "worst_inst_target_distance",
    "cumulative_inst_target_regret",
    "cumulative_bsf_target_regret",
    "auc_running_target_regret",
    "hitting_time",
]

# Calibration scalars summarised in calibration_summary.json.
_CALIB_TEST_GRID_KEYS: list[str] = [
    "nlpd",
    "coverage",
    "mean_standardized_residual",
    "std_standardized_residual",
    "mean_abs_standardized_residual",
    "pit_mean",
    "pit_std",
    "confidence_violation_rate",
]
_CALIB_NEAR_TARGET_KEYS: list[str] = _CALIB_TEST_GRID_KEYS + [
    "near_target_count",
    "near_target_fraction",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_scalars(seeds: list[ProcessedSeedData], method: str, key: str) -> list[float]:
    """Per-seed values of one final scalar for one method, dropping ``None``."""
    out: list[float] = []
    for psd in seeds:
        v = getattr(psd.methods[method].final, key)
        if v is None:
            continue
        out.append(float(v))
    return out


def _stack_calibration(
    seeds: list[ProcessedSeedData],
    method: str,
    bucket: str,
    key: str,
) -> list[float]:
    """Per-seed values of one calibration scalar from the final checkpoint,
    dropping ``NaN`` and ``None``."""
    out: list[float] = []
    for psd in seeds:
        calibrations = psd.methods[method].calibrations
        if not calibrations:
            continue
        report = getattr(calibrations[-1], bucket)
        if report is None:
            continue
        v = report.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        out.append(fv)
    return out


def _stack_curves(
    seeds: list[ProcessedSeedData],
    method: str,
    key: str,
) -> np.ndarray:
    """``(n_seeds, n_steps)`` array of one convergence curve."""
    rows = [getattr(psd.methods[method].convergence, key) for psd in seeds]
    return np.asarray(rows, dtype=np.float64)


def _maybe_summary(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    return summarize_sample(values)


def _pairs_from_methods(method_names: list[str]) -> list[tuple[str, str]]:
    """Pair every ``right_*`` with every ``wrong_*`` method.  Order matters
    because the statistics functions use a one-sided ``RIGHT < WRONG`` test.
    """
    rights = [m for m in method_names if m.startswith("right_")]
    wrongs = [m for m in method_names if m.startswith("wrong_")]
    return [(r, w) for r in rights for w in wrongs]


# ---------------------------------------------------------------------------
# Aggregate JSON builders
# ---------------------------------------------------------------------------

def _build_final_performance(
    seeds: list[ProcessedSeedData],
    methods: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(seeds), "methods": {}}
    for m in methods:
        out["methods"][m] = {
            key: _maybe_summary(_stack_scalars(seeds, m, key))
            for key in _FINAL_SCALAR_KEYS
        }
        # Failure rate is binary, not a continuous scalar — handled here.
        flags = _stack_scalars(seeds, m, "failure_indicator")
        if flags:
            out["methods"][m]["failure_rate"] = failure_rate(flags, threshold=0.5)
    return out


def _build_convergence_curves(
    seeds: list[ProcessedSeedData],
    methods: list[str],
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
) -> dict[str, Any]:
    curve_keys = [
        "instantaneous_target_distance",
        "running_best_target_distance",
        "cumulative_inst_target_regret",
        "cumulative_bsf_target_regret",
    ]
    iteration = seeds[0].methods[methods[0]].convergence.iteration

    out: dict[str, Any] = {
        "n_seeds":   len(seeds),
        "statistic": statistic,
        "confidence": confidence,
        "iteration": iteration,
        "methods":   {},
    }
    for m in methods:
        method_curves: dict[str, dict[str, list[float]]] = {}
        for key in curve_keys:
            curves = _stack_curves(seeds, m, key)
            if curves.shape[0] == 0:
                continue
            agg = aggregate_convergence_curves(
                {m: curves},
                statistic=statistic,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
            )[m]
            method_curves[key] = {
                "center": agg["center"].tolist(),
                "lower":  agg["lower"].tolist(),
                "upper":  agg["upper"].tolist(),
            }
        out["methods"][m] = method_curves
    return out


def _build_calibration_summary(
    seeds: list[ProcessedSeedData],
    methods: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(seeds), "methods": {}}
    for m in methods:
        out["methods"][m] = {
            "test_grid": {
                key: _maybe_summary(_stack_calibration(seeds, m, "test_grid", key))
                for key in _CALIB_TEST_GRID_KEYS
            },
            "near_target": {
                key: _maybe_summary(_stack_calibration(seeds, m, "near_target", key))
                for key in _CALIB_NEAR_TARGET_KEYS
            },
        }
    return out


def _build_statistical_tests(
    seeds: list[ProcessedSeedData],
    methods: list[str],
    failure_threshold: Optional[float],
) -> dict[str, Any]:
    pairs = _pairs_from_methods(methods)
    out: dict[str, Any] = {"n_seeds": len(seeds), "comparisons": []}
    for right, wrong in pairs:
        per_metric: dict[str, Any] = {}
        for key in _FINAL_SCALAR_KEYS:
            r = _stack_scalars(seeds, right, key)
            w = _stack_scalars(seeds, wrong, key)
            if not r or not w:
                continue
            entry: dict[str, Any] = {
                "n_right":      len(r),
                "n_wrong":      len(w),
                "mann_whitney": mann_whitney_comparison(r, w, alternative="less"),
                "rank_biserial": rank_biserial_correlation(r, w),
                "cohens_d_right_better": cohens_d_right_better(r, w),
            }
            if failure_threshold is not None:
                entry["right_failure_rate"] = failure_rate(r, failure_threshold)
                entry["wrong_failure_rate"] = failure_rate(w, failure_threshold)
            per_metric[key] = entry
        out["comparisons"].append({
            "right":   right,
            "wrong":   wrong,
            "metrics": per_metric,
        })
    return out


# ---------------------------------------------------------------------------
# Run-dir orchestration
# ---------------------------------------------------------------------------

def _discover_processed(run_dir: Path) -> dict[str, list[Path]]:
    """Map ``benchmark_name -> sorted list[seed JSON paths]``."""
    root = run_dir / "processed" / "per_seed"
    if not root.exists():
        raise FileNotFoundError(
            f"No processed/per_seed/ under {run_dir}.  Run process.py first."
        )
    out: dict[str, list[Path]] = {}
    for bench_dir in sorted(root.iterdir()):
        if not bench_dir.is_dir():
            continue
        files = sorted(bench_dir.glob("seed_*_processed.json"))
        if files:
            out[bench_dir.name] = files
    return out


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", default=None,
                        help="Filter (repeatable).  Defaults to all benchmarks.")
    parser.add_argument("--statistic", choices=("mean", "median"), default="median")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args(argv)
    run_dir: Path = args.run_dir.resolve()

    config = load_json(run_dir / "config.json")
    failure_threshold = config.get("failure_threshold")
    if failure_threshold is not None:
        failure_threshold = float(failure_threshold)

    run_logger = RunLogger(run_dir)
    benchmarks = _discover_processed(run_dir)
    if args.benchmark is not None:
        benchmarks = {k: v for k, v in benchmarks.items() if k in args.benchmark}
    if not benchmarks:
        logger.warning("No processed seed files matched the filters; nothing to do.")
        return

    for benchmark, paths in benchmarks.items():
        seeds = [ProcessedSeedData.from_dict(load_json(p)) for p in paths]
        method_names = list(seeds[0].methods.keys())
        logger.info(
            f"[{benchmark}] aggregating {len(seeds)} seed(s); "
            f"methods={method_names}"
        )

        save_json(
            _build_final_performance(seeds, method_names),
            run_logger.processed_aggregate_path(benchmark, "final_performance.json"),
        )
        save_json(
            _build_convergence_curves(
                seeds, method_names,
                statistic=args.statistic,
                confidence=args.confidence,
                n_bootstrap=args.n_bootstrap,
            ),
            run_logger.processed_aggregate_path(benchmark, "convergence_curves.json"),
        )
        save_json(
            _build_calibration_summary(seeds, method_names),
            run_logger.processed_aggregate_path(benchmark, "calibration_summary.json"),
        )
        save_json(
            _build_statistical_tests(seeds, method_names, failure_threshold),
            run_logger.processed_aggregate_path(benchmark, "statistical_tests.json"),
        )
        logger.info(
            f"  -> wrote 4 aggregate files to "
            f"processed/aggregate/{benchmark}"
        )


if __name__ == "__main__":
    main()
