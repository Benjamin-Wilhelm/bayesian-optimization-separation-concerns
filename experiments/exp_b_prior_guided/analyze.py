"""Experiment B — across-seed aggregator.

Loads every ``processed/per_seed/<benchmark>/seed_NNN_processed.json`` file
in a run directory and writes aggregate JSON files per benchmark, mirroring
Experiment A's ``analyze.py`` with prior-guided BO metrics.

This is the only stage that touches multiple seeds.  It reuses
:mod:`common.statistics`; it never reloads raw trajectories.

CLI::

    python -m experiments.exp_b_prior_guided.analyze \\
        --run-dir results/exp_b_prior_guided/runs/<run_id>

Outputs (under ``<run_dir>/processed/aggregate/<benchmark>/``):

- ``final_performance.json``   — per-method descriptive stats on headline
  BO scalars (simple regret, cumulative regret, lambda_final, ...).
- ``convergence_curves.json``  — per-method regret / best-value curves on
  the evaluation axis, plus prior-value and lambda curves on the BO axis.
- ``calibration_summary.json`` — per-method calibration scalars at the final
  checkpoint, for the global / prior-region / joint-posterior levels.
- ``statistical_tests.json``   — each method against the baseline
  (``vanilla_ucb`` by default): paired Wilcoxon (primary) plus
  Mann-Whitney, rank-biserial, Cohen's d (secondary), on
  smaller-is-better regret scalars.
- ``calibration_curves.json``  — per-method calibration curves over GP
  checkpoints (written when ``aggregate.aggregate_calibration_curves``).
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
    mann_whitney_comparison,
    rank_biserial_correlation,
    summarize_sample,
    wilcoxon_paired_comparison,
)
from data.processed_prior import ProcessedPriorSeedData

logger = logging.getLogger("analyze_exp_b")


# Headline scalars summarized in final_performance.json.
_FINAL_SCALAR_KEYS: list[str] = [
    "final_best_value",
    "final_simple_regret",
    "mean_inst_regret",
    "worst_inst_regret",
    "cumulative_inst_regret",
    "auc_simple_regret",
    "hitting_time",
    "lambda_final",
    "mean_prior_value",
]

# Smaller-is-better scalars used for pairwise statistical tests.
_TEST_SCALAR_KEYS: list[str] = [
    "final_simple_regret",
    "mean_inst_regret",
    "worst_inst_regret",
    "cumulative_inst_regret",
    "auc_simple_regret",
    "hitting_time",
]

# Convergence curves on the full evaluation axis.
_EVAL_CURVE_KEYS: list[str] = [
    "instantaneous_regret",
    "running_best_value",
    "simple_regret",
    "cumulative_inst_regret",
]

# Curves on the BO-iteration axis (only exist for BO steps).
_BO_CURVE_KEYS: list[str] = [
    "prior_value_at_candidate",
    "lambda_per_iteration",
]

# Calibration scalars summarized per level.
_CALIB_GLOBAL_KEYS: list[str] = [
    "nlpd",
    "coverage",
    "mean_standardized_residual",
    "std_standardized_residual",
    "mean_abs_standardized_residual",
    "pit_mean",
    "pit_std",
    "confidence_violation_rate",
]
_CALIB_PRIOR_REGION_JOINT_KEYS: list[str] = [
    "joint_nlpd",
    "joint_mahalanobis_sq",
    "joint_logdet",
    "joint_n_test",
]
_CALIB_PRIOR_REGION_KEYS: list[str] = _CALIB_GLOBAL_KEYS + [
    "prior_region_count",
    "prior_region_fraction",
] + _CALIB_PRIOR_REGION_JOINT_KEYS
_CALIB_JOINT_KEYS: list[str] = [
    "nlpd_joint",
    "mahalanobis_sq",
    "logdet",
]

# Standardized-residual metrics may use a uniform (simultaneous) band.
_RESIDUAL_METRIC_KEYS: set[str] = {
    "mean_standardized_residual",
    "std_standardized_residual",
    "mean_abs_standardized_residual",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_scalars(
    seeds: list[ProcessedPriorSeedData], method: str, key: str
) -> list[float]:
    """Per-seed values of one final scalar for one method, dropping ``None``."""
    out: list[float] = []
    for psd in seeds:
        v = getattr(psd.methods[method].final, key)
        if v is None:
            continue
        out.append(float(v))
    return out


def _stack_scalars_paired(
    seeds: list[ProcessedPriorSeedData],
    method: str,
    baseline: str,
    key: str,
) -> tuple[list[float], list[float]]:
    """Seed-aligned values for a (method, baseline) pair.

    A seed contributes only if both methods have a non-``None`` value, so
    the returned lists are paired index-for-index.
    """
    m_out: list[float] = []
    b_out: list[float] = []
    for psd in seeds:
        mv = getattr(psd.methods[method].final, key, None)
        bv = getattr(psd.methods[baseline].final, key, None)
        if mv is None or bv is None:
            continue
        m_out.append(float(mv))
        b_out.append(float(bv))
    return m_out, b_out


def _stack_calibration(
    seeds: list[ProcessedPriorSeedData],
    method: str,
    bucket: str,
    key: str,
) -> list[float]:
    """Per-seed values of one calibration scalar at the final checkpoint."""
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
    seeds: list[ProcessedPriorSeedData],
    method: str,
    key: str,
) -> np.ndarray:
    """``(n_seeds, n_steps)`` array of one convergence curve.

    Seeds where the curve is ``None`` (e.g. ``lambda_per_iteration`` for
    standard-GP methods) are skipped.
    """
    rows = []
    for psd in seeds:
        curve = getattr(psd.methods[method].convergence, key)
        if curve is None:
            continue
        rows.append(curve)
    if not rows:
        return np.zeros((0, 0))
    return np.asarray(rows, dtype=np.float64)


def _maybe_summary(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    return summarize_sample(values)


def _pairs_from_methods(
    method_names: list[str],
    baseline: str,
) -> list[tuple[str, str]]:
    """Pair every non-baseline method with the baseline.

    Order matters: the statistics functions use one-sided tests of
    ``method < baseline`` on smaller-is-better metrics.
    """
    if baseline not in method_names:
        return []
    return [(m, baseline) for m in method_names if m != baseline]


# ---------------------------------------------------------------------------
# Aggregate JSON builders
# ---------------------------------------------------------------------------

def _build_final_performance(
    seeds: list[ProcessedPriorSeedData],
    methods: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(seeds), "methods": {}}
    for m in methods:
        out["methods"][m] = {
            key: _maybe_summary(_stack_scalars(seeds, m, key))
            for key in _FINAL_SCALAR_KEYS
        }
    return out


def _band_to_jsonable(band: dict) -> dict:
    out: dict = {}
    for k, v in band.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _serialize_band(agg: dict, band_type: str) -> dict:
    if band_type == "both":
        return {
            "pointwise": _band_to_jsonable(agg["pointwise"]),
            "uniform":   _band_to_jsonable(agg["uniform"]),
        }
    return _band_to_jsonable(agg)


def _aggregate_curve_set(
    seeds: list[ProcessedPriorSeedData],
    method: str,
    keys: list[str],
    statistic: str,
    confidence: float,
    n_bootstrap: int,
    band_type: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        curves = _stack_curves(seeds, method, key)
        if curves.shape[0] == 0:
            continue
        agg = aggregate_convergence_curves(
            {method: curves},
            statistic=statistic,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            band_type=band_type,
        )[method]
        out[key] = _serialize_band(agg, band_type)
    return out


def _build_convergence_curves(
    seeds: list[ProcessedPriorSeedData],
    methods: list[str],
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    band_type: str = "pointwise",
) -> dict[str, Any]:
    iteration = seeds[0].methods[methods[0]].convergence.iteration
    bo_iteration = seeds[0].methods[methods[0]].convergence.bo_iteration

    out: dict[str, Any] = {
        "n_seeds":      len(seeds),
        "statistic":    statistic,
        "confidence":   confidence,
        "band_type":    band_type,
        "n_bootstrap":  int(n_bootstrap),
        "iteration":    iteration,
        "bo_iteration": bo_iteration,
        "methods":      {},
    }
    for m in methods:
        method_curves = _aggregate_curve_set(
            seeds, m, _EVAL_CURVE_KEYS,
            statistic, confidence, n_bootstrap, band_type,
        )
        method_curves.update(_aggregate_curve_set(
            seeds, m, _BO_CURVE_KEYS,
            statistic, confidence, n_bootstrap, band_type,
        ))
        out["methods"][m] = method_curves
    return out


def _build_calibration_summary(
    seeds: list[ProcessedPriorSeedData],
    methods: list[str],
) -> dict[str, Any]:
    """Aggregate calibration scalars across seeds, on the final checkpoint."""
    test_grid = seeds[0].test_grid.to_dict()
    for psd in seeds[1:]:
        if psd.test_grid.to_dict() != test_grid:
            logger.warning(
                "Inconsistent PriorTestGrid across seeds — using seed 0's "
                "grid in the aggregate output."
            )
            break

    out: dict[str, Any] = {
        "n_seeds":   len(seeds),
        "test_grid": test_grid,
        "methods":   {},
    }
    for m in methods:
        out["methods"][m] = {
            "global_calibration": {
                key: _maybe_summary(
                    _stack_calibration(seeds, m, "global_calibration", key)
                )
                for key in _CALIB_GLOBAL_KEYS
            },
            "prior_region": {
                key: _maybe_summary(
                    _stack_calibration(seeds, m, "prior_region", key)
                )
                for key in _CALIB_PRIOR_REGION_KEYS
            },
            "joint_posterior": {
                key: _maybe_summary(
                    _stack_calibration(seeds, m, "joint_posterior", key)
                )
                for key in _CALIB_JOINT_KEYS
            },
        }
    return out


def _stack_calibration_per_checkpoint(
    seeds: list[ProcessedPriorSeedData],
    method: str,
    bucket: str,
    key: str,
) -> tuple[list[int], np.ndarray]:
    """Stack a calibration scalar across seeds and checkpoints.

    Missing entries are encoded as ``np.nan``; the checkpoint iteration set
    is taken from seed 0 and mismatches are warned about.
    """
    if not seeds:
        return [], np.zeros((0, 0))
    ref_calibrations = seeds[0].methods[method].calibrations
    iteration_indices = [c.iteration_index for c in ref_calibrations]

    rows: list[list[float]] = []
    for psd in seeds:
        calibrations = psd.methods[method].calibrations
        seed_iter = [c.iteration_index for c in calibrations]
        if seed_iter != iteration_indices:
            logger.warning(
                "Mismatched calibration iteration set in seed %s for method "
                "'%s'; aggregate curve will use NaN for missing checkpoints.",
                psd.metadata.get("seed"), method,
            )
        by_iter = {c.iteration_index: c for c in calibrations}
        row: list[float] = []
        for it in iteration_indices:
            report = by_iter.get(it)
            bucket_dict = getattr(report, bucket) if report is not None else None
            v = bucket_dict.get(key) if bucket_dict is not None else None
            try:
                row.append(float(v) if v is not None else float("nan"))
            except (TypeError, ValueError):
                row.append(float("nan"))
        rows.append(row)
    return iteration_indices, np.asarray(rows, dtype=np.float64)


def _aggregate_scalar_curve(
    values: np.ndarray,
    confidence: float,
    n_bootstrap: int,
    band_type: str,
    statistic: str,
    random_state: Optional[int],
) -> Optional[dict]:
    """Aggregate a ``(n_seeds, n_checkpoints)`` scalar curve (NaN-tolerant)."""
    if values.size == 0:
        return None
    finite_mask = np.isfinite(values).all(axis=1)
    clean = values[finite_mask]
    if clean.shape[0] >= 2:
        agg = aggregate_convergence_curves(
            {"_": clean},
            statistic=statistic,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            band_type=band_type,
            random_state=random_state,
        )["_"]
        return _serialize_band(agg, band_type)
    n_steps = values.shape[1]
    mean = np.full(n_steps, np.nan)
    median = np.full(n_steps, np.nan)
    n = np.zeros(n_steps, dtype=int)
    for t in range(n_steps):
        col = values[:, t]
        col = col[np.isfinite(col)]
        n[t] = int(len(col))
        if len(col) > 0:
            mean[t] = float(np.mean(col))
            median[t] = float(np.median(col))
    return {
        "center":    median.tolist() if statistic == "median" else mean.tolist(),
        "lower":     [None] * n_steps,
        "upper":     [None] * n_steps,
        "band_type": "none",
        "statistic": statistic,
        "n_per_step": n.tolist(),
        "note": "insufficient complete rows for bootstrap; pointwise summary only",
    }


def _band_type_for_calibration_key(
    key: str,
    calibration_band_type: str,
    residual_band_type: str,
) -> str:
    return residual_band_type if key in _RESIDUAL_METRIC_KEYS else calibration_band_type


def _build_calibration_curves(
    seeds: list[ProcessedPriorSeedData],
    methods: list[str],
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    calibration_band_type: str = "pointwise",
    residual_band_type: str = "uniform",
    random_state: Optional[int] = 0,
) -> dict[str, Any]:
    """Per-method calibration curves over GP checkpoints, across seeds."""
    prior_region_keys = _CALIB_GLOBAL_KEYS + _CALIB_PRIOR_REGION_JOINT_KEYS

    out: dict[str, Any] = {
        "n_seeds":     len(seeds),
        "statistic":   statistic,
        "confidence":  confidence,
        "calibration_band_type": calibration_band_type,
        "residual_band_type":    residual_band_type,
        "n_bootstrap": int(n_bootstrap),
        "methods":     {},
    }

    def _agg_level(m: str, bucket: str, keys) -> dict[str, Any]:
        level: dict[str, Any] = {}
        for key in keys:
            _, values = _stack_calibration_per_checkpoint(seeds, m, bucket, key)
            bt = _band_type_for_calibration_key(
                key, calibration_band_type, residual_band_type
            )
            agg = _aggregate_scalar_curve(
                values, confidence, n_bootstrap, bt, statistic, random_state
            )
            if agg is not None:
                level[key] = agg
        return level

    for m in methods:
        ref_it = [c.iteration_index for c in seeds[0].methods[m].calibrations]
        out["methods"][m] = {
            "iteration":          ref_it,
            "global_calibration": _agg_level(m, "global_calibration", _CALIB_GLOBAL_KEYS),
            "prior_region":       _agg_level(m, "prior_region", prior_region_keys),
            "joint_posterior":    _agg_level(m, "joint_posterior", _CALIB_JOINT_KEYS),
        }
    return out


def _build_statistical_tests(
    seeds: list[ProcessedPriorSeedData],
    methods: list[str],
    baseline: str,
) -> dict[str, Any]:
    pairs = _pairs_from_methods(methods, baseline)
    out: dict[str, Any] = {
        "n_seeds":     len(seeds),
        "baseline":    baseline,
        "comparisons": [],
    }
    if not pairs:
        logger.warning(
            "Baseline method '%s' not found among %s; no statistical tests "
            "produced.", baseline, methods,
        )
    for method, base in pairs:
        per_metric: dict[str, Any] = {}
        for key in _TEST_SCALAR_KEYS:
            m_vals = _stack_scalars(seeds, method, key)
            b_vals = _stack_scalars(seeds, base, key)
            if not m_vals or not b_vals:
                continue
            m_paired, b_paired = _stack_scalars_paired(seeds, method, base, key)
            per_metric[key] = {
                "n_method":   len(m_vals),
                "n_baseline": len(b_vals),
                "n_paired":   len(m_paired),
                # Primary paired test: smaller metric is better, so the
                # method improves on the baseline when baseline - method > 0.
                "wilcoxon_paired": (
                    wilcoxon_paired_comparison(
                        m_paired, b_paired, alternative="greater"
                    )
                    if len(m_paired) > 0 else None
                ),
                # Secondary, unpaired tests.
                "mann_whitney": mann_whitney_comparison(
                    m_vals, b_vals, alternative="less"
                ),
                "rank_biserial": rank_biserial_correlation(m_vals, b_vals),
                "cohens_d_method_better": cohens_d_right_better(m_vals, b_vals),
            }
        out["comparisons"].append({
            "method":   method,
            "baseline": base,
            "metrics":  per_metric,
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
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline method for statistical tests.  Falls "
                             "back to config['aggregate']['baseline_method'] "
                             "or 'vanilla_ucb'.")
    band_choices = ("pointwise", "uniform", "both")
    parser.add_argument("--convergence-band-type", choices=band_choices, default=None)
    parser.add_argument("--calibration-band-type", choices=band_choices, default=None)
    parser.add_argument("--residual-band-type", choices=band_choices, default=None)
    return parser.parse_args(argv)


def _resolve_band_types(
    aggregate_cfg: dict[str, Any], args: argparse.Namespace
) -> dict[str, str]:
    """Resolve per-family bootstrap band types (CLI > config > default)."""
    bands = aggregate_cfg.get("bands", {})

    def pick(cli_value: Optional[str], flat_key: str, nested_key: str, default: str) -> str:
        if cli_value is not None:
            return cli_value
        if flat_key in aggregate_cfg:
            return str(aggregate_cfg[flat_key])
        if nested_key in bands:
            return str(bands[nested_key])
        return default

    return {
        "convergence": pick(args.convergence_band_type,
                            "convergence_band_type", "convergence", "pointwise"),
        "calibration": pick(args.calibration_band_type,
                            "calibration_band_type", "calibration", "pointwise"),
        "residual":    pick(args.residual_band_type,
                            "residual_band_type", "residuals", "uniform"),
    }


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args(argv)
    run_dir: Path = args.run_dir.resolve()

    config = load_json(run_dir / "config.json")
    aggregate_cfg = config.get("aggregate", {})
    band_types = _resolve_band_types(aggregate_cfg, args)
    n_bootstrap = (
        args.n_bootstrap if args.n_bootstrap is not None
        else int(aggregate_cfg.get("n_bootstrap", 5000))
    )
    confidence = (
        args.confidence if args.confidence is not None
        else float(aggregate_cfg.get("confidence", 0.95))
    )
    baseline = (
        args.baseline if args.baseline is not None
        else str(aggregate_cfg.get("baseline_method", "vanilla_ucb"))
    )
    aggregate_calibration_curves = bool(
        aggregate_cfg.get("aggregate_calibration_curves", True)
    )
    aggregate_instantaneous_curves = bool(
        aggregate_cfg.get("aggregate_instantaneous_curves", True)
    )

    run_logger = RunLogger(run_dir)
    benchmarks = _discover_processed(run_dir)
    if args.benchmark is not None:
        benchmarks = {k: v for k, v in benchmarks.items() if k in args.benchmark}
    if not benchmarks:
        logger.warning("No processed seed files matched the filters; nothing to do.")
        return

    for benchmark, paths in benchmarks.items():
        seeds = [ProcessedPriorSeedData.from_dict(load_json(p)) for p in paths]
        method_names = list(seeds[0].methods.keys())
        logger.info(
            f"[{benchmark}] aggregating {len(seeds)} seed(s); "
            f"methods={method_names}"
        )

        save_json(
            _build_final_performance(seeds, method_names),
            run_logger.processed_aggregate_path(benchmark, "final_performance.json"),
        )
        n_written = 3
        if aggregate_instantaneous_curves:
            save_json(
                _build_convergence_curves(
                    seeds, method_names,
                    statistic=args.statistic,
                    confidence=confidence,
                    n_bootstrap=n_bootstrap,
                    band_type=band_types["convergence"],
                ),
                run_logger.processed_aggregate_path(
                    benchmark, "convergence_curves.json"
                ),
            )
            n_written += 1
        save_json(
            _build_calibration_summary(seeds, method_names),
            run_logger.processed_aggregate_path(benchmark, "calibration_summary.json"),
        )
        save_json(
            _build_statistical_tests(seeds, method_names, baseline),
            run_logger.processed_aggregate_path(benchmark, "statistical_tests.json"),
        )
        if aggregate_calibration_curves:
            save_json(
                _build_calibration_curves(
                    seeds, method_names,
                    statistic=args.statistic,
                    confidence=confidence,
                    n_bootstrap=n_bootstrap,
                    calibration_band_type=band_types["calibration"],
                    residual_band_type=band_types["residual"],
                ),
                run_logger.processed_aggregate_path(
                    benchmark, "calibration_curves.json"
                ),
            )
            n_written += 1
        logger.info(
            f"  -> wrote {n_written} aggregate files to "
            f"processed/aggregate/{benchmark}"
        )


if __name__ == "__main__":
    main()
