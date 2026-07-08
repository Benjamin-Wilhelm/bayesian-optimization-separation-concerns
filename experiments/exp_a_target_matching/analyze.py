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
- ``convergence_curves.json``    — per-method performance/convergence curves only
  (target distance, instantaneous distance, cumulative regrets).  Band type from
  ``aggregate.convergence_band_type`` (default ``"pointwise"``).
- ``calibration_summary.json``   — per-method descriptive stats on calibration scalars
  at the final checkpoint, for the global / near-target / joint-posterior levels.
- ``statistical_tests.json``     — pairwise paired Wilcoxon (primary) plus
  Mann-Whitney, rank-biserial, Cohen's d (secondary).
- ``calibration_curves.json``    — per-method calibration curves only (NLPD, coverage,
  violation rate, residual/PIT stats, joint NLPD).  Non-residual curves use
  ``aggregate.calibration_band_type`` (default ``"pointwise"``); residual curves use
  ``aggregate.residual_band_type`` (default ``"uniform"``).  Written only when
  ``aggregate.aggregate_calibration_curves`` is true.

The two curve files are disjoint: performance curves never appear in
``calibration_curves.json`` and calibration scalars never appear in
``convergence_curves.json``.
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
    wilcoxon_paired_comparison,
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
# Note: ``coverage`` is the empirical coverage of the GP predictive interval
# used for UCB-style acquisitions.
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
# Near-target joint-posterior scalars (computed only when both
# ``near_target_epsilon`` and ``m_test_joint_near_target`` are set).
_CALIB_NEAR_TARGET_JOINT_KEYS: list[str] = [
    "joint_nlpd",
    "joint_mahalanobis_sq",
    "joint_logdet",
    "joint_n_test",
    "joint_total_count",
]
_CALIB_NEAR_TARGET_KEYS: list[str] = _CALIB_GLOBAL_KEYS + [
    "near_target_count",
    "near_target_fraction",
] + _CALIB_NEAR_TARGET_JOINT_KEYS

# Joint-posterior scalars (computed only when m_test_joint_distrib is set).
_CALIB_JOINT_KEYS: list[str] = [
    "nlpd_joint",
    "mahalanobis_sq",
    "logdet",
]

# Standardized-residual calibration scalars.  These are the only calibration
# curves that may use a *uniform* (simultaneous) bootstrap band: a single
# excursion along the checkpoints should not be over-interpreted.  All other
# calibration curves (nlpd, coverage, violation rate, pit, joint posterior)
# stay pointwise.  Future residual quantiles are listed defensively.
_RESIDUAL_METRIC_KEYS: set[str] = {
    "mean_standardized_residual",
    "std_standardized_residual",
    "mean_abs_standardized_residual",
    "residual_q05",
    "residual_q50",
    "residual_q95",
}


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


def _stack_scalars_paired(
    seeds: list[ProcessedSeedData],
    right: str,
    wrong: str,
    key: str,
) -> tuple[list[float], list[float]]:
    """Seed-aligned values of one final scalar for the RIGHT/WRONG pair.

    A seed contributes only if **both** methods have a non-``None`` value for
    ``key`` on that seed, so the returned lists are paired index-for-index —
    a prerequisite for :func:`wilcoxon_paired_comparison`.
    """
    r_out: list[float] = []
    w_out: list[float] = []
    for psd in seeds:
        rv = getattr(psd.methods[right].final, key, None)
        wv = getattr(psd.methods[wrong].final, key, None)
        if rv is None or wv is None:
            continue
        r_out.append(float(rv))
        w_out.append(float(wv))
    return r_out, w_out


def _stack_calibration(
    seeds: list[ProcessedSeedData],
    method: str,
    bucket: str,
    key: str,
) -> list[float]:
    """Per-seed values of one calibration scalar from the *final* checkpoint
    of ``CalibrationReport`` list (``calibrations[-1]``), dropping ``NaN``
    and ``None``.

    ``bucket`` is the attribute name on :class:`CalibrationReport`, either
    ``"global_calibration"`` or ``"near_target"``.
    """
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


def _band_to_jsonable(band: dict) -> dict:
    """Convert numpy arrays in a band dict into JSON-serialisable lists."""
    out: dict = {}
    for k, v in band.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _serialize_band(agg: dict, band_type: str) -> dict:
    """Convert a band returned by :func:`aggregate_convergence_curves` to a
    JSON-serialisable nested dict, handling the single-band and ``"both"``
    cases uniformly.
    """
    if band_type == "both":
        return {
            "pointwise": _band_to_jsonable(agg["pointwise"]),
            "uniform":   _band_to_jsonable(agg["uniform"]),
        }
    return _band_to_jsonable(agg)


def _build_convergence_curves(
    seeds: list[ProcessedSeedData],
    methods: list[str],
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    band_type: str = "pointwise",
) -> dict[str, Any]:
    curve_keys = [
        "instantaneous_target_distance",
        "running_best_target_distance",
        "cumulative_inst_target_regret",
        "cumulative_bsf_target_regret",
    ]
    iteration = seeds[0].methods[methods[0]].convergence.iteration

    out: dict[str, Any] = {
        "n_seeds":     len(seeds),
        "statistic":   statistic,
        "confidence":  confidence,
        "band_type":   band_type,
        "n_bootstrap": int(n_bootstrap),
        "iteration":   iteration,
        "methods":     {},
    }
    for m in methods:
        method_curves: dict[str, Any] = {}
        for key in curve_keys:
            curves = _stack_curves(seeds, m, key)
            if curves.shape[0] == 0:
                continue
            agg = aggregate_convergence_curves(
                {m: curves},
                statistic=statistic,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                band_type=band_type,
            )[m]
            serialized = _serialize_band(agg, band_type)
            # Pointwise mean across seeds, stored alongside the (median) center
            # so figures can overlay it as a secondary dashed curve.  NaN-robust.
            serialized["mean"] = np.nanmean(curves, axis=0).tolist()
            method_curves[key] = serialized
        out["methods"][m] = method_curves
    return out


def _build_calibration_summary(
    seeds: list[ProcessedSeedData],
    methods: list[str],
) -> dict[str, Any]:
    """Aggregate calibration scalars across seeds, on the final checkpoint.

    Includes the shared :class:`TestGrid` metadata once at the top so
    downstream consumers know what ``coverage`` etc. are computed against.
    """
    # All seeds in a run should share the same test grid; warn if not.
    test_grid = seeds[0].test_grid.to_dict()
    for psd in seeds[1:]:
        if psd.test_grid.to_dict() != test_grid:
            logger.warning(
                "Inconsistent TestGrid across seeds — using seed 0's grid in "
                "the aggregate output. Heterogeneous test grids will make "
                "coverage values not directly comparable."
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
            "near_target": {
                key: _maybe_summary(
                    _stack_calibration(seeds, m, "near_target", key)
                )
                for key in _CALIB_NEAR_TARGET_KEYS
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
    seeds: list[ProcessedSeedData],
    method: str,
    bucket: str,
    key: str,
) -> tuple[list[int], np.ndarray]:
    """Stack a calibration scalar across seeds **and** checkpoints.

    Returns:
        ``(iteration_indices, values)`` where:

        - ``iteration_indices`` is the list of checkpoint iterations taken
          from seed 0 (assumed shared across seeds — warned otherwise).
        - ``values`` is a 2-D ``(n_seeds, n_checkpoints)`` array of floats,
          with missing entries encoded as ``np.nan``.
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
            if report is None:
                row.append(float("nan"))
                continue
            bucket_dict = getattr(report, bucket)
            if bucket_dict is None:
                row.append(float("nan"))
                continue
            v = bucket_dict.get(key)
            if v is None:
                row.append(float("nan"))
                continue
            try:
                row.append(float(v))
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
    """Aggregate a ``(n_seeds, n_checkpoints)`` scalar curve.

    Rows containing only NaNs are dropped.  Columns with any remaining NaN
    are aggregated using only the valid rows of that column — when this
    causes the dimensionality to change, the function falls back to a
    per-column pointwise summary.  Returns ``None`` when nothing usable
    remains.
    """
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
    # Fallback: per-column descriptive summary, skipping NaN.
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
    """Pick the bootstrap band type for one calibration scalar.

    Standardized-residual metrics (see :data:`_RESIDUAL_METRIC_KEYS`) use
    ``residual_band_type``; every other calibration metric (nlpd, coverage,
    violation rate, pit, joint posterior) uses ``calibration_band_type``.
    """
    return residual_band_type if key in _RESIDUAL_METRIC_KEYS else calibration_band_type


def _build_calibration_curves(
    seeds: list[ProcessedSeedData],
    methods: list[str],
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    calibration_band_type: str = "pointwise",
    residual_band_type: str = "uniform",
    random_state: Optional[int] = 0,
) -> dict[str, Any]:
    """Per-method calibration curves aggregated across seeds.

    Aggregates the scalar fields of ``CalibrationReport.global_calibration``
    (and ``.near_target`` / ``.joint_posterior``) over the GP snapshot
    checkpoints.  No per-test-point quantities are emitted.

    Band type is chosen *per metric*: residual curves use
    ``residual_band_type`` (default ``"uniform"``); all other calibration
    curves use ``calibration_band_type`` (default ``"pointwise"``).  See
    :func:`_band_type_for_calibration_key`.
    """
    global_keys = _CALIB_GLOBAL_KEYS
    # Per-checkpoint near-target curves track the standard calibration scalars
    # plus the near-target joint NLPD scalars.
    near_target_keys = _CALIB_GLOBAL_KEYS + _CALIB_NEAR_TARGET_JOINT_KEYS
    joint_keys = _CALIB_JOINT_KEYS

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
        # Reference iteration list = checkpoints of method m, seed 0.
        ref_it = [c.iteration_index for c in seeds[0].methods[m].calibrations]
        method_out: dict[str, Any] = {
            "iteration":          ref_it,
            "global_calibration": _agg_level(m, "global_calibration", global_keys),
            "near_target":        _agg_level(m, "near_target", near_target_keys),
            "joint_posterior":    _agg_level(m, "joint_posterior", joint_keys),
        }

        out["methods"][m] = method_out
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
            # Paired arrays for the Wilcoxon signed-rank test: seeds share the
            # initial design, so we compare RIGHT vs WRONG seed-for-seed.
            r_paired, w_paired = _stack_scalars_paired(seeds, right, wrong, key)
            entry: dict[str, Any] = {
                "n_right":      len(r),
                "n_wrong":      len(w),
                "n_paired":     len(r_paired),
                # Primary paired test (smaller metric is better => RIGHT better
                # when wrong - right > 0).
                "wilcoxon_paired": (
                    wilcoxon_paired_comparison(
                        r_paired, w_paired, alternative="greater"
                    )
                    if len(r_paired) > 0 else None
                ),
                # Secondary, unpaired tests kept for compatibility.
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
    parser.add_argument("--confidence", type=float, default=None,
                        help="CI level. Falls back to "
                             "config['aggregate']['confidence'] or 0.95.")
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="Bootstrap resamples. Falls back to "
                             "config['aggregate']['n_bootstrap'] or 5000.")
    band_choices = ("pointwise", "uniform", "both")
    parser.add_argument("--convergence-band-type", choices=band_choices, default=None,
                        help="Band type for performance/convergence curves. "
                             "Falls back to config or 'pointwise'.")
    parser.add_argument("--calibration-band-type", choices=band_choices, default=None,
                        help="Band type for non-residual calibration curves. "
                             "Falls back to config or 'pointwise'.")
    parser.add_argument("--residual-band-type", choices=band_choices, default=None,
                        help="Band type for standardized-residual curves. "
                             "Falls back to config or 'uniform'.")
    return parser.parse_args(argv)


def _resolve_band_types(aggregate_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Resolve per-family bootstrap band types.

    Precedence: CLI flag > flat config key > nested ``bands:`` config key >
    built-in default.  There is intentionally **no** single global band type;
    each curve family is configured independently so that e.g. residual
    curves can use a uniform band while performance curves stay pointwise.
    """
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
    failure_threshold = config.get("failure_threshold")
    if failure_threshold is not None:
        failure_threshold = float(failure_threshold)

    # Aggregation options: CLI overrides config['aggregate'] which overrides
    # the built-in defaults.  Band types are resolved per curve family — there
    # is no single global band type.
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
        n_written = 4
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
        save_json(
            _build_calibration_summary(seeds, method_names),
            run_logger.processed_aggregate_path(benchmark, "calibration_summary.json"),
        )
        save_json(
            _build_statistical_tests(seeds, method_names, failure_threshold),
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
