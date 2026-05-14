"""Processed per-seed dataclasses for one (experiment, benchmark, seed).

A processed seed JSON is **derived entirely** from the corresponding raw
seed file by ``process.py``.  No new experimental information is introduced;
no GP refitting is performed.  Across-seed aggregation lives in
``analyze.py`` and uses ``common/statistics.py``.

Canonical JSON structure::

    {
      "metadata":          {SeedMetadata fields, copied from raw},
      "target":            float,
      "epsilon":           float | null,
      "failure_threshold": float | null,
      "oracle_best":       float,
      "test_grid":         {"n_test": int, "seed": int},
      "methods": {
        "<method_name>": {
          "train_label": ...,
          "final":       {FinalScalars},
          "convergence": {ConvergenceCurves},
          "calibration": {CalibrationReport}
        }
      }
    }
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional


# ---------------------------------------------------------------------------
# FinalScalars  — per-method, per-seed scalars
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FinalScalars:
    """Scalar end-of-run metrics for one method under one seed.

    All quantities are computed over the **full evaluation history**
    (initial design + BO budget), using ``f_noiseless`` when available.

    Attributes:
        n_evaluations: ``initial_design_size + budget``.
        final_best_target_distance: ``min_t |f(x_t) - T|``.
        final_target_simple_regret: ``final_best_target_distance - oracle_best``.
        mean_inst_target_distance: Mean of per-step ``|f(x_t) - T|``.
        median_inst_target_distance: Median of per-step ``|f(x_t) - T|``.
        worst_inst_target_distance: ``max_t |f(x_t) - T|``.
        cumulative_inst_target_regret: ``sum_t (|f(x_t) - T| - oracle_best)``.
        cumulative_bsf_target_regret: Same but on running-best distances.
        auc_running_target_regret: Trapezoidal area under the running-best
            target-regret curve.
        hitting_time: First iteration at which the running best is within
            ``epsilon`` of the target; ``None`` if ``epsilon`` was not set,
            ``n_evaluations + 1`` if the threshold was never hit.
        failure_indicator: ``1`` if ``final_best_target_distance >
            failure_threshold``, else ``0``; ``None`` if no threshold was set.
        best_x: Query point that achieved ``final_best_target_distance``.
    """

    n_evaluations: int
    final_best_target_distance: float
    final_target_simple_regret: float
    mean_inst_target_distance: float
    median_inst_target_distance: float
    worst_inst_target_distance: float
    cumulative_inst_target_regret: float
    cumulative_bsf_target_regret: float
    auc_running_target_regret: float
    hitting_time: Optional[int]
    failure_indicator: Optional[int]
    best_x: list[float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FinalScalars":
        return cls(**d)


# ---------------------------------------------------------------------------
# ConvergenceCurves  — per-method, per-seed time series
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConvergenceCurves:
    """Per-iteration time series for one method under one seed.

    All lists have the same length ``n_evaluations = initial_design_size +
    budget``.  ``iteration`` is 1-based and indexes the full evaluation
    history.  Useful as inputs to :func:`common.statistics.bootstrap_ci_curves`.

    Attributes:
        iteration: Iteration index, 1..n_evaluations.
        instantaneous_target_distance: ``|f(x_t) - T|``.
        running_best_target_distance: ``min_{s<=t} |f(x_s) - T|``.
        cumulative_inst_target_regret: ``sum_{s<=t} (|f(x_s) - T| -
            oracle_best)``.
        cumulative_bsf_target_regret: ``sum_{s<=t} (min_{r<=s} |f(x_r) - T| -
            oracle_best)``.
    """

    iteration:                       list[int]
    instantaneous_target_distance:   list[float]
    running_best_target_distance:    list[float]
    cumulative_inst_target_regret:   list[float]
    cumulative_bsf_target_regret:    list[float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConvergenceCurves":
        return cls(**d)


# ---------------------------------------------------------------------------
# CalibrationReport  — per-method, per-seed calibration on a held-out grid
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CalibrationPredictionGrid :
    X_test: list[list[float]]
    y_true: list[float]
    posterior_mean: list[float]
    posterior_std: list[float]

@dataclasses.dataclass
class CalibrationReport:
    """GP calibration diagnostics on a fixed held-out test grid.

    The test grid is produced by ``benchmark.test_set(n_test, seed)`` and
    is shared across methods for one seed.  GP predictions are obtained by
    rebuilding the GP via ``load_state_dict`` — no refitting.

    For the RIGHT method (GP on raw ``y_obs``) the ground truth is
    ``f(X_test)``.  For the WRONG method (GP on ``|y_obs - T|``) it is
    ``|f(X_test) - T|``.  ``process.py`` picks the right one based on
    ``MethodTrajectory.train_label``.

    Attributes:
        iteration_index: BO iteration at which the GP was snapshotted
            (usually the final iteration, but other indices may be added).
        n_test: Number of held-out test points.
        test_grid_seed: Sobol seed used for the test grid.
        test_grid: Output of :func:`common.diagnostics.calibration_summary`
            over all test points (NLPD, coverage, residual stats, PIT,
            violation rate).
        near_target: Output of
            :func:`common.diagnostics.near_target_calibration_summary` —
            calibration restricted to test points with
            ``|f - T| <= epsilon_near``.  ``None`` when ``epsilon_near`` is
            not configured.
    """

    iteration_index: int
    n_test:          int
    test_grid_seed:  int
    test_grid:       dict[str, float]
    near_target:     Optional[dict[str, Any]]
    predictions: Optional[CalibrationPredictionGrid] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationReport":
        raw_pred = d.get("predictions")
        predictions = (
            CalibrationPredictionGrid(**raw_pred) if raw_pred is not None else None
        )
        return cls(
            iteration_index=d["iteration_index"],
            n_test=d["n_test"],
            test_grid_seed=d["test_grid_seed"],
            test_grid=d["test_grid"],
            near_target=d.get("near_target"),
            predictions=predictions,
        )


# ---------------------------------------------------------------------------
# MethodProcessed  — one method under one seed
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MethodProcessed:
    """Processed metrics + diagnostics for one method under one seed.

    Attributes:
        method_name: Identifier used as the JSON key.
        train_label: Copied from :class:`data.raw.MethodTrajectory` so the
            processed file is self-describing.
        final: Scalar end-of-run metrics.
        convergence: Per-iteration convergence time series.
        calibration: Held-out grid calibration diagnostics.
    """

    method_name:  str
    train_label:  str
    final:        FinalScalars
    convergence:  ConvergenceCurves
    calibrations:  list[CalibrationReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_label": self.train_label,
            "final":       self.final.to_dict(),
            "convergence": self.convergence.to_dict(),
            "calibrations": [c.to_dict() for c in self.calibrations],
        }

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "MethodProcessed":
        return cls(
            method_name=name,
            train_label=d["train_label"],
            final=FinalScalars.from_dict(d["final"]),
            convergence=ConvergenceCurves.from_dict(d["convergence"]),
            calibrations=[CalibrationReport.from_dict(c) for c in d["calibrations"]],
        )


# ---------------------------------------------------------------------------
# ProcessedSeedData  — top-level
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProcessedSeedData:
    """All processed quantities for one (experiment, benchmark, seed).

    ``metadata`` is a verbatim copy of :class:`data.raw.SeedMetadata`'s dict
    so the processed file is self-contained — analyze.py never has to look
    back at the raw file.

    Attributes:
        metadata: Copy of the raw seed metadata.
        target: Target value ``T``.
        epsilon: Hitting-time tolerance, or ``None`` if not configured.
        failure_threshold: Failure criterion, or ``None`` if not configured.
        oracle_best: ``min_x |f(x) - T|``.  Zero if the target is attainable;
            otherwise the benchmark's analytic value, when known.
        test_grid: Grid configuration ``{"n_test": int, "seed": int}``.
        methods: Per-method processed data, keyed by method name.
    """

    metadata:          dict[str, Any]
    target:            float
    epsilon:           Optional[float]
    failure_threshold: Optional[float]
    oracle_best:       float
    test_grid:         dict[str, int]
    methods:           dict[str, MethodProcessed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata":          self.metadata,
            "target":            self.target,
            "epsilon":           self.epsilon,
            "failure_threshold": self.failure_threshold,
            "oracle_best":       self.oracle_best,
            "test_grid":         self.test_grid,
            "methods":           {n: m.to_dict() for n, m in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProcessedSeedData":
        return cls(
            metadata=d["metadata"],
            target=d["target"],
            epsilon=d.get("epsilon"),
            failure_threshold=d.get("failure_threshold"),
            oracle_best=d["oracle_best"],
            test_grid=d["test_grid"],
            methods={
                name: MethodProcessed.from_dict(name, md)
                for name, md in d.get("methods", {}).items()
            },
        )
