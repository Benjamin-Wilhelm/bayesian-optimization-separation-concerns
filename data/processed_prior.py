"""Processed per-seed dataclasses for Experiment B (prior-guided BO).

Derived entirely from the corresponding :mod:`data.raw_prior` file by
Experiment B's ``process.py``; no BO is rerun and no GP is refit.
Across-seed aggregation lives in Experiment B's ``analyze.py``.

Differences from :mod:`data.processed` (Experiment A):

- target-matching metrics are replaced by standard BO metrics under the
  **maximization** convention on the raw oriented objective (``f_opt =
  max_x g(x)`` on the natural scale):

  .. math::

      r_t = f^* - f(x_t)                       \\quad \\text{(instantaneous)}
      R_t = f^* - \\max_{s \\le t} f(x_s)       \\quad \\text{(simple regret)}

- the near-target calibration bucket is replaced by **prior-region**
  calibration on ``{x : pi(x) >= threshold}``;
- per-method lambda / prior-value trajectories are stored as curves.

:class:`~data.processed.CalibrationPredictionGrid` is reused verbatim.

Canonical JSON structure::

    {
      "metadata":   {PriorSeedMetadata fields, copied from raw},
      "f_opt":      float,
      "epsilon":    float | null,
      "test_grid":  {PriorTestGrid fields},
      "methods": {
        "<method_name>": {
          "train_label": "y_obs",
          "model_kind":  ...,
          "final":       {PriorFinalScalars},
          "convergence": {PriorConvergenceCurves},
          "calibrations": [PriorCalibrationReport, ...]
        }
      }
    }
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from data.processed import CalibrationPredictionGrid


# ---------------------------------------------------------------------------
# PriorFinalScalars  — per-method, per-seed scalars
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorFinalScalars:
    """Scalar end-of-run BO metrics for one method under one seed.

    All quantities are computed over the **full evaluation history**
    (initial design + BO budget) on the noiseless oriented values
    ``f_noiseless`` (raw scale, maximization convention, oracle ``f_opt``).

    Attributes:
        n_evaluations: ``initial_design_size + budget``.
        final_best_value: ``max_t f(x_t)``.
        final_simple_regret: ``f_opt - final_best_value``.
        mean_inst_regret: Mean of per-step ``f_opt - f(x_t)``.
        median_inst_regret: Median of per-step instantaneous regret.
        worst_inst_regret: ``max_t (f_opt - f(x_t))``.
        cumulative_inst_regret: ``sum_t (f_opt - f(x_t))``.
        auc_simple_regret: Trapezoidal area under the running simple-regret
            curve.
        hitting_time: First iteration with running simple regret <=
            ``epsilon``; ``None`` if ``epsilon`` was not configured,
            ``n_evaluations + 1`` if never hit.
        best_x: Query point achieving ``final_best_value``.
        lambda_final: Final fitted ``lambda`` (prior-mean methods only).
        mean_prior_value: Mean of ``pi(x_t)`` over the BO iterations —
            summarizes how strongly the method sampled inside the prior
            region.
    """

    n_evaluations: int
    final_best_value: float
    final_simple_regret: float
    mean_inst_regret: float
    median_inst_regret: float
    worst_inst_regret: float
    cumulative_inst_regret: float
    auc_simple_regret: float
    hitting_time: Optional[int]
    best_x: list[float]
    lambda_final: Optional[float]
    mean_prior_value: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorFinalScalars":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# PriorConvergenceCurves  — per-method, per-seed time series
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorConvergenceCurves:
    """Per-iteration time series for one method under one seed.

    The first four lists share length ``n_evaluations = initial_design_size
    + budget`` and are indexed by ``iteration`` (1-based over the full
    evaluation history).  The prior/lambda curves only exist for BO
    iterations and are indexed by ``bo_iteration`` (1-based over the
    budget); ``lambda_per_iteration`` is ``None`` for standard-GP methods.

    Attributes:
        iteration: 1..n_evaluations.
        instantaneous_regret: ``f_opt - f(x_t)``.
        running_best_value: ``max_{s<=t} f(x_s)``.
        simple_regret: ``f_opt - max_{s<=t} f(x_s)``.
        cumulative_inst_regret: ``sum_{s<=t} (f_opt - f(x_s))``.
        bo_iteration: 1..budget.
        prior_value_at_candidate: ``pi(x_t)`` per BO iteration.
        lambda_per_iteration: Fitted ``lambda`` per BO iteration, or ``None``.
    """

    iteration:                list[int]
    instantaneous_regret:     list[float]
    running_best_value:       list[float]
    simple_regret:            list[float]
    cumulative_inst_regret:   list[float]

    bo_iteration:             list[int]
    prior_value_at_candidate: list[float]
    lambda_per_iteration:     Optional[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorConvergenceCurves":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# PriorTestGrid  — held-out grid metadata
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorTestGrid:
    """Metadata describing held-out test grids used for Exp B calibration.

    ``X_test`` / ``f_test`` are never stored; the global grid is
    reconstructed deterministically from ``(benchmark, n_test, seed)`` and
    the prior-region subset is obtained by masking the global grid with
    ``pi(x) >= prior_region_threshold``.

    Attributes:
        n_test: Number of points in the global held-out Sobol grid.
        seed: Sobol seed of the global grid.
        prior_region_threshold: Threshold on ``pi(x)`` defining the prior
            region; ``None`` if prior-region calibration is disabled.
        n_prior_region: Number of global-grid points inside the prior
            region (``None`` when disabled).
        m_test_joint_distrib: Optional subset size for joint-posterior
            diagnostics on the global grid.
    """

    n_test: int
    seed: int
    prior_region_threshold: Optional[float] = None
    n_prior_region: Optional[int] = None
    m_test_joint_distrib: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorTestGrid":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# PriorCalibrationReport  — per-method, per-checkpoint calibration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorCalibrationReport:
    """GP calibration diagnostics on the held-out grid at one snapshot.

    Calibration is always evaluated on the objective ``f`` (normalized
    noiseless values), for every method — including the prior-mean GPs,
    whose posterior also represents ``f``.

    Attributes:
        iteration_index: BO iteration at which the GP was snapshotted.
        global_calibration: :func:`common.diagnostics.calibration_summary`
            over the full test grid (NLPD, coverage, residuals, PIT,
            violation rate).
        prior_region: Same summary restricted to test points with
            ``pi(x) >= threshold`` — measures overconfidence in the region
            favored by the prior.  ``None`` when disabled or empty.
        joint_posterior: Optional joint-Gaussian NLPD diagnostics on a
            global-grid subset.  ``None`` when not computed.
        predictions: Optional posterior mean/std on the global grid
            (reused :class:`~data.processed.CalibrationPredictionGrid`).
    """

    iteration_index: int
    global_calibration: dict[str, Any]
    prior_region: Optional[dict[str, Any]]
    joint_posterior: Optional[dict[str, Any]] = None
    predictions: Optional[CalibrationPredictionGrid] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_index": self.iteration_index,
            "global_calibration": self.global_calibration,
            "prior_region": self.prior_region,
            "joint_posterior": self.joint_posterior,
            "predictions": (
                self.predictions.to_dict()
                if self.predictions is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorCalibrationReport":
        raw_pred = d.get("predictions")
        return cls(
            iteration_index=d["iteration_index"],
            global_calibration=d["global_calibration"],
            prior_region=d.get("prior_region"),
            joint_posterior=d.get("joint_posterior"),
            predictions=(
                CalibrationPredictionGrid.from_dict(raw_pred)
                if raw_pred is not None
                else None
            ),
        )


# ---------------------------------------------------------------------------
# PriorMethodProcessed  — one method under one seed
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorMethodProcessed:
    """Processed metrics + diagnostics for one method under one seed."""

    method_name:  str
    train_label:  str
    model_kind:   str
    final:        PriorFinalScalars
    convergence:  PriorConvergenceCurves
    calibrations: list[PriorCalibrationReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_label":  self.train_label,
            "model_kind":   self.model_kind,
            "final":        self.final.to_dict(),
            "convergence":  self.convergence.to_dict(),
            "calibrations": [c.to_dict() for c in self.calibrations],
        }

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "PriorMethodProcessed":
        return cls(
            method_name=name,
            train_label=d["train_label"],
            model_kind=d["model_kind"],
            final=PriorFinalScalars.from_dict(d["final"]),
            convergence=PriorConvergenceCurves.from_dict(d["convergence"]),
            calibrations=[
                PriorCalibrationReport.from_dict(c) for c in d["calibrations"]
            ],
        )


# ---------------------------------------------------------------------------
# ProcessedPriorSeedData  — top-level
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProcessedPriorSeedData:
    """All processed Experiment B quantities for one (benchmark, seed).

    Attributes:
        metadata: Verbatim copy of the raw seed metadata dict (contains the
            prior and orientation configuration).
        f_opt: Oracle maximum of the oriented objective on the raw scale.
        epsilon: Hitting-time tolerance on the simple regret, or ``None``.
        test_grid: Shared held-out grid description.
        methods: Per-method processed data, keyed by method name.
    """

    metadata:   dict[str, Any]
    f_opt:      float
    epsilon:    Optional[float]
    test_grid:  PriorTestGrid
    methods:    dict[str, PriorMethodProcessed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata":  self.metadata,
            "f_opt":     self.f_opt,
            "epsilon":   self.epsilon,
            "test_grid": self.test_grid.to_dict(),
            "methods":   {n: m.to_dict() for n, m in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProcessedPriorSeedData":
        return cls(
            metadata=d["metadata"],
            f_opt=d["f_opt"],
            epsilon=d.get("epsilon"),
            test_grid=PriorTestGrid.from_dict(d["test_grid"]),
            methods={
                name: PriorMethodProcessed.from_dict(name, md)
                for name, md in d.get("methods", {}).items()
            },
        )
