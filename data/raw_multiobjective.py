"""Raw-data dataclasses for Experiment C (JSON-serialisable).

Two on-disk layouts, one per experiment mode, both stored as
``raw/<benchmark>/seed_NNN.json`` (reusing :class:`common.logging.RunLogger`).

``fixed_scalarization`` — one file per (benchmark, seed):

    {
      "mode": "fixed_scalarization",
      "metadata": {...},
      "benchmark": {...},                       # MultiObjectiveBenchmark.metadata()
      "cases": [                                # one per (scalarization, lambda)
        {
          "scalarization": "linear",
          "lambda": {"angle_degrees": 45.0, "vector": [...]},
          "rho_star": <float>,
          "pairing_hash": "<hex>",              # shared by right & wrong
          "initial_design": {"X": [...], "raw_y": [...], "oriented_y": [...]},
          "methods": {"right": <MethodTrajectory>, "wrong": <MethodTrajectory>}
        }, ...
      ]
    }

``pareto_front`` — one file per (benchmark, seed):

    {
      "mode": "pareto_front",
      "metadata": {...},
      "benchmark": {...},
      "reference_point": [...],
      "hv_star": <float>,
      "evaluation_lambdas": [[...], ...],       # shared across all methods
      "methods": {
        "right_algorithm_1": <ParetoMethod>,
        "right_algorithm_2": <ParetoMethod>,
        "wrong_algorithm_2": <ParetoMethod>
      }
    }

The raw data stores enough for ``process.py`` / ``analyze.py`` to work
**without re-running BO or re-fitting a GP**: every queried ``x`` is stored,
and all Experiment C metrics are model-independent (they re-evaluate the
analytic benchmark at the stored ``x``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Fixed-scalarization records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FixedStepRecord:
    """One BO iteration of a fixed-scalarization run."""

    iteration: int
    x: list[float]
    raw_y: list[float]
    oriented_y: list[float]
    model_y: list[float]
    scalarized_value: float
    acq_value: float
    beta: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixedStepRecord":
        return cls(**d)


@dataclasses.dataclass
class MethodTrajectory:
    """One method (right/wrong) within one fixed-scalarization case."""

    method_name: str
    train_label: str
    trajectory: list[FixedStepRecord]
    gp_hyperparameters: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "train_label": self.train_label,
            "trajectory": [s.to_dict() for s in self.trajectory],
            "gp_hyperparameters": self.gp_hyperparameters,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MethodTrajectory":
        return cls(
            method_name=d["method_name"],
            train_label=d["train_label"],
            trajectory=[FixedStepRecord.from_dict(s) for s in d["trajectory"]],
            gp_hyperparameters=d.get("gp_hyperparameters", []),
        )


@dataclasses.dataclass
class FixedCase:
    """One (scalarization, lambda) case with paired right/wrong trajectories."""

    scalarization: str
    lambda_angle_degrees: float
    lambda_vector: list[float]
    rho_star: float
    pairing_hash: str
    initial_X: list[list[float]]
    initial_raw_y: list[list[float]]
    initial_oriented_y: list[list[float]]
    methods: dict[str, MethodTrajectory]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scalarization": self.scalarization,
            "lambda": {
                "angle_degrees": self.lambda_angle_degrees,
                "vector": self.lambda_vector,
            },
            "rho_star": self.rho_star,
            "pairing_hash": self.pairing_hash,
            "initial_design": {
                "X": self.initial_X,
                "raw_y": self.initial_raw_y,
                "oriented_y": self.initial_oriented_y,
            },
            "methods": {k: v.to_dict() for k, v in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixedCase":
        init = d["initial_design"]
        return cls(
            scalarization=d["scalarization"],
            lambda_angle_degrees=float(d["lambda"]["angle_degrees"]),
            lambda_vector=list(d["lambda"]["vector"]),
            rho_star=float(d["rho_star"]),
            pairing_hash=d["pairing_hash"],
            initial_X=init["X"],
            initial_raw_y=init["raw_y"],
            initial_oriented_y=init["oriented_y"],
            methods={
                k: MethodTrajectory.from_dict(v) for k, v in d["methods"].items()
            },
        )


@dataclasses.dataclass
class FixedSeedData:
    """Top-level fixed-scalarization seed file."""

    metadata: dict[str, Any]
    benchmark: dict[str, Any]
    cases: list[FixedCase]
    mode: str = "fixed_scalarization"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metadata": self.metadata,
            "benchmark": self.benchmark,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixedSeedData":
        return cls(
            metadata=d["metadata"],
            benchmark=d["benchmark"],
            cases=[FixedCase.from_dict(c) for c in d["cases"]],
            mode=d.get("mode", "fixed_scalarization"),
        )


# ---------------------------------------------------------------------------
# Pareto-front records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParetoEvaluation:
    """One objective evaluation in the global (round-robin) order."""

    global_iteration: int
    subrun_index: int
    local_iteration: int
    is_initial_design: bool
    lambda_vector: list[float]
    x: list[float]
    raw_y: list[float]
    oriented_y: list[float]
    model_y: list[float]
    acq_value: Optional[float]
    beta: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParetoEvaluation":
        return cls(**d)


@dataclasses.dataclass
class ParetoSubrunMeta:
    """Per-subrun bookkeeping for Algorithm 2 (right or wrong)."""

    subrun_index: int
    lambda_vector: list[float]
    local_budget: int
    initial_design_size: int
    pairing_hash: str
    gp_hyperparameters: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParetoSubrunMeta":
        return cls(**d)


@dataclasses.dataclass
class ParetoMethod:
    """One pareto-front method (algorithm_1 or algorithm_2).

    ``global_evaluations`` is the full evaluation history in global
    (round-robin) order; ``subruns`` is populated only for Algorithm 2.  For
    Algorithm 1 ``optimization_lambdas`` holds the per-iteration sampled
    preferences and ``subruns`` is empty.
    """

    method_name: str
    algorithm: str            # "algorithm_1" | "algorithm_2"
    architecture: str         # "right" | "wrong"
    global_evaluations: list[ParetoEvaluation]
    subruns: list[ParetoSubrunMeta]
    optimization_lambdas: list[list[float]]
    global_order: list[list[int]]   # [[subrun_index, local_iteration], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "global_evaluations": [e.to_dict() for e in self.global_evaluations],
            "subruns": [s.to_dict() for s in self.subruns],
            "optimization_lambdas": self.optimization_lambdas,
            "global_order": self.global_order,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParetoMethod":
        return cls(
            method_name=d["method_name"],
            algorithm=d["algorithm"],
            architecture=d["architecture"],
            global_evaluations=[
                ParetoEvaluation.from_dict(e) for e in d["global_evaluations"]
            ],
            subruns=[ParetoSubrunMeta.from_dict(s) for s in d.get("subruns", [])],
            optimization_lambdas=d.get("optimization_lambdas", []),
            global_order=d.get("global_order", []),
        )


@dataclasses.dataclass
class ParetoSeedData:
    """Top-level pareto-front seed file."""

    metadata: dict[str, Any]
    benchmark: dict[str, Any]
    reference_point: list[float]
    hv_star: float
    evaluation_lambdas: list[list[float]]
    methods: dict[str, ParetoMethod]
    mode: str = "pareto_front"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metadata": self.metadata,
            "benchmark": self.benchmark,
            "reference_point": self.reference_point,
            "hv_star": self.hv_star,
            "evaluation_lambdas": self.evaluation_lambdas,
            "methods": {k: v.to_dict() for k, v in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParetoSeedData":
        return cls(
            metadata=d["metadata"],
            benchmark=d["benchmark"],
            reference_point=d["reference_point"],
            hv_star=float(d["hv_star"]),
            evaluation_lambdas=d["evaluation_lambdas"],
            methods={k: ParetoMethod.from_dict(v) for k, v in d["methods"].items()},
            mode=d.get("mode", "pareto_front"),
        )


__all__ = [
    "FixedStepRecord",
    "MethodTrajectory",
    "FixedCase",
    "FixedSeedData",
    "ParetoEvaluation",
    "ParetoSubrunMeta",
    "ParetoMethod",
    "ParetoSeedData",
]
