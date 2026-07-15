"""Processed per-seed dataclasses for Experiment C (multi-objective).

A processed seed JSON is **derived entirely** from the corresponding raw seed
file (``data/raw_multiobjective.py``) by ``process.py``.  No new experimental
information is introduced and no GP is refitted: every metric re-evaluates the
analytic benchmark at the stored ``x``.  Across-seed aggregation lives in
``analyze.py`` / ``common/statistics.py``.

Two on-disk layouts, one per experiment mode, mirroring the two raw layouts.

``fixed_scalarization`` — one file per (benchmark, seed)::

    {
      "mode": "fixed_scalarization",
      "metadata": {...},                          # copied from raw
      "benchmark": {...},                         # copied from raw
      "reference_point": [...],
      "cases": [                                  # one per (scalarization, lambda)
        {
          "scalarization": "linear",
          "lambda": {"angle_degrees": 45.0, "vector": [...]},
          "rho_star": <float>,
          "pairing_hash": "<hex>",
          "methods": {
            "right": {FixedMethodProcessed}, "wrong": {FixedMethodProcessed}
          }
        }, ...
      ]
    }

``pareto_front`` — one file per (benchmark, seed)::

    {
      "mode": "pareto_front",
      "metadata": {...},
      "benchmark": {...},
      "reference_point": [...],
      "hv_star": <float>,
      "n_evaluation_lambdas": <int>,
      "methods": {
        "right_algorithm_1": {ParetoMethodProcessed}, ...
      }
    }

The hierarchy of classes mirrors this structure::

    ProcessedFixedSeedData
    ├── cases: list[FixedCaseProcessed]
    │       └── methods: dict[str, FixedMethodProcessed]
    ProcessedParetoSeedData
    └── methods: dict[str, ParetoMethodProcessed]
                └── subruns: list[ParetoSubrunMeta]   (reused from raw schema)
"""

from __future__ import annotations

import dataclasses
from typing import Any

from data.raw_multiobjective import ParetoSubrunMeta

# ---------------------------------------------------------------------------
# Fixed-scalarization processed records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FixedMethodProcessed:
    """Per-method, per-seed processed curves for one fixed-scalarization case.

    All lists are aligned on ``iteration`` (1-based, initial design included).
    ``best_so_far``/``simple_regret`` are cumulative-best quantities; the
    ``cumulative_regret`` accumulates the *instantaneous* regrets.
    """

    iteration: list[int]
    n_initial_design: int
    scalarized_value: list[float]
    best_so_far: list[float]
    simple_regret: list[float]
    instantaneous_regret: list[float]
    cumulative_regret: list[float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixedMethodProcessed":
        return cls(**d)


@dataclasses.dataclass
class FixedCaseProcessed:
    """One (scalarization, lambda) case with paired right/wrong curves."""

    scalarization: str
    lambda_angle_degrees: float
    lambda_vector: list[float]
    rho_star: float
    pairing_hash: str
    methods: dict[str, FixedMethodProcessed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scalarization": self.scalarization,
            "lambda": {
                "angle_degrees": self.lambda_angle_degrees,
                "vector": self.lambda_vector,
            },
            "rho_star": self.rho_star,
            "pairing_hash": self.pairing_hash,
            "methods": {k: v.to_dict() for k, v in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixedCaseProcessed":
        return cls(
            scalarization=d["scalarization"],
            lambda_angle_degrees=float(d["lambda"]["angle_degrees"]),
            lambda_vector=list(d["lambda"]["vector"]),
            rho_star=float(d["rho_star"]),
            pairing_hash=d["pairing_hash"],
            methods={
                k: FixedMethodProcessed.from_dict(v) for k, v in d["methods"].items()
            },
        )


@dataclasses.dataclass
class ProcessedFixedSeedData:
    """Top-level processed fixed-scalarization seed file."""

    metadata: dict[str, Any]
    benchmark: dict[str, Any]
    reference_point: list[float]
    cases: list[FixedCaseProcessed]
    mode: str = "fixed_scalarization"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metadata": self.metadata,
            "benchmark": self.benchmark,
            "reference_point": self.reference_point,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProcessedFixedSeedData":
        return cls(
            metadata=d["metadata"],
            benchmark=d["benchmark"],
            reference_point=list(d["reference_point"]),
            cases=[FixedCaseProcessed.from_dict(c) for c in d["cases"]],
            mode=d.get("mode", "fixed_scalarization"),
        )


# ---------------------------------------------------------------------------
# Pareto-front processed records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParetoMethodProcessed:
    """Per-method, per-seed processed curves for the pareto-front mode.

    Curves are aligned on ``global_iteration`` (global round-robin order).
    ``subruns`` is populated only for Algorithm 2 (empty for Algorithm 1) and
    reuses :class:`data.raw_multiobjective.ParetoSubrunMeta`.
    """

    algorithm: str
    architecture: str
    global_iteration: list[int]
    hypervolume: list[float]
    hypervolume_regret: list[float]
    empirical_bayes_regret: list[float]
    subruns: list[ParetoSubrunMeta]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "global_iteration": self.global_iteration,
            "hypervolume": self.hypervolume,
            "hypervolume_regret": self.hypervolume_regret,
            "empirical_bayes_regret": self.empirical_bayes_regret,
            "subruns": [s.to_dict() for s in self.subruns],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParetoMethodProcessed":
        return cls(
            algorithm=d["algorithm"],
            architecture=d["architecture"],
            global_iteration=list(d["global_iteration"]),
            hypervolume=list(d["hypervolume"]),
            hypervolume_regret=list(d["hypervolume_regret"]),
            empirical_bayes_regret=list(d["empirical_bayes_regret"]),
            subruns=[ParetoSubrunMeta.from_dict(s) for s in d.get("subruns", [])],
        )


@dataclasses.dataclass
class ProcessedParetoSeedData:
    """Top-level processed pareto-front seed file."""

    metadata: dict[str, Any]
    benchmark: dict[str, Any]
    reference_point: list[float]
    hv_star: float
    n_evaluation_lambdas: int
    methods: dict[str, ParetoMethodProcessed]
    mode: str = "pareto_front"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metadata": self.metadata,
            "benchmark": self.benchmark,
            "reference_point": self.reference_point,
            "hv_star": self.hv_star,
            "n_evaluation_lambdas": self.n_evaluation_lambdas,
            "methods": {k: v.to_dict() for k, v in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProcessedParetoSeedData":
        return cls(
            metadata=d["metadata"],
            benchmark=d["benchmark"],
            reference_point=list(d["reference_point"]),
            hv_star=float(d["hv_star"]),
            n_evaluation_lambdas=int(d["n_evaluation_lambdas"]),
            methods={
                k: ParetoMethodProcessed.from_dict(v) for k, v in d["methods"].items()
            },
            mode=d.get("mode", "pareto_front"),
        )


__all__ = [
    "FixedMethodProcessed",
    "FixedCaseProcessed",
    "ProcessedFixedSeedData",
    "ParetoMethodProcessed",
    "ProcessedParetoSeedData",
]
