"""Structured dataclasses for Bayesian optimisation experiment results.

Hierarchy::

    ExperimentSeedResult
    ├── metadata:           SeedMetadata
    ├── raw_results:        {method_name -> MethodRawResult}
    └── processed_results:  {method_name -> MethodProcessedResult}

    MethodRawResult
    └── trajectory: list[StepResult]

``to_dict`` / ``from_dict`` helpers produce the canonical JSON structure::

    {
      "metadata": {...},
      "methods": {
        "method_name": {
          "trajectory":    [...],
          "final_metrics": {...}
        }
      }
    }
"""

from __future__ import annotations

import dataclasses
from typing import Any

from common.models import ModelState


@dataclasses.dataclass
class StepResult:
    """One BO iteration.

    Attributes:
        iteration: 1-based index within the BO budget.
        x: query point as a plain ``list[float]`` (length == dim).
        y_obs: noisy observation given to the BO algorithm, i.e.
            ``y_t = f(x_t) + epsilon_t``.
        f_noiseless: true benchmark value ``f(x_t)``; ``None`` for real-data
            problems where the ground truth is unavailable.
        target_distance: ``|f_noiseless - target|``; ``None`` when
            ``f_noiseless`` is unavailable.
        acq_value: acquisition function value at the selected candidate.
        train_size: number of training points used to fit the GP *before*
            this query.
        posterior_mean_at_candidate: GP posterior mean evaluated at ``x``.
        posterior_std_at_candidate: GP posterior standard deviation at ``x``.
    """

    iteration: int
    x: list[float]
    y_obs: float
    f_noiseless: float | None
    target_distance: float | None
    acq_value: float
    train_size: int
    posterior_mean_at_candidate: float
    posterior_std_at_candidate: float

    @classmethod
    def from_tensors(
        cls,
        iteration: int,
        x_cand: Any,        # torch.Tensor (1, dim)
        y_obs_val: Any,     # torch.Tensor (1, 1)
        f_val: Any,         # torch.Tensor (1, 1) or None
        acq_value: Any,     # torch.Tensor scalar
        train_size: int,
        posterior_mean: float,
        posterior_std: float,
        target: float,
        ) -> StepResult:
        """Construct from raw tensors; converts everything to plain Python.

        ``x_cand.squeeze(0).tolist()`` always produces a ``list[float]``,
        even in the 1-D case, ensuring a consistent type across problem dims.
        ``f_val`` and ``target_distance`` are ``None`` when the true function
        value is unavailable (real-data problems).
        """
        f = f_val.squeeze().item() if f_val is not None else None
        return cls(
            iteration=iteration,
            x=x_cand.squeeze(0).tolist(),
            y_obs=y_obs_val.squeeze().item(),
            f_noiseless=f,
            target_distance=abs(f - target) if f is not None else None,
            acq_value=acq_value.item(),
            train_size=train_size,
            posterior_mean_at_candidate=posterior_mean,
            posterior_std_at_candidate=posterior_std,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepResult:
        return cls(**d)


@dataclasses.dataclass
class MethodRawResult:
    """Full observation trajectory for one method under one seed.

    Attributes:
        method_name: identifier used as the JSON key, e.g.
            ``"right_target_matching_ucb"``.
        trajectory: ordered list of per-iteration results.
    """

    method_name: str
    trajectory: list[StepResult]
    model_states: list[ModelState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "trajectory": [s.to_dict() for s in self.trajectory],
            "model_states": [s.to_dict() for s in self.model_states],
        }

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> MethodRawResult:
        return cls(
            method_name=name,
            trajectory=[StepResult.from_dict(s) for s in d["trajectory"]],
            model_states=[ModelState.from_dict(s) for s in d["model_states"]],
        )


@dataclasses.dataclass
class MethodProcessedResult:
    """Aggregated final metrics for one method under one seed.

    Attributes:
        method_name: matches the key in the outer methods dict.
        best_target_distance: ``min_t |f(x_t) - target|`` over the full
            trajectory (initial design + BO budget); computed from
            ``f_noiseless`` values, not ``y_obs``.
        best_x: query point that achieved ``best_target_distance``.
        n_evaluations: total number of function evaluations
            (initial design + BO budget).
    """

    method_name: str
    best_target_distance: float | None
    best_x: list[float]
    n_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        """Return metrics dict without ``method_name`` (it is the outer key)."""
        d = dataclasses.asdict(self)
        d.pop("method_name")
        return d

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> MethodProcessedResult:
        return cls(
            method_name=name,
            best_target_distance=d.get("best_target_distance"),
            best_x=d.get("best_x", []),
            n_evaluations=d.get("n_evaluations", 0),
        )


@dataclasses.dataclass
class SeedMetadata:
    """Metadata for one (experiment, benchmark, seed) execution.

    Attributes:
        experiment: experiment identifier, e.g. ``"exp_a_target_matching"``.
        problem_type: ``"synthetic"`` or ``"real_data"``.
        benchmark: problem name, e.g. ``"sine_trend_1d"``.
        seed: random seed used for the Sobol design and BO trajectory.
        budget: number of BO iterations (excluding the initial design).
        initial_design_size: number of Sobol initial design points.
        target: target value ``T`` for the target-matching objective.
        noise_std: observation noise standard deviation.
    """

    experiment: str
    problem_type: str
    benchmark: str
    seed: int
    budget: int
    initial_design_size: int
    target: float
    noise_std: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeedMetadata:
        return cls(**d)


@dataclasses.dataclass
class ExperimentSeedResult:
    """All results for one (experiment, benchmark, seed) triple.

    Combines metadata, per-method raw trajectories, and per-method processed
    metrics.  ``to_dict`` produces the canonical JSON structure expected by
    ``RunLogger.save_seed_result``.

    Attributes:
        metadata: run-level metadata.
        raw_results: keyed by method name; each value holds the full
            trajectory.
        processed_results: keyed by method name; each value holds aggregated
            final metrics.  A method may be absent if processing has not yet
            been run.
    """

    metadata: SeedMetadata
    raw_results: dict[str, MethodRawResult]
    processed_results: dict[str, MethodProcessedResult]

    def to_dict(self) -> dict[str, Any]:
        """Produce the canonical seed JSON structure.

        The output format is::

            {
              "metadata": {...},
              "methods": {
                "<method_name>": {
                  "trajectory":    [StepResult dicts],
                  "model_states": [ModelState dicts]
                  "final_metrics": {best_target_distance, best_x, n_evaluations}
                }
              }
            }
        """
        methods: dict[str, Any] = {}
        for name, raw in self.raw_results.items():
            processed = self.processed_results.get(name)
            methods[name] = {
                "trajectory": [s.to_dict() for s in raw.trajectory],
                "model_states": [s.to_dict() for s in raw.model_states],
                "final_metrics": processed.to_dict() if processed is not None else {},
                
            }
        return {
            "metadata": self.metadata.to_dict(),
            "methods": methods,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentSeedResult:
        """Reconstruct from a canonical seed JSON dict."""
        metadata = SeedMetadata.from_dict(d["metadata"])
        raw_results: dict[str, MethodRawResult] = {}
        processed_results: dict[str, MethodProcessedResult] = {}
        for name, method_data in d.get("methods", {}).items():
            raw_results[name] = MethodRawResult.from_dict(name, method_data)
            fm = method_data.get("final_metrics")
            if fm:
                processed_results[name] = MethodProcessedResult.from_dict(name, fm)
        return cls(
            metadata=metadata,
            raw_results=raw_results,
            processed_results=processed_results,
        )
