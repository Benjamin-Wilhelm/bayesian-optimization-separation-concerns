"""Raw experimental data dataclasses for Experiment B (prior-guided BO).

One JSON file per (experiment, benchmark, seed), mirroring
:mod:`data.raw` (Experiment A) but extended with prior-specific fields.
:class:`~data.raw.InitialDesign` and :class:`~data.raw.ModelSnapshot` are
reused verbatim; Experiment A schemas are not modified.

Canonical JSON structure::

    {
      "metadata":        {PriorSeedMetadata fields},
      "initial_design":  {InitialDesign fields},
      "methods": {
        "<method_name>": {
          "train_label":     "y_obs",
          "model_kind":      "standard_gp" | "prior_mean_gp",
          "learn_lambda":    bool | null,
          "trajectory":      [PriorStepRecord, ...],
          "model_snapshots": [ModelSnapshot, ...]
        }
      }
    }

Additional raw information stored relative to Experiment A:

- the prior configuration (center, lengthscale, label) in the metadata;
- the orientation parameters (raw f_min, f_max, direction, and the derived
  oriented range / f_opt) in the metadata;
- ``prior_value_at_candidate`` per BO step;
- ``lambda_value`` per BO step (the fitted ``lambda`` of the prior-mean GP
  used to select that candidate; ``None`` for standard-GP methods) — the
  lambda trajectory of the adaptive method;
- GP hyperparameters (including ``lambda``) inside each
  :class:`~data.raw.ModelSnapshot`, as in Experiment A.

Design notes carried over from Experiment A:
- ``train_X`` / ``train_Y`` are never stored; ``process.py`` reconstructs
  them from ``InitialDesign + trajectory[:t]`` (all methods train on raw
  ``y_obs``).
- Each ``ModelSnapshot`` stores a serialized ``state_dict`` so the GP is
  rebuilt by ``load_state_dict`` — never refit.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from data.raw import InitialDesign, ModelSnapshot


# ---------------------------------------------------------------------------
# PriorStepRecord  — one BO iteration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorStepRecord:
    """One BO iteration recorded by Experiment B's ``run.py``.

    Attributes:
        iteration: 1-based index within the BO budget.
        x: Query point as ``list[float]`` (length ``dim``).
        y_obs: Noisy observation of the oriented objective (raw scale).
        f_noiseless: True oriented benchmark value ``g(x)`` (raw scale);
            ``None`` for real data.
        acq_value: Acquisition value at the selected candidate.
        train_size: GP training-set size before this query.
        posterior_mean_at_candidate: GP posterior mean at ``x`` before the
            candidate was added to the training set.
        posterior_std_at_candidate: GP posterior std at ``x``.
        prior_value_at_candidate: ``pi(x)`` at the selected candidate
            (native-domain prior).
        lambda_value: Fitted ``lambda`` of the prior-mean GP used at this
            iteration; ``None`` for methods with a standard GP.
    """

    iteration: int
    x: list[float]
    y_obs: float
    f_noiseless: Optional[float]
    acq_value: float
    train_size: int
    posterior_mean_at_candidate: float
    posterior_std_at_candidate: float
    prior_value_at_candidate: float
    lambda_value: Optional[float]

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
        prior_value: float,
        lambda_value: Optional[float],
    ) -> "PriorStepRecord":
        """Construct from raw tensors; converts everything to plain Python."""
        f: Optional[float] = f_val.squeeze().item() if f_val is not None else None
        return cls(
            iteration=iteration,
            x=x_cand.squeeze(0).tolist(),
            y_obs=y_obs_val.squeeze().item(),
            f_noiseless=f,
            acq_value=acq_value.item(),
            train_size=train_size,
            posterior_mean_at_candidate=posterior_mean,
            posterior_std_at_candidate=posterior_std,
            prior_value_at_candidate=float(prior_value),
            lambda_value=(
                float(lambda_value) if lambda_value is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorStepRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# PriorMethodTrajectory  — one method under one seed
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorMethodTrajectory:
    """Full BO-phase data for one Experiment B method under one seed.

    Attributes:
        method_name: Identifier used as the JSON key, e.g. ``"vanilla_ucb"``.
        train_label: Always ``"y_obs"`` — every Experiment B method trains
            the GP on raw noisy observations of ``f``.
        model_kind: ``"standard_gp"`` (vanilla / alpha-pi) or
            ``"prior_mean_gp"`` (lambda methods).  Tells ``process.py`` how
            to rebuild the GP from snapshots.
        learn_lambda: For ``"prior_mean_gp"`` only — whether ``lambda`` was
            trained by MLL (adaptive) or frozen at 1 (fixed).  ``None`` for
            standard GPs.
        trajectory: Ordered :class:`PriorStepRecord` list (BO iterations).
        model_snapshots: One :class:`~data.raw.ModelSnapshot` per iteration,
            captured before the candidate was added to the training set.
    """

    method_name: str
    train_label: str
    model_kind: str
    learn_lambda: Optional[bool]
    trajectory: list[PriorStepRecord]
    model_snapshots: list[ModelSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_label":     self.train_label,
            "model_kind":      self.model_kind,
            "learn_lambda":    self.learn_lambda,
            "trajectory":      [s.to_dict() for s in self.trajectory],
            "model_snapshots": [m.to_dict() for m in self.model_snapshots],
        }

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "PriorMethodTrajectory":
        return cls(
            method_name=name,
            train_label=d["train_label"],
            model_kind=d["model_kind"],
            learn_lambda=d.get("learn_lambda"),
            trajectory=[PriorStepRecord.from_dict(s) for s in d["trajectory"]],
            model_snapshots=[ModelSnapshot.from_dict(m) for m in d["model_snapshots"]],
        )


# ---------------------------------------------------------------------------
# PriorSeedMetadata
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PriorSeedMetadata:
    """Metadata for one (experiment, benchmark, seed) execution of Exp B.

    Attributes:
        experiment: ``"exp_b_prior_guided"``.
        problem_type: ``"synthetic"`` or ``"real_data"``.
        benchmark: Oriented problem name (same as the raw benchmark name,
            e.g. ``"forrester_1d"`` — no ``_normalized`` suffix).
        base_benchmark: Wrapped benchmark name, e.g. ``"forrester_1d"``.
        seed: Random seed for the Sobol design and BO trajectory.
        budget: Number of BO iterations (excluding the initial design).
        initial_design_size: Number of Sobol initial design points.
        noise_std: Observation noise std on the raw oriented scale.
        dim: Input dimension.
        bounds: Box bounds, ``[[lb_1, ..., lb_d], [ub_1, ..., ub_d]]``.
        orientation: ``{"f_min", "f_max", "direction", "f_lower", "f_upper",
            "f_opt"}`` actually used (raw inner extrema + derived oriented
            range), so reconstruction is independent of later default changes.
        prior: Prior configuration ``{"center", "lengthscale"}`` plus an
            optional descriptive ``"label"`` (e.g. ``"good"`` / ``"bad"``).
        f_opt: Oracle maximum of the oriented objective ``g`` on the raw
            scale (``-min f_raw`` for a minimization benchmark).
        alpha: Prior-strength weight used by ``alpha_pi_ucb``.
        beta: UCB exploration parameter shared by all methods.
    """

    experiment: str
    problem_type: str
    benchmark: str
    base_benchmark: str
    seed: int
    budget: int
    initial_design_size: int
    noise_std: float
    dim: int
    bounds: Optional[list[list[float]]] = None
    orientation: Optional[dict[str, Any]] = None
    prior: Optional[dict[str, Any]] = None
    f_opt: float = 0.0
    alpha: Optional[float] = None
    beta: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorSeedMetadata":
        # Filter unknown keys so the loader tolerates schema evolution.
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# RawPriorSeedData  — top-level
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RawPriorSeedData:
    """All raw Experiment B data for one (experiment, benchmark, seed).

    Attributes:
        metadata: Run-level metadata.
        initial_design: Sobol initial design shared by every method
            (reused :class:`~data.raw.InitialDesign`).
        methods: Per-method trajectories, keyed by method name.
    """

    metadata:        PriorSeedMetadata
    initial_design:  InitialDesign
    methods:         dict[str, PriorMethodTrajectory]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata":       self.metadata.to_dict(),
            "initial_design": self.initial_design.to_dict(),
            "methods":        {name: m.to_dict() for name, m in self.methods.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawPriorSeedData":
        return cls(
            metadata=PriorSeedMetadata.from_dict(d["metadata"]),
            initial_design=InitialDesign.from_dict(d["initial_design"]),
            methods={
                name: PriorMethodTrajectory.from_dict(name, md)
                for name, md in d.get("methods", {}).items()
            },
        )
