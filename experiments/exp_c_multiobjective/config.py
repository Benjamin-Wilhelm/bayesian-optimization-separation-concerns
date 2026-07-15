"""Configuration parsing and early validation for Experiment C.

Configs are plain dicts loaded from YAML (as in Experiments A and B).  This
module centralises the *early* validation demanded by the scientific
protocol, so incompatibilities raise a clear error before any run starts
rather than producing silently wrong results.

Two experiment modes, with disjoint method sets:

- ``fixed_scalarization`` methods: ``right``, ``wrong`` (fixed lambda);
- ``pareto_front`` methods: ``right_algorithm_1``, ``right_algorithm_2``,
  ``wrong_algorithm_2``.

The critical budget invariant for Algorithm 2 is enforced here:
``sum_ell T_ell == N`` and every local budget must fit its per-subrun initial
design.
"""

from __future__ import annotations

from typing import Any, Mapping

from common.scalarizations import SCALARIZATION_KINDS

FIXED_METHODS: tuple[str, ...] = ("right", "wrong")
PARETO_METHODS: tuple[str, ...] = (
    "right_algorithm_1",
    "right_algorithm_2",
    "wrong_algorithm_2",
)
EXPERIMENT_MODES: tuple[str, ...] = ("fixed_scalarization", "pareto_front")


def budget_split(total: int, n_subruns: int) -> list[int]:
    """Split a global budget ``N`` across ``L`` subruns as evenly as possible.

    The first ``N mod L`` subruns receive one extra evaluation::

        base = N // L
        budgets[i] = base + (1 if i < N % L else 0)

    Guarantees ``sum(budgets) == total``.

    Raises:
        ValueError: If ``total`` or ``n_subruns`` is non-positive, or if
            ``total < n_subruns`` (some subrun would get a zero budget).
    """
    if total <= 0:
        raise ValueError(f"budget must be positive, got {total}.")
    if n_subruns <= 0:
        raise ValueError(f"n_subruns must be positive, got {n_subruns}.")
    if total < n_subruns:
        raise ValueError(
            f"global budget {total} < number of subruns {n_subruns}: at least "
            "one subrun would receive no evaluation. Reduce n_scalarizations "
            "or raise the budget."
        )
    base = total // n_subruns
    remainder = total % n_subruns
    budgets = [base + (1 if i < remainder else 0) for i in range(n_subruns)]
    assert sum(budgets) == total
    return budgets


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_common(config: Mapping[str, Any]) -> None:
    mode = config.get("experiment_mode")
    _require(mode in EXPERIMENT_MODES,
             f"experiment_mode must be one of {EXPERIMENT_MODES}, got {mode!r}.")
    _require(int(config.get("budget", 0)) > 0, "budget must be a positive integer.")
    seeds = config.get("seeds")
    _require(isinstance(seeds, Mapping) and "start" in seeds and "end" in seeds,
             "seeds must be a mapping with integer 'start' and 'end'.")
    _require(int(seeds["start"]) <= int(seeds["end"]),
             "seeds.start must be <= seeds.end.")
    noise = config.get("noise", {}) or {}
    _require(float(noise.get("std", 0.0)) >= 0.0, "noise.std must be >= 0.")
    acq = config.get("acquisition", {}) or {}
    _require(int(acq.get("raw_samples", 1)) > 0, "acquisition.raw_samples must be > 0.")
    _require(int(acq.get("num_restarts", 1)) > 0, "acquisition.num_restarts must be > 0.")
    ref = config.get("reference", {}) or {}
    _require(int(ref.get("pareto_front_resolution", 10000)) >= 2,
             "reference.pareto_front_resolution must be >= 2.")


def validate_fixed_scalarization(config: Mapping[str, Any]) -> None:
    validate_common(config)
    methods = list(config.get("methods", []))
    _require(len(methods) > 0, "methods must be non-empty.")
    for meth in methods:
        _require(meth in FIXED_METHODS,
                 f"fixed_scalarization method '{meth}' unknown; "
                 f"allowed: {FIXED_METHODS}.")
    scals = list(config.get("scalarizations", []))
    _require(len(scals) > 0, "scalarizations must be non-empty.")
    for s in scals:
        _require(s in SCALARIZATION_KINDS,
                 f"unknown scalarization '{s}'; allowed: {SCALARIZATION_KINDS}.")
    lam = config.get("lambda", {}) or {}
    _require(lam.get("mode") == "fixed_angles",
             "fixed_scalarization requires lambda.mode == 'fixed_angles'.")
    angles = list(lam.get("angles_degrees", []))
    _require(len(angles) > 0, "lambda.angles_degrees must be non-empty.")
    for a in angles:
        _require(0.0 < float(a) < 90.0,
                 f"lambda angle {a} must be strictly in (0, 90) degrees.")
    n_init = int(config.get("initial_design_size", 0))
    budget = int(config["budget"])
    _require(0 < n_init < budget,
             f"initial_design_size must satisfy 0 < n_init < budget "
             f"(got n_init={n_init}, budget={budget}).")


def validate_pareto_front(config: Mapping[str, Any]) -> None:
    validate_common(config)
    methods = list(config.get("methods", []))
    _require(len(methods) > 0, "methods must be non-empty.")
    for meth in methods:
        _require(meth in PARETO_METHODS,
                 f"pareto_front method '{meth}' unknown; allowed: {PARETO_METHODS}.")
    scal = config.get("scalarization")
    _require(scal in SCALARIZATION_KINDS,
             f"pareto_front scalarization must be one of {SCALARIZATION_KINDS}, "
             f"got {scal!r}.")
    lam = config.get("lambda", {}) or {}
    _require(lam.get("distribution") == "uniform_positive_sphere",
             "pareto_front requires lambda.distribution == 'uniform_positive_sphere'.")
    n_scal = int(lam.get("n_scalarizations_algorithm_2", 0))
    _require(n_scal > 0, "lambda.n_scalarizations_algorithm_2 must be positive.")
    _require(int(lam.get("evaluation_samples", 0)) > 0,
             "lambda.evaluation_samples must be positive.")
    budget = int(config["budget"])
    alg1 = config.get("algorithm_1", {}) or {}
    n_init_1 = int(alg1.get("initial_design_size", 0))
    _require(0 < n_init_1 <= budget,
             "algorithm_1.initial_design_size must satisfy 0 < n_init <= budget.")
    alg2 = config.get("algorithm_2", {}) or {}
    n_init_sub = int(alg2.get("initial_design_size_per_subrun", 0))
    _require(n_init_sub > 0, "algorithm_2.initial_design_size_per_subrun must be > 0.")
    # Budget split must fit each subrun's initial design.
    local_budgets = budget_split(budget, n_scal)
    for i, tb in enumerate(local_budgets):
        _require(
            tb >= n_init_sub,
            f"Algorithm 2 subrun {i} local budget {tb} is smaller than the "
            f"per-subrun initial design {n_init_sub}. Reduce "
            "initial_design_size_per_subrun or n_scalarizations, or raise the "
            "budget.",
        )
    sched = str(alg2.get("schedule", "round_robin"))
    _require(sched == "round_robin",
             f"only 'round_robin' scheduling is supported, got '{sched}'.")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate a full config according to its ``experiment_mode``."""
    mode = config.get("experiment_mode")
    if mode == "fixed_scalarization":
        validate_fixed_scalarization(config)
    elif mode == "pareto_front":
        validate_pareto_front(config)
    else:
        raise ValueError(
            f"experiment_mode must be one of {EXPERIMENT_MODES}, got {mode!r}."
        )


__all__ = [
    "FIXED_METHODS",
    "PARETO_METHODS",
    "EXPERIMENT_MODES",
    "budget_split",
    "validate_config",
    "validate_fixed_scalarization",
    "validate_pareto_front",
]
