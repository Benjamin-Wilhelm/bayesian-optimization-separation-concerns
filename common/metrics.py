"""Reusable evaluation metrics for BO experiments.

All functions are **pure and stateless** — they accept tensors or scalars
and return tensors.  No model fitting, acquisition logic, or I/O lives here.

Scientific convention
---------------------
- Synthetic metrics should receive ``f_noiseless`` (the true benchmark value).
  ``y_obs`` is for model training; it is a biased proxy for evaluation.
- Real-data callers may pass ``y_obs`` or externally provided validation
  values in place of ``f_noiseless``; the caller is responsible for being
  explicit about which quantity is passed.

Trajectory convention
---------------------
- A single BO run produces a 1-D float tensor of shape ``(N,)`` where
  ``values[t]`` is the evaluation at step ``t`` (0-indexed).
- Multi-seed aggregation functions accept shape ``(S,)`` where ``S`` is
  the number of seeds.

Minimisation convention
-----------------------
Standard BO metrics (simple regret, cumulative regret) assume the goal is
to **minimise** ``f``.  For maximisation problems, negate ``f_values``
and ``f_opt`` before calling.

Target-matching metrics are sign-agnostic because they use ``|f - T|``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

# Scalar-like type accepted as ``f_opt``, ``target``, etc.
Scalar = Union[float, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_1d_double(values: torch.Tensor, name: str = "values") -> torch.Tensor:
    """Validate and cast a trajectory tensor to shape ``(N,)`` double."""
    values = values.to(torch.double)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (N,) or 2-D (N, 1), got shape {tuple(values.shape)}"
        )
    return values


def _scalar(value: Scalar) -> torch.Tensor:
    """Convert a float or scalar tensor to a 0-D double tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(torch.double).squeeze()
    return torch.tensor(value, dtype=torch.double)


def _coerce_to_1d_double(
    values: Union[torch.Tensor, "np.ndarray", list],
    name: str = "values",
) -> torch.Tensor:
    """Accept a tensor, numpy array, or list; return a 1-D double torch.Tensor.

    This is a superset of ``_to_1d_double`` that additionally converts
    numpy arrays and Python lists via ``torch.as_tensor``.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.double)
    return _to_1d_double(values, name)


# ---------------------------------------------------------------------------
# Standard BO metrics  (minimisation convention)
# ---------------------------------------------------------------------------

def instantaneous_regret(
    f_values: torch.Tensor,
    f_opt: Scalar,
    ) -> torch.Tensor:
    """Per-step regret ``f(x_t) - f_opt`` for each iteration.

    Args:
        f_values: Noiseless function values along the trajectory, shape
            ``(N,)``.  Must use ``f_noiseless`` for synthetic problems.
        f_opt: Global minimum value ``f(x*)``.

    Returns:
        Tensor of shape ``(N,)`` with per-step regret values.  Values are
        non-negative when the algorithm has not found the optimum.
    """
    f_values = _to_1d_double(f_values, "f_values")
    return f_values - _scalar(f_opt)


def simple_regret(
    f_values: torch.Tensor,
    f_opt: Scalar,
    ) -> torch.Tensor:
    """Final simple regret after ``N`` iterations.

    R_N = min_{1 <= t <= N} f(x_t) - f*

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        f_opt: Global minimum ``f^* = min_{x} f(x)``.

    Returns:
        Scalar tensor with the simple regret.
    """
    f_values = _to_1d_double(f_values, "f_values")
    return f_values.min() - _scalar(f_opt)


def running_simple_regret(
    f_values: torch.Tensor,
    f_opt: Scalar,
    ) -> torch.Tensor:
    """Simple regret at each step ``t``, i.e. ``min_{1<=s<=t} f(x_s) - f*``.

    Useful for plotting convergence curves.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        f_opt: Global minimum ``f^*``.

    Returns:
        Tensor of shape ``(N,)`` — the simple regret after each step.
    """
    f_values = _to_1d_double(f_values, "f_values")
    running_min = torch.cummin(f_values, dim=0).values
    return running_min - _scalar(f_opt)


def cumulative_regret(
    f_values: torch.Tensor,
    f_opt: Scalar,
) -> torch.Tensor:
    """Running sum of instantaneous regrets.

    C_t = \sum_{s=1}^t (f(x_s) - f*), for t = 1,...,N

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        f_opt: Global minimum ``f^*``.

    Returns:
        Tensor of shape ``(N,)`` — cumulative regret after each step.
    """
    ir = instantaneous_regret(f_values, f_opt)
    return ir.cumsum(dim=0)


def average_cumulative_regret(
    f_values: torch.Tensor,
    f_opt: Scalar,
    ) -> torch.Tensor:
    """Running average of instantaneous regrets ``C_t / t``.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        f_opt: Global minimum ``f^*``.

    Returns:
        Tensor of shape ``(N,)`` — average cumulative regret after each step.
    """
    cr = cumulative_regret(f_values, f_opt)
    steps = torch.arange(1, len(cr) + 1, dtype=torch.double)
    return cr / steps


def best_so_far(
    values: Union[torch.Tensor, list],
    minimize: bool = True,
) -> torch.Tensor:
    """Running best value seen up to each step.

    For minimisation (``minimize=True``) returns ``min_{s<=t} values[s]``
    at each ``t``.  For maximisation returns the running maximum.

    Args:
        values: 1-D sequence of values, shape ``(N,)``.  Accepts tensors,
            numpy arrays, or plain Python lists.
        minimize: If ``True`` return the running minimum; otherwise the
            running maximum.

    Returns:
        Tensor of shape ``(N,)`` — running best at each step.
    """
    v = _coerce_to_1d_double(values, "values")
    if minimize:
        return torch.cummin(v, dim=0).values
    return torch.cummax(v, dim=0).values


# ---------------------------------------------------------------------------
# Target-matching metrics
# ---------------------------------------------------------------------------

def target_distance(
    f_values: Union[torch.Tensor, list],
    target: Scalar,
) -> torch.Tensor:
    """Per-step absolute distance to the target ``|f(x_t) - T|``.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.  Use
            ``f_noiseless`` for synthetic problems.  For real-data problems
            pass ``y_obs`` or held-out validation values explicitly.
            Accepts tensors, numpy arrays, or plain Python lists.
        target: Scalar target value ``T``.

    Returns:
        Tensor of shape ``(N,)`` with per-step target distances.
    """
    f_values = _coerce_to_1d_double(f_values, "f_values")
    return (f_values - _scalar(target)).abs()


def target_best_so_far(
    f_values: Union[torch.Tensor, list],
    target: Scalar,
) -> torch.Tensor:
    """Running best target distance ``min_{s<=t} |f(x_s) - T|`` at each step.

    Unlike ``running_target_simple_regret``, this returns the absolute
    running-best distance without subtracting an oracle lower bound.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.  Accepts
            tensors, numpy arrays, or plain Python lists.
        target: Scalar target value ``T``.

    Returns:
        Tensor of shape ``(N,)`` — non-increasing sequence of running-best
        target distances.
    """
    distances = target_distance(f_values, target)
    return torch.cummin(distances, dim=0).values


def target_simple_regret(
    f_values: torch.Tensor,
    target: Scalar,
    best_achievable: Scalar = 0.0,
) -> torch.Tensor:
    """Compute the final target-matching simple regret.

    The regret is defined as

        min_{1 <= t <= N} |f(x_t) - T| - min_{x in X} |f(x) - T|.

    It measures how close the best sampled point is to the target level, relative
    to the best achievable target distance on the domain. If the target is
    attainable, ``best_achievable = 0`` and the regret is simply the best target
    distance found during the trajectory.

    Args:
        f_values: Noiseless function values along the trajectory, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: Best possible target distance on the domain,
            ``min_x |f(x) - T|``. Defaults to zero when the target is attainable.

    Returns:
        Scalar target-matching simple regret.
    """
    distances = target_distance(f_values, target)
    return distances.min() - _scalar(best_achievable)


def running_target_simple_regret(
    f_values: torch.Tensor,
    target: Scalar,
    best_achievable: Scalar = 0.0,
) -> torch.Tensor:
    """Compute the running target-matching simple regret.

    At each iteration ``t``, the regret is defined as

        min_{1 <= s <= t} |f(x_s) - T| - min_{x in X} |f(x) - T|.

    This metric measures how quickly the optimization trajectory approaches the
    target level. It is mainly used for convergence curves and trajectory analysis.

    Args:
        f_values: Noiseless function values along the trajectory, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: Best possible target distance on the domain,
            ``min_x |f(x) - T|``.

    Returns:
        Tensor of shape ``(N,)`` containing the running target-matching simple
        regret at each iteration.
    """
    distances = target_distance(f_values, target)
    running_min = torch.cummin(distances, dim=0).values
    return running_min - _scalar(best_achievable)


def target_cumulative_regret(
    f_values: Union[torch.Tensor, list],
    target: Scalar,
    best_achievable: Scalar = 0.0,
    cumulative_best: bool = False,
) -> torch.Tensor:
    """Compute the cumulative target-matching regret.

    Two accumulation modes are available.

    If ``cumulative_best=False``, the function accumulates instantaneous target
    regrets:

        sum_{s=1}^t (|f(x_s) - T| - min_x |f(x) - T|).

    If ``cumulative_best=True``, it accumulates best-so-far target regrets:

        sum_{s=1}^t (min_{r <= s} |f(x_r) - T| - min_x |f(x) - T|).

    The instantaneous version measures the quality of all sampled points, while
    the best-so-far version measures the cumulative progress of the recommendation
    trajectory.

    Args:
        f_values: Noiseless function values along the trajectory, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: Best possible target distance on the domain,
            ``min_x |f(x) - T|``. Defaults to zero when the target is attainable.
        cumulative_best: If ``True``, accumulate running best-so-far regrets
            instead of instantaneous regrets.

    Returns:
        Tensor of shape ``(N,)`` containing the cumulative target-matching regret
        after each iteration.
    """
    distances = target_distance(f_values, target)
    if cumulative_best:
        distances = torch.cummin(distances, dim=0).values
    ir = distances - _scalar(best_achievable)
    return ir.cumsum(dim=0)


def target_average_cumulative_regret(
    f_values: torch.Tensor,
    target: Scalar,
    best_achievable: Scalar = 0.0,
) -> torch.Tensor:
    """Average cumulative target regret ``C_t^{target} / t``.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: ``min_{x} |f(x) - T|``.

    Returns:
        Tensor of shape ``(N,)`` — average cumulative target regret.
    """
    cr = target_cumulative_regret(f_values, target, best_achievable)
    steps = torch.arange(1, len(cr) + 1, dtype=torch.double)
    return cr / steps


# ---------------------------------------------------------------------------
# Hitting time
# ---------------------------------------------------------------------------

def hitting_time(
    distances: Union[torch.Tensor, list],
    tolerance: float,
    not_hit_value: Optional[int] = None,
) -> int:
    """First iteration index at which ``distances[t] <= tolerance``.

    Counts from 1 (first BO step) to N (last step).  If the tolerance is
    never reached, returns ``N + 1`` by default, or ``not_hit_value`` if
    that argument is provided.

    This function is agnostic to what ``distances`` represents.  For
    target matching pass ``target_best_so_far(f_values, target)`` (the
    running-best distances) so that the result is the first iteration at
    which the algorithm found a solution within ``tolerance`` of the target.
    Passing instantaneous ``target_distance(f_values, target)`` is also
    valid when per-step threshold arrival is of interest.

    Args:
        distances: Non-negative 1-D tensor of shape ``(N,)``.  Accepts
            tensors, numpy arrays, or plain Python lists.
        tolerance: Non-negative threshold (``epsilon`` in the literature).
        not_hit_value: Value to return when the threshold is never met.
            If ``None`` (default), returns ``len(distances) + 1``.

    Returns:
        1-based iteration index of the first hit, or the not-hit sentinel.
    """
    distances = _coerce_to_1d_double(distances, "distances")
    hit = (distances <= tolerance).nonzero(as_tuple=False)
    if hit.numel() == 0:
        return not_hit_value if not_hit_value is not None else len(distances) + 1
    return int(hit[0].item()) + 1  # convert 0-based index to 1-based


# ---------------------------------------------------------------------------
# Multi-objective metrics
# ---------------------------------------------------------------------------

def hypervolume_indicator(
    Y: torch.Tensor,
    reference_point: torch.Tensor,
) -> torch.Tensor:
    """Hypervolume indicator of a set of objective vectors.

    Uses BoTorch's ``DominatedPartitioning`` for exact computation.

    Assumes **maximisation** convention (BoTorch's default).  For
    minimisation problems, negate ``Y`` and ``reference_point`` before
    calling.

    Args:
        Y: Objective vectors, shape ``(n, m)``.  Each row is one solution.
        reference_point: Anti-ideal reference point, shape ``(m,)``.  Must
            be dominated by every point in the Pareto front for the
            hypervolume to be meaningful.

    Returns:
        Scalar tensor with the hypervolume value.
    """
    from botorch.utils.multi_objective.hypervolume import DominatedPartitioning  # local to avoid hard dep

    Y = Y.to(torch.double)
    reference_point = reference_point.to(torch.double)
    bd = DominatedPartitioning(ref_point=reference_point, Y=Y)
    return bd.compute_hypervolume()


def per_objective_best(
    Y: torch.Tensor,
) -> torch.Tensor:
    """Best value achieved for each objective independently.

    Assumes **maximisation** convention.  For minimisation, negate ``Y``
    before calling (and negate the result).

    This is *not* the Pareto front — it gives the theoretical upper bound
    if objectives could be maximised independently.

    Args:
        Y: Objective vectors, shape ``(n, m)``.

    Returns:
        Tensor of shape ``(m,)`` — column-wise maximum.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n, m), got shape {tuple(Y.shape)}")
    return Y.to(torch.double).max(dim=0).values


# ---------------------------------------------------------------------------
# Cross-run aggregation
# ---------------------------------------------------------------------------
#
# Failure rate has been removed here.  It is intrinsically a cross-seed
# summary and lives in :mod:`common.statistics` (``statistics.failure_rate``)
# alongside the other aggregate utilities.


def high_quantile(
    metric_values: torch.Tensor,
    q: float,
    ) -> torch.Tensor:
    """``q``-th quantile of a metric across runs.

    For example, ``high_quantile(regrets, 0.9)`` gives the 90th-percentile
    simple regret across seeds — a tail-risk summary robust to outlier runs.

    Args:
        metric_values: Per-seed scalar metric, shape ``(S,)``.
        q: Quantile in ``(0, 1]``.

    Returns:
        Scalar tensor — the ``q``-th quantile.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    metric_values = _to_1d_double(metric_values, "metric_values")
    return torch.quantile(metric_values, q)


# ---------------------------------------------------------------------------
# Per-trajectory summary
# ---------------------------------------------------------------------------

def final_run_summary(
    f_values: Union[torch.Tensor, list],
    target: Scalar,
    oracle_best: float = 0.0,
    epsilon: Optional[float] = None,
    failure_threshold: Optional[float] = None,
) -> dict:
    """Compute a comprehensive per-trajectory summary for target matching.

    All values in the returned dictionary are Python scalars so the dict
    can be serialised directly to JSON.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.  Use
            ``f_noiseless`` for synthetic problems.  Accepts tensors,
            numpy arrays, or plain Python lists.
        target: Target value ``T``.
        oracle_best: ``min_{x} |f(x) - T|``.  Defaults to ``0.0``, which is
            valid when the target lies in the range of ``f``.
        epsilon: If provided, the hitting-time threshold; ``hitting_time``
            is computed using the running-best target distances.
        failure_threshold: If provided, a run is flagged as failed when
            ``final_best_target_distance > failure_threshold``.

    Returns:
        Dictionary with the following keys (all JSON-serialisable):

        - ``final_best_target_distance``: ``min_{t} |f(x_t) - T|``
        - ``final_target_simple_regret``: above minus ``oracle_best``
        - ``cumulative_instantaneous_target_regret``: sum over steps of
          ``|f(x_t) - T| - oracle_best``
        - ``cumulative_best_so_far_target_regret``: sum over steps of
          ``min_{s<=t}|f(x_s)-T| - oracle_best``
        - ``worst_instantaneous_target_distance``: ``max_t |f(x_t) - T|``
        - ``mean_instantaneous_target_distance``: mean over steps
        - ``median_instantaneous_target_distance``: median over steps
        - ``hitting_time`` *(only if epsilon is provided)*
        - ``failure_indicator`` *(only if failure_threshold is provided)*:
          ``1`` if the run failed, ``0`` otherwise
    """
    f_t = _coerce_to_1d_double(f_values, "f_values")
    dist = target_distance(f_t, target)
    bsf = torch.cummin(dist, dim=0).values

    final_bsf = float(bsf[-1].item())
    cum_inst = target_cumulative_regret(
        f_t, target, best_achievable=oracle_best, cumulative_best=False
    )
    cum_bsf = target_cumulative_regret(
        f_t, target, best_achievable=oracle_best, cumulative_best=True
    )

    result: dict = {
        "final_best_target_distance": final_bsf,
        "final_target_simple_regret": final_bsf - float(oracle_best),
        "cumulative_instantaneous_target_regret": float(cum_inst[-1].item()),
        "cumulative_best_so_far_target_regret": float(cum_bsf[-1].item()),
        "worst_instantaneous_target_distance": float(dist.max().item()),
        "mean_instantaneous_target_distance": float(dist.mean().item()),
        "median_instantaneous_target_distance": float(dist.median().item()),
    }

    if epsilon is not None:
        result["hitting_time"] = hitting_time(bsf, epsilon)

    if failure_threshold is not None:
        result["failure_indicator"] = int(final_bsf > failure_threshold)

    return result


# ---------------------------------------------------------------------------
# Optional convergence-curve utilities
# ---------------------------------------------------------------------------

def area_under_curve(
    values: Union[torch.Tensor, list],
) -> float:
    """Area under a convergence curve using the trapezoidal rule.

    Equivalent to the discrete integral with unit step spacing:
    ``0.5 * (v[0] + 2*v[1] + ... + 2*v[N-2] + v[N-1])``.

    Useful as a scalar summary of a full convergence trajectory.

    Args:
        values: 1-D sequence of non-negative metric values, shape ``(N,)``.

    Returns:
        Scalar float — area under the curve.
    """
    v = _coerce_to_1d_double(values, "values")
    if len(v) == 0:
        return 0.0
    if len(v) == 1:
        return float(v[0].item())
    return float((0.5 * (v[0] + 2.0 * v[1:-1].sum() + v[-1])).item())


def time_to_fraction_of_range(
    f_values: Union[torch.Tensor, list],
    target: Scalar,
    f_range: float,
    fraction: float = 0.05,
) -> int:
    """First iteration where running-best target distance ≤ ``fraction * f_range``.

    A normalised hitting-time metric useful for comparing benchmarks with
    different function value scales.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        target: Target value ``T``.
        f_range: Scale of the function, e.g. ``f_max - f_min`` over the domain.
            Must be positive.
        fraction: Fraction of the range to use as the threshold.  Defaults to
            ``0.05`` (5% of the function range).

    Returns:
        1-based iteration index of the first hit, or ``N + 1`` if never hit.
    """
    if f_range <= 0:
        raise ValueError(f"f_range must be positive, got {f_range}")
    bsf = target_best_so_far(f_values, target)
    return hitting_time(bsf, fraction * float(f_range))



