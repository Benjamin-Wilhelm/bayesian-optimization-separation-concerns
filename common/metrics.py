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


# ---------------------------------------------------------------------------
# Target-matching metrics
# ---------------------------------------------------------------------------

def target_distance(
    f_values: torch.Tensor,
    target: Scalar,
) -> torch.Tensor:
    """Per-step absolute distance to the target ``|f(x_t) - T|``.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.  Use
            ``f_noiseless`` for synthetic problems.  For real-data problems
            pass ``y_obs`` or held-out validation values explicitly.
        target: Scalar target value ``T``.

    Returns:
        Tensor of shape ``(N,)`` with per-step target distances.
    """
    f_values = _to_1d_double(f_values, "f_values")
    return (f_values - _scalar(target)).abs()


def target_simple_regret(
    f_values: torch.Tensor,
    target: Scalar,
    best_achievable: Scalar = 0.0,
) -> torch.Tensor:
    """Final target-matching simple regret after ``N`` iterations.

    .. math::

        R_N^{\\mathrm{target}}
        = \\min_{1 \\le t \\le N} |f(x_t) - T|
          - \\min_{x \\in \\mathcal{X}} |f(x) - T|.

    When the target is attainable (``best_achievable = 0``), this simplifies
    to ``min_{1<=t<=N} |f(x_t) - T|``.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: ``min_{x} |f(x) - T|``.  Zero when the target lies
            in the range of ``f``; provide the true value for out-of-range
            targets.

    Returns:
        Scalar tensor with the target-matching simple regret.
    """
    distances = target_distance(f_values, target)
    return distances.min() - _scalar(best_achievable)


def running_target_simple_regret(
    f_values: torch.Tensor,
    target: Scalar,
    best_achievable: Scalar = 0.0,
) -> torch.Tensor:
    """Target-matching simple regret at each step ``t``.

    .. math::

        R_t^{\\mathrm{target}}
        = \\min_{1 \\le s \\le t} |f(x_s) - T|
          - \\min_{x \\in \\mathcal{X}} |f(x) - T|.

    Useful for convergence plots.

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: ``min_{x} |f(x) - T|``.

    Returns:
        Tensor of shape ``(N,)`` — running target simple regret.
    """
    distances = target_distance(f_values, target)
    running_min = torch.cummin(distances, dim=0).values
    return running_min - _scalar(best_achievable)


def target_cumulative_regret(
    f_values: torch.Tensor,
    target: Scalar,
    best_achievable: Scalar = 0.0,
) -> torch.Tensor:
    """Running sum of target-matching instantaneous regrets.

    .. math::

        C_t^{\\mathrm{target}}
        = \\sum_{s=1}^{t}
          \\bigl(|f(x_s) - T| - \\min_{x} |f(x) - T|\\bigr).

    Args:
        f_values: Noiseless trajectory values, shape ``(N,)``.
        target: Target value ``T``.
        best_achievable: ``min_{x} |f(x) - T|``.

    Returns:
        Tensor of shape ``(N,)`` — cumulative target regret after each step.
    """
    distances = target_distance(f_values, target)
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
    distances: torch.Tensor,
    tolerance: float,
    ) -> int:
    """First iteration index at which ``distances[t] <= tolerance``.

    Counts from 1 (first BO step) to N (last step).  Returns ``N + 1`` if
    the tolerance is never reached within the trajectory, signalling failure.

    This function is agnostic to what ``distances`` represents.  For
    target matching, pass ``target_distance(f_values, target)``.  For
    standard threshold-based hitting time, pass
    ``running_simple_regret(f_values, f_opt)`` with the desired tolerance.

    Args:
        distances: Non-negative 1-D tensor of shape ``(N,)``.
        tolerance: Non-negative threshold.

    Returns:
        1-based iteration index of the first hit, or ``N + 1`` if never hit.
    """
    distances = _to_1d_double(distances, "distances")
    hit = (distances <= tolerance).nonzero(as_tuple=False)
    if hit.numel() == 0:
        return len(distances) + 1
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

def failure_rate(
    metric_values: torch.Tensor,
    threshold: float,
    ) -> torch.Tensor:
    """Fraction of runs where a metric **exceeds** a threshold.

    Useful for computing the probability that simple regret or target
    distance is above a tolerable level after a fixed budget.

    Args:
        metric_values: Per-seed scalar metric, shape ``(S,)``.
        threshold: Failure criterion.  A run is counted as failed when
            its metric value is strictly greater than ``threshold``.

    Returns:
        Scalar tensor in ``[0, 1]`` — the failure rate.
    """
    metric_values = _to_1d_double(metric_values, "metric_values")
    return (metric_values > threshold).double().mean()


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
