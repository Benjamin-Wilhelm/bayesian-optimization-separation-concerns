"""Exact 2-D hypervolume, dominance, and the hypervolume--scalarization identity.

All computations use the **maximization** convention: a point ``a`` dominates
``b`` when ``a >= b`` componentwise with at least one strict inequality, and a
point contributes to the hypervolume only where it lies above the reference
point ``z``.  For Experiment C's two objectives the hypervolume is computed
**exactly** (never Monte-Carlo) via a rectangle sweep.

The optional :func:`hypervolume_scalarization_estimate` implements the radial
identity

.. math::

    \mathrm{HV}_z(Y) = c_m \,
        \mathbb{E}_{\lambda \sim S_+^{m-1}}
        \big[ \max_{y \in Y} \rho^{HV}_\lambda(y - z) \big],
    \qquad c_m = \frac{\pi^{m/2}}{2^m \Gamma(m/2 + 1)},

with ``c_2 = pi / 4`` for two objectives.  It is a **diagnostic only** (tests
and convention checks) and is never used for the main 2-D curves.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from common.scalarizations import scalarize_hypervolume

Tensor = torch.Tensor


def hypervolume_ball_constant(m: int) -> float:
    """Return ``c_m = pi^{m/2} / (2^m * Gamma(m/2 + 1))`` (``c_2 = pi/4``)."""
    return math.pi ** (m / 2.0) / (2.0 ** m * math.gamma(m / 2.0 + 1.0))


def non_dominated_mask(Y: Tensor) -> Tensor:
    """Boolean mask of non-dominated rows of ``Y`` (maximization).

    Args:
        Y: Objective vectors, shape ``(n, m)``.

    Returns:
        Bool tensor of shape ``(n,)``; ``True`` where the row is not strictly
        dominated by any other row.
    """
    Y = Y.to(torch.double)
    n = Y.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    # ge[i, j] : Y[j] >= Y[i] componentwise (all objectives)
    ge_all = (Y.unsqueeze(0) >= Y.unsqueeze(1)).all(dim=-1)   # (i, j)
    gt_any = (Y.unsqueeze(0) > Y.unsqueeze(1)).any(dim=-1)    # (i, j)
    dominated_by = ge_all & gt_any                            # j dominates i
    dominated = dominated_by.any(dim=1)
    return ~dominated


def filter_valid(Y: Tensor, reference_point: Tensor) -> Tensor:
    """Keep finite points that strictly dominate ``z`` in every objective."""
    Y = Y.to(torch.double)
    z = torch.as_tensor(reference_point, dtype=torch.double).reshape(-1)
    finite = torch.isfinite(Y).all(dim=-1)
    above = (Y > z).all(dim=-1)
    return Y[finite & above]


def exact_hypervolume_2d(Y: Tensor, reference_point: Tensor) -> float:
    """Exact hypervolume of ``Y`` w.r.t. ``z`` in 2-D (maximization).

    Dominated points and duplicates do not change the result.  Points not
    strictly above ``z`` are ignored.

    Args:
        Y: Objective vectors, shape ``(n, 2)``.
        reference_point: Oriented reference ``z``, shape ``(2,)``.

    Returns:
        Non-negative float hypervolume.
    """
    Y = Y.to(torch.double)
    if Y.ndim != 2 or Y.shape[-1] != 2:
        raise ValueError(f"exact_hypervolume_2d expects (n, 2), got {tuple(Y.shape)}.")
    z = torch.as_tensor(reference_point, dtype=torch.double).reshape(-1)
    pts = filter_valid(Y, z)
    if pts.shape[0] == 0:
        return 0.0
    pts = pts[non_dominated_mask(pts)]
    # Sort by first objective descending; among non-dominated 2-D points the
    # second objective is then increasing.
    order = torch.argsort(pts[:, 0], descending=True)
    pts = pts[order]
    hv = 0.0
    prev_y1 = float(z[1].item())
    for i in range(pts.shape[0]):
        y0 = float(pts[i, 0].item())
        y1 = float(pts[i, 1].item())
        width = y0 - float(z[0].item())
        height = y1 - prev_y1
        if height > 0.0 and width > 0.0:
            hv += width * height
        if y1 > prev_y1:
            prev_y1 = y1
    return hv


def running_hypervolume_2d(
    Y: Tensor,
    reference_point: Tensor,
) -> Tensor:
    """Hypervolume of every prefix ``Y[:t]`` (monotone non-decreasing).

    Args:
        Y: Objective vectors in evaluation order, shape ``(n, 2)``.
        reference_point: Oriented reference ``z``, shape ``(2,)``.

    Returns:
        Tensor of shape ``(n,)`` with ``HV_z(Y[:t+1])`` at each step.
    """
    Y = Y.to(torch.double)
    n = Y.shape[0]
    out = torch.empty(n, dtype=torch.double)
    for t in range(n):
        out[t] = exact_hypervolume_2d(Y[: t + 1], reference_point)
    # Enforce monotonicity against floating-point wobble.
    return torch.cummax(out, dim=0).values


def hypervolume_regret_2d(
    Y: Tensor,
    reference_point: Tensor,
    hv_star: float,
) -> Tensor:
    """Prefix hypervolume regret ``HV*(Y*) - HV_z(Y[:t])``, shape ``(n,)``."""
    running = running_hypervolume_2d(Y, reference_point)
    return (float(hv_star) - running).clamp_min(0.0)


def hypervolume_scalarization_estimate(
    Y: Tensor,
    reference_point: Tensor,
    lambdas: Tensor,
) -> float:
    """Monte-Carlo hypervolume via the radial identity (diagnostic only).

    ``HV ~= c_m * mean_lambda max_{y in Y} rho^HV_lambda(y - z)`` for ``lambda``
    drawn uniformly on the positive unit sphere.

    Args:
        Y: Objective vectors, shape ``(n, m)``.
        reference_point: Oriented reference ``z``, shape ``(m,)``.
        lambdas: Directions on ``S_+^{m-1}``, shape ``(M, m)``.

    Returns:
        Float hypervolume estimate.
    """
    Y = Y.to(torch.double)
    m = Y.shape[-1]
    c_m = hypervolume_ball_constant(m)
    # values: (M, n, m) via broadcasting lambdas (M, 1, m) with Y (1, n, m)
    scal = scalarize_hypervolume(
        Y.unsqueeze(0), lambdas.unsqueeze(1), reference_point
    )  # (M, n)
    best = scal.max(dim=-1).values  # (M,)
    return float(c_m * best.mean().item())


__all__ = [
    "hypervolume_ball_constant",
    "non_dominated_mask",
    "filter_valid",
    "exact_hypervolume_2d",
    "running_hypervolume_2d",
    "hypervolume_regret_2d",
    "hypervolume_scalarization_estimate",
]
