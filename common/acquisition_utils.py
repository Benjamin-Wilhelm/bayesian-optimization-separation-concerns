"""Generic acquisition function optimisation utilities.

Provides thin wrappers around BoTorch's ``optimize_acqf`` that work with
*any* :class:`~botorch.acquisition.AcquisitionFunction`.

No experiment-specific or task-specific logic lives here.  All objective
structure (targets, constraints, costs, priors) belongs in the
``acquisitions/`` directory.

Usage::

    from botorch.acquisition import UpperConfidenceBound
    from common.acquisition_utils import optimize_acquisition

    acq = UpperConfidenceBound(model, beta=2.0)
    candidates, acq_value = optimize_acquisition(acq, bounds, q=1)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_NUM_RESTARTS: int = 10
_DEFAULT_RAW_SAMPLES: int = 256


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def optimize_acquisition(
    acq_fn: AcquisitionFunction,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = _DEFAULT_NUM_RESTARTS,
    raw_samples: int = _DEFAULT_RAW_SAMPLES,
    fixed_features: Optional[dict[int, float]] = None,
    options: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise an acquisition function over a box-constrained domain.

    Wraps :func:`botorch.optim.optimize_acqf` with sensible defaults and
    returns detached tensors suitable for downstream use without keeping
    the computational graph alive.

    Args:
        acq_fn: Any BoTorch acquisition function.  Must be in ``eval`` mode;
            the model it wraps should already be fitted.
        bounds: Box constraints, shape ``(2, d)``.  ``bounds[0]`` are lower
            bounds and ``bounds[1]`` are upper bounds.
        q: Number of candidates to return (batch size).  Use ``q=1`` for
            sequential BO.
        num_restarts: Number of random restarts for the L-BFGS-B optimiser.
            Higher values reduce the chance of converging to a poor local
            optimum at the cost of more compute.
        raw_samples: Number of quasi-random samples used to initialise the
            multi-start optimisation.
        fixed_features: Optional dict mapping dimension indices to fixed
            values, forwarded to
            :func:`~botorch.optim.optimize_acqf_mixed` semantics.  Currently
            passed directly to ``optimize_acqf``; if ``None``, all dimensions
            are free.
        options: Optional dict of L-BFGS-B solver options, e.g.
            ``{"maxiter": 200}``.

    Returns:
        Tuple ``(candidates, acq_value)`` where:

        - ``candidates``: Detached tensor of shape ``(q, d)`` containing the
          optimised candidate points.
        - ``acq_value``: Detached scalar tensor, the acquisition value at the
          returned candidates.

    Raises:
        ValueError: If ``bounds`` does not have shape ``(2, d)`` or if ``q``
            is not a positive integer.

    Example::

        candidates, value = optimize_acquisition(
            acq_fn=ucb,
            bounds=problem.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
        next_x = candidates  # shape (1, d)
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(
            f"bounds must have shape (2, d), got {tuple(bounds.shape)}"
        )
    if q < 1:
        raise ValueError(f"q must be a positive integer, got {q}")

    bounds = bounds.to(torch.double)

    candidates, acq_value = optimize_acqf(
        acq_function=acq_fn,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features=fixed_features,
        options=options or {},
    )

    return candidates.detach(), acq_value.detach()


# ---------------------------------------------------------------------------
# Sequential convenience wrapper
# ---------------------------------------------------------------------------

def suggest_next_point(
    acq_fn: AcquisitionFunction,
    bounds: torch.Tensor,
    num_restarts: int = _DEFAULT_NUM_RESTARTS,
    raw_samples: int = _DEFAULT_RAW_SAMPLES,
    options: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the single best next candidate under ``acq_fn``.

    Convenience wrapper around :func:`optimize_acquisition` for the common
    sequential BO case where ``q=1``.

    Args:
        acq_fn: Any BoTorch acquisition function.
        bounds: Box constraints, shape ``(2, d)``.
        num_restarts: Number of random restarts.
        raw_samples: Number of raw samples for initialisation.
        options: Optional L-BFGS-B solver options.

    Returns:
        Tuple ``(x_next, acq_value)`` where:

        - ``x_next``: Detached tensor of shape ``(1, d)``.
        - ``acq_value``: Detached scalar tensor.
    """
    return optimize_acquisition(
        acq_fn=acq_fn,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )


# ---------------------------------------------------------------------------
# Numerical safeguard
# ---------------------------------------------------------------------------

def clamp_variance(
    variance: torch.Tensor,
    min_variance: float = 1e-6,
) -> torch.Tensor:
    """Clamp predictive variance to a positive lower bound.

    Posterior variance can become numerically negative due to floating-point
    cancellation in the Cholesky solve.  Clamping prevents downstream
    issues such as ``sqrt`` of a negative number in UCB-style acquisitions.

    Args:
        variance: Predictive variance tensor, any shape.
        min_variance: Minimum allowed value.  Defaults to ``1e-6``.

    Returns:
        Clamped variance tensor, same shape as input.
    """
    return variance.clamp(min=min_variance)
