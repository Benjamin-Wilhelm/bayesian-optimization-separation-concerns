"""Initial design generation for Bayesian optimization.

Provides quasi-random initial candidate sets for seeding the BO loop.
The main entry point is :func:`sobol_design`.

Usage::

    X_init = sobol_design(problem.bounds, n=10, seed=seed)
    obs = problem.observe(X_init)
    # train only on obs.y_obs
"""

from __future__ import annotations

import torch


def sobol_design(
    bounds: torch.Tensor,
    n: int,
    seed: int = 0,
    ) -> torch.Tensor:
    """Generate n quasi-random initial design points via a Sobol sequence.

    Points are drawn in the unit hypercube and linearly mapped to ``bounds``.

    Args:
        bounds: Box constraints, shape (2, dim).  bounds[0] is the lower bound, bounds[1] is the upper bound.
        n: Number of initial design points.
        seed: Scramble seed for the Sobol engine; fully determines the design.

    Returns:
        Tensor of shape (n, dim) with dtype torch.double.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(
            f"bounds must have shape (2, dim), got {tuple(bounds.shape)}"
        )
    dim = bounds.shape[1]
    sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_unit = sobol.draw(n).to(torch.double)
    lb = bounds[0].to(torch.double)
    ub = bounds[1].to(torch.double)
    return lb + (ub - lb) * X_unit
