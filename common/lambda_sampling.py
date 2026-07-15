"""Preference (lambda) sampling for multi-objective scalarizations.

Two utilities:

- :func:`sample_positive_sphere` draws directions uniformly on the positive
  orthant of the unit sphere ``S_+^{m-1} = {lambda in R_+^m : ||lambda||_2 = 1}``
  via the folded-Gaussian construction ``lambda = |g| / ||g||`` with
  ``g ~ N(0, I)``.  Drawing i.i.d. Uniform(0, 1) coordinates and normalizing
  would **not** be uniform on the sphere, so it is deliberately avoided.
- :func:`fixed_angle_grid_2d` builds a deterministic 2-D grid
  ``lambda = (cos theta, sin theta)`` from angles in degrees (default
  ``10, 30, 45, 60, 80``), used by the ``fixed_scalarization`` experiment.

Both return unit-L2 directions.  The same directions are reused across the
``linear`` and ``hypervolume`` scalarizations so the comparison is controlled.
An optional L1 view (``lambda / sum_j lambda_j``) is provided for interpreting
linear weights as proportions, but results always keep the original spherical
direction.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

Tensor = torch.Tensor

DEFAULT_FIXED_ANGLES_DEG: tuple[float, ...] = (10.0, 30.0, 45.0, 60.0, 80.0)

# Guard against a (measure-zero) zero-norm draw.
_NORM_EPS: float = 1e-12


def sample_positive_sphere(
    n: int,
    m: int,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Sample ``n`` directions uniformly on ``S_+^{m-1}``.

    Args:
        n: Number of directions.
        m: Objective-space dimension.
        generator: Optional :class:`torch.Generator` for reproducibility.
        dtype: Output dtype (default ``torch.double``).
        device: Output device.

    Returns:
        Tensor of shape ``(n, m)``, strictly non-negative, unit L2 norm per row.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}.")
    g = torch.randn((n, m), generator=generator, dtype=dtype, device=device)
    lambdas = g.abs()
    norm = lambdas.norm(dim=-1, keepdim=True)
    # Replace any degenerate zero-norm row with a uniform direction.
    degenerate = norm.squeeze(-1) <= _NORM_EPS
    if bool(degenerate.any()):
        fallback = torch.full((m,), 1.0 / math.sqrt(m), dtype=dtype, device=device)
        lambdas[degenerate] = fallback
        norm = lambdas.norm(dim=-1, keepdim=True)
    return lambdas / norm


def fixed_angle_grid_2d(
    angles_degrees: Sequence[float] = DEFAULT_FIXED_ANGLES_DEG,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Deterministic 2-D preference grid ``lambda = (cos theta, sin theta)``.

    Args:
        angles_degrees: Angles in degrees; each must lie strictly in
            ``(0, 90)`` so both weights are strictly positive.
        dtype: Output dtype.
        device: Output device.

    Returns:
        Tensor of shape ``(len(angles), 2)``, unit L2 norm per row.
    """
    angles = list(angles_degrees)
    if not angles:
        raise ValueError("angles_degrees must be non-empty.")
    for a in angles:
        if not 0.0 < float(a) < 90.0:
            raise ValueError(
                f"fixed angles must lie strictly in (0, 90) degrees, got {a}."
            )
    theta = torch.tensor([math.radians(float(a)) for a in angles], dtype=dtype,
                         device=device)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def l1_normalize(lambdas: Tensor) -> Tensor:
    """Return the L1-normalized view ``lambda / sum_j lambda_j`` (proportions)."""
    lambdas = lambdas.to(torch.double)
    return lambdas / lambdas.sum(dim=-1, keepdim=True).clamp_min(_NORM_EPS)


__all__ = [
    "DEFAULT_FIXED_ANGLES_DEG",
    "sample_positive_sphere",
    "fixed_angle_grid_2d",
    "l1_normalize",
]
