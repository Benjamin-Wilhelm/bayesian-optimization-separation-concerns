"""Scalarization functions rho_lambda for multi-objective BO (Experiment C).

Two monotone scalarizations, both defined in the **maximization** convention
(``values`` are oriented objective vectors ``F`` or objective-wise
acquisitions ``A``):

Weighted sum (``linear``)
    ``rho^lin_lambda(y) = sum_j lambda_j * y_j``.

Hypervolume / Golovin--Zhang radial scalarization (``hypervolume``)
    ``rho^HV_lambda(y - z) = [ min_j max(0, y_j - z_j) / lambda_j ]^m``,
    with ``z`` the (oriented) reference point and ``lambda`` clamped away from
    zero purely for numerical safety.

Shape contract (documented precisely to avoid ambiguous broadcasting):

- ``values``: ``(..., m)`` — the last dimension indexes objectives.
- ``lambdas``: ``(m,)`` or any shape broadcastable against ``values`` on the
  leading dimensions, ending in ``m``.
- ``reference_point`` (hypervolume only): ``(m,)``.
- Output: ``values.shape[:-1]`` after broadcasting with ``lambdas``.

This covers all required cases:

- one point, one weight            -> ``(m,) , (m,)          -> scalar``
- several points, one weight       -> ``(n, m) , (m,)        -> (n,)``
- several points, several weights  -> ``(1, n, m) , (M, 1, m) -> (M, n)``
- BoTorch t-batch acquisition      -> ``(b, m) , (m,)        -> (b,)``

All functions are vectorised, autograd-compatible, and differentiable almost
everywhere.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor

# Central numerical floor for weights in the hypervolume scalarization.  It
# only protects the division; sampled weights themselves stay strictly
# non-negative (see :mod:`lambda_sampling`).
HYPERVOLUME_LAMBDA_EPS: float = 1e-9


def _check_last_dim(values: Tensor, lambdas: Tensor) -> None:
    if values.shape[-1] != lambdas.shape[-1]:
        raise ValueError(
            f"values and lambdas must share the objective dimension m; got "
            f"values[..., {values.shape[-1]}] and lambdas[..., {lambdas.shape[-1]}]."
        )


def scalarize_linear(values: Tensor, lambdas: Tensor) -> Tensor:
    r"""Weighted-sum scalarization ``sum_j lambda_j * y_j``.

    The maximizer is invariant to a strictly-positive rescaling of all
    weights (only the *direction* of ``lambda`` matters).
    """
    values = values.to(torch.double)
    lambdas = torch.as_tensor(lambdas, dtype=torch.double)
    _check_last_dim(values, lambdas)
    return (values * lambdas).sum(dim=-1)


def scalarize_hypervolume(
    values: Tensor,
    lambdas: Tensor,
    reference_point: Tensor,
    eps: float = HYPERVOLUME_LAMBDA_EPS,
) -> Tensor:
    r"""Radial hypervolume scalarization (maximization convention).

    .. math::

        \rho^{HV}_\lambda(y) =
            \Big[ \min_j \frac{\max(0, y_j - z_j)}{\lambda_j} \Big]^m

    Values that do not dominate ``z`` in some coordinate are truncated to
    zero there, so the scalarization is zero whenever ``y`` fails to dominate
    ``z``.  ``lambda`` is clamped to ``eps`` from below for the division only.

    Args:
        values: Oriented objective/acquisition vectors, ``(..., m)``.
        lambdas: Preference directions, broadcastable, ending in ``m``.
        reference_point: Oriented reference ``z``, shape ``(m,)``.
        eps: Numerical floor applied to ``lambda`` before dividing.
    """
    values = values.to(torch.double)
    lambdas = torch.as_tensor(lambdas, dtype=torch.double)
    _check_last_dim(values, lambdas)
    z = torch.as_tensor(reference_point, dtype=torch.double).reshape(-1)
    if z.numel() != values.shape[-1]:
        raise ValueError(
            f"reference_point must have m={values.shape[-1]} entries, "
            f"got {z.numel()}."
        )
    m = values.shape[-1]
    lambda_safe = lambdas.clamp_min(eps)
    gap = (values - z).clamp_min(0.0)
    ratio = gap / lambda_safe
    inner = ratio.min(dim=-1).values
    return inner.pow(m)


def scalarize(
    values: Tensor,
    lambdas: Tensor,
    reference_point: Tensor | None = None,
    kind: str = "linear",
) -> Tensor:
    """Dispatch to :func:`scalarize_linear` or :func:`scalarize_hypervolume`.

    Args:
        values: Oriented objective vectors, ``(..., m)``.
        lambdas: Preference directions, broadcastable, ending in ``m``.
        reference_point: Required for ``kind='hypervolume'``.
        kind: ``'linear'`` or ``'hypervolume'``.
    """
    if kind == "linear":
        return scalarize_linear(values, lambdas)
    if kind == "hypervolume":
        if reference_point is None:
            raise ValueError(
                "reference_point is required for the hypervolume scalarization."
            )
        return scalarize_hypervolume(values, lambdas, reference_point)
    raise ValueError(f"Unknown scalarization kind '{kind}'. Use 'linear' or 'hypervolume'.")


SCALARIZATION_KINDS: tuple[str, ...] = ("linear", "hypervolume")


__all__ = [
    "HYPERVOLUME_LAMBDA_EPS",
    "SCALARIZATION_KINDS",
    "scalarize_linear",
    "scalarize_hypervolume",
    "scalarize",
]
