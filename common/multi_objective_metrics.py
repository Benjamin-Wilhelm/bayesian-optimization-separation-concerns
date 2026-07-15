"""Scalarized and Pareto-front metrics for Experiment C.

All functions are pure and stateless.  They operate in the **maximization**
convention on oriented objective values.

Scalarized (fixed-lambda) metrics, for a trajectory of scalarized values
``v_t = rho_lambda(F(x_t))`` and the reference optimum ``rho_lambda^* =
max_x rho_lambda(F(x))``:

- best-so-far ``b_t = max_{s<=t} v_s``;
- simple regret ``r^simple_t = rho^* - b_t``;
- instantaneous regret ``r^inst_t = rho^* - v_t``;
- cumulative regret ``R_t = sum_{s<=t} r^inst_s``.

Critical invariant: the cumulative regret accumulates the **instantaneous**
regrets, never the best-so-far simple regrets.

Pareto-front metrics reuse :mod:`common.hypervolume`
for the exact 2-D hypervolume; the empirical Bayes regret is defined here.
"""

from __future__ import annotations

import torch

from common.scalarizations import scalarize

Tensor = torch.Tensor

# Regret values below ``-REGRET_TOL`` signal a real bug (sign / reference /
# pairing error); values within ``[-REGRET_TOL, 0]`` are floating-point or
# front-discretization noise and are clamped to zero.
REGRET_TOL: float = 1e-6


def _clip_regret(regret: Tensor, tol: float = REGRET_TOL) -> Tensor:
    """Clamp tiny negative regrets to zero; raise on a genuinely negative one."""
    min_val = float(regret.min().item()) if regret.numel() else 0.0
    if min_val < -abs(tol):
        raise ValueError(
            f"Negative regret {min_val:.3e} exceeds tolerance {tol:.1e}; this "
            "indicates a reference-optimum, orientation, or pairing bug rather "
            "than numerical noise."
        )
    return regret.clamp_min(0.0)


def scalarized_trajectory_values(
    oriented_Y: Tensor,
    lambdas: Tensor,
    kind: str,
    reference_point: Tensor | None = None,
) -> Tensor:
    """Scalarize each queried oriented objective vector.

    Args:
        oriented_Y: Oriented objective vectors in evaluation order, ``(n, m)``.
        lambdas: Either a single direction ``(m,)`` (returns ``(n,)``) or a
            batch ``(M, m)`` (returns ``(M, n)``).
        kind: ``'linear'`` or ``'hypervolume'``.
        reference_point: Required for the hypervolume scalarization.

    Returns:
        ``(n,)`` for a single direction, else ``(M, n)``.
    """
    oriented_Y = oriented_Y.to(torch.double)
    lambdas = torch.as_tensor(lambdas, dtype=torch.double)
    if lambdas.ndim == 1:
        return scalarize(oriented_Y, lambdas, reference_point, kind)
    # (M, n): oriented_Y (1, n, m) x lambdas (M, 1, m)
    return scalarize(
        oriented_Y.unsqueeze(0), lambdas.unsqueeze(1), reference_point, kind
    )


def reference_scalar_optimum(
    front_oriented: Tensor,
    lambdas: Tensor,
    kind: str,
    reference_point: Tensor | None = None,
) -> Tensor:
    """``rho_lambda^* = max over the dense true front`` for each direction.

    Because both scalarizations are monotone non-decreasing per objective,
    the maximum over the whole domain is attained on the Pareto front, so a
    dense evaluation of the analytic front is exact up to discretization.

    Returns a scalar tensor for a single direction, else shape ``(M,)``.
    """
    scal = scalarized_trajectory_values(
        front_oriented, lambdas, kind, reference_point
    )
    return scal.max(dim=-1).values


def scalarized_regrets(
    v: Tensor,
    rho_star: float,
    tol: float = REGRET_TOL,
) -> dict[str, Tensor]:
    """Best-so-far, simple, instantaneous, and cumulative scalarized regret.

    Args:
        v: Scalarized trajectory values ``v_t``, shape ``(n,)``.
        rho_star: Reference optimum ``rho_lambda^*``.
        tol: Negative-regret tolerance (see :data:`REGRET_TOL`).

    Returns:
        Dict with ``best_so_far``, ``simple_regret``, ``instantaneous_regret``,
        ``cumulative_regret`` — all tensors of shape ``(n,)``.
    """
    v = v.to(torch.double).reshape(-1)
    best = torch.cummax(v, dim=0).values
    simple = _clip_regret(float(rho_star) - best, tol)
    inst = _clip_regret(float(rho_star) - v, tol)
    cum = inst.cumsum(dim=0)
    return {
        "best_so_far": best,
        "simple_regret": simple,
        "instantaneous_regret": inst,
        "cumulative_regret": cum,
    }


def empirical_bayes_regret(
    oriented_Y: Tensor,
    front_oriented: Tensor,
    eval_lambdas: Tensor,
    kind: str,
    reference_point: Tensor | None = None,
) -> Tensor:
    r"""Running empirical Bayes regret over a set of evaluation preferences.

    .. math::

        R_{Bayes}(X_t) = \frac{1}{M} \sum_{j=1}^{M}
            \Big[ \max_x \rho_{\tilde\lambda_j}(F(x))
                  - \max_{x \in X_t} \rho_{\tilde\lambda_j}(F(x)) \Big]

    The per-direction optima are taken over the dense analytic front.  The
    same ``eval_lambdas`` must be used for every method compared.

    Args:
        oriented_Y: Queried oriented objective vectors in evaluation order,
            shape ``(n, m)``.
        front_oriented: Dense analytic front (oriented), shape ``(K, m)``.
        eval_lambdas: Evaluation preferences, shape ``(M, m)``.
        kind: Scalarization kind.
        reference_point: Required for the hypervolume scalarization.

    Returns:
        Tensor of shape ``(n,)`` — Bayes regret after each evaluation.
    """
    opt = reference_scalar_optimum(
        front_oriented, eval_lambdas, kind, reference_point
    )                                                              # (M,)
    scal = scalarized_trajectory_values(
        oriented_Y, eval_lambdas, kind, reference_point
    )                                                              # (M, n)
    running = torch.cummax(scal, dim=1).values                     # (M, n)
    per_lambda_regret = (opt.unsqueeze(1) - running).clamp_min(0.0)  # (M, n)
    return per_lambda_regret.mean(dim=0)                           # (n,)


__all__ = [
    "REGRET_TOL",
    "scalarized_trajectory_values",
    "reference_scalar_optimum",
    "scalarized_regrets",
    "empirical_bayes_regret",
]
