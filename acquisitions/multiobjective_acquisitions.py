"""Acquisition functions for Experiment C (multi-objective SoC).

RIGHT — :class:`RightScalarizedUCB`
    The models are generic: one GP per objective (a
    :class:`~botorch.models.ModelListGP`), trained on the raw oriented
    objectives.  The preference ``lambda`` never enters the GP targets.  The
    objective-wise UCBs

        A_{j,t}(x) = mu_{j,t}(x) + sqrt(beta_t) * sigma_{j,t}(x)

    are combined by the scalarization **after** the acquisition vector is
    built:

        a^RIGHT_t(x; lambda) = rho_lambda(A_t(x)).

WRONG
    A single scalar GP is trained on the transformed observations
    ``g_lambda(x) = rho_lambda(F(x))`` for a **fixed** lambda, and a standard
    scalar UCB is maximized.  This is exactly
    :class:`acquisitions.prior_guided.VanillaUCB`, which is reused directly
    (``a^WRONG_t(x) = mu^g_t(x) + sqrt(beta_t) * sigma^g_t(x)``) — no second
    scalarization is applied to the GP output.

Both acquisitions are sequential (q=1).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.utils.transforms import t_batch_mode_transform

from common.scalarizations import scalarize

# WRONG uses the shared scalar UCB unchanged.
from acquisitions.prior_guided import VanillaUCB  # noqa: F401  (re-exported)

Tensor = torch.Tensor

_MIN_VARIANCE: float = 1e-10


class RightScalarizedUCB(AcquisitionFunction):
    r"""SoC-compliant scalarized UCB over independent per-objective GPs.

    Args:
        model: A :class:`~botorch.models.ModelListGP` of ``m`` single-output
            GPs, one per objective, trained on the oriented objectives.
        lambdas: Preference direction, shape ``(m,)``.
        beta: Non-negative exploration parameter ``beta_t``.
        scalarization: ``'linear'`` or ``'hypervolume'``.
        reference_point: Oriented reference ``z`` (required for
            ``'hypervolume'``), shape ``(m,)``.
    """

    def __init__(
        self,
        model: ModelListGP,
        lambdas: Tensor,
        beta: float = 2.0,
        scalarization: str = "linear",
        reference_point: Optional[Tensor] = None,
    ) -> None:
        super().__init__(model=model)
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        self.beta = float(beta)
        self.scalarization = str(scalarization)
        self.register_buffer("lambdas", torch.as_tensor(lambdas, dtype=torch.double).reshape(-1))
        if reference_point is not None:
            self.register_buffer(
                "reference_point",
                torch.as_tensor(reference_point, dtype=torch.double).reshape(-1),
            )
        else:
            self.reference_point = None  # type: ignore[assignment]
        if self.scalarization == "hypervolume" and self.reference_point is None:
            raise ValueError(
                "reference_point is required for the hypervolume scalarization."
            )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the scalarized UCB at ``X`` of shape ``batch x 1 x d``."""
        if X.shape[-2] != 1:
            raise ValueError(
                f"RightScalarizedUCB supports q=1 only, got q={X.shape[-2]}."
            )
        posterior = self.model.posterior(X)
        mean = posterior.mean                                    # (batch, 1, m)
        var = posterior.variance.clamp_min(_MIN_VARIANCE)        # (batch, 1, m)
        acq_vec = mean + math.sqrt(self.beta) * var.sqrt()       # (batch, 1, m)
        ref = self.reference_point
        scal = scalarize(
            acq_vec,
            self.lambdas.to(acq_vec),
            reference_point=(ref.to(acq_vec) if ref is not None else None),
            kind=self.scalarization,
        )                                                        # (batch, 1)
        return scal.squeeze(-1)                                  # (batch,)


__all__ = ["RightScalarizedUCB", "VanillaUCB"]
