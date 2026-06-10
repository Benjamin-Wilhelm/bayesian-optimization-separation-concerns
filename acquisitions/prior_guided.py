"""Prior-guided acquisition functions for Experiment B.

All acquisitions assume the **maximization** convention on output-normalized
benchmarks (``f_norm in [0, 1]``, optimum at 1.0) and a GP trained on raw
noisy observations ``y_obs`` of the objective ``f``.  Sequential (q=1) only.

Three methods support the prior-injection comparison:

``VanillaUCB``
    GP-UCB without prior::

        a(x) = mu(x) + sqrt(beta) * sigma(x)

``AlphaPiUCB``  (SoC-compliant)
    The GP stays generic; the Gaussian domain prior ``pi(x)`` reshapes the
    acquisition only, by weighting the exploration term::

        w(x) = (1 - alpha) + alpha * pi(x)
        a(x) = mu(x) + w(x) * sqrt(beta) * sigma(x)

    ``alpha = 0`` recovers ``VanillaUCB``; ``alpha = 1`` makes the
    exploration bonus fully prior-weighted.

``LambdaPriorUCB``  (SoC-violating baselines)
    Standard UCB on the posterior of a GP whose **prior mean** already
    encodes the prior, ``f ~ GP(lambda * pi, k)`` (see
    :class:`common.priors.PriorMeanModule`).  No residual reconstruction is
    needed: the posterior returned by the model represents ``f`` directly.

SoC reading of the comparison:

    models remain generic; task-specific decision logic lives in
    acquisitions.  ``alpha_pi_ucb`` respects this; ``lambda_pi_*`` move the
    prior into the model on purpose, as the contrast condition.
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.utils.transforms import t_batch_mode_transform

from common.priors import GaussianPrior

_MIN_VARIANCE: float = 1e-10


class VanillaUCB(AnalyticAcquisitionFunction):
    r"""Standard GP-UCB for maximization.

    .. math::

        a_t(x) = \mu_t(x) + \sqrt{\beta_t} \, \sigma_t(x)

    Args:
        model: BoTorch GP fitted on raw ``y_obs``.  Single output.
        beta: Nonnegative exploration parameter ``beta_t``.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        beta: float = 2.0,
        ) -> None:
        super().__init__(model=model)
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.beta: float = float(beta)

    def _mean_sigma(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = self.model.posterior(X)
        mean = posterior.mean                                       # (..., 1, 1)
        sigma = posterior.variance.clamp_min(_MIN_VARIANCE).sqrt()  # (..., 1, 1)
        return mean, sigma

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the UCB at ``X`` of shape ``batch_shape x 1 x d``.

        Returns a tensor of shape ``batch_shape``.

        Raises:
            ValueError: If the ``q`` dimension of ``X`` is not 1.
        """
        if X.shape[-2] != 1:
            raise ValueError(
                f"{type(self).__name__} supports q=1 only, got q={X.shape[-2]}."
            )
        mean, sigma = self._mean_sigma(X)
        value = mean + math.sqrt(self.beta) * sigma  # (..., 1, 1)
        return value.squeeze(-1).squeeze(-1)


class AlphaPiUCB(VanillaUCB):
    r"""SoC-compliant prior-weighted GP-UCB.

    The prior weights the exploration term only:

    .. math::

        w(x) &= (1 - \alpha) + \alpha \, \pi(x) \\
        a_t(x) &= \mu_t(x) + w(x) \, \sqrt{\beta_t} \, \sigma_t(x)

    The prior ``pi`` is evaluated in the problem's **native domain
    coordinates** (the acquisition receives untransformed candidates from
    ``optimize_acqf``); any input normalization inside the GP is irrelevant
    here.

    SoC contract:
        The model must be a generic GP trained on raw ``y_obs`` with no
        prior in its mean.  The prior appears only in this acquisition.

    Args:
        model: BoTorch GP fitted on raw ``y_obs``.  Single output.
        prior: :class:`common.priors.GaussianPrior` in native domain
            coordinates.
        alpha: Prior-strength weight in ``[0, 1]``.
        beta: Nonnegative exploration parameter ``beta_t``.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        prior: GaussianPrior,
        alpha: float = 0.5,
        beta: float = 2.0,
        ) -> None:
        super().__init__(model=model, beta=beta)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha: float = float(alpha)
        # Stored as buffers so the acquisition is self-describing and
        # device-mobile; the prior itself is stateless.
        self.register_buffer("prior_center", prior.center.clone())
        self.register_buffer("prior_lengthscale", prior.lengthscale.clone())

    def _pi(self, X: torch.Tensor) -> torch.Tensor:
        c = self.prior_center.to(dtype=X.dtype, device=X.device)
        l = self.prior_lengthscale.to(dtype=X.dtype, device=X.device)
        z = (X - c) / l
        return torch.exp(-0.5 * z.pow(2).sum(dim=-1))

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the prior-weighted UCB at ``X`` (``batch_shape x 1 x d``)."""
        if X.shape[-2] != 1:
            raise ValueError(
                f"AlphaPiUCB supports q=1 only, got q={X.shape[-2]}."
            )
        mean, sigma = self._mean_sigma(X)
        mu = mean.squeeze(-1).squeeze(-1)       # batch
        sig = sigma.squeeze(-1).squeeze(-1)     # batch
        pi = self._pi(X.squeeze(-2))            # batch
        weight = (1.0 - self.alpha) + self.alpha * pi
        return mu + weight * math.sqrt(self.beta) * sig


class LambdaPriorUCB(VanillaUCB):
    r"""Standard UCB on a prior-mean GP posterior.

    Mathematically identical to :class:`VanillaUCB`,

    .. math::

        a_t(x) = \mu_t(x) + \sqrt{\beta_t} \, \sigma_t(x),

    but documented as a distinct class because its contract differs: the
    model is expected to carry the prior in its **mean module**
    (``f ~ GP(lambda * pi, k)``, fixed or adaptive ``lambda`` — see
    :class:`common.priors.PriorMeanModule`).  The posterior already
    represents the objective ``f``, so no residual reconstruction is
    performed here.

    Args:
        model: BoTorch GP with a :class:`~common.priors.PriorMeanModule`
            mean, fitted on raw ``y_obs``.  Single output.
        beta: Nonnegative exploration parameter ``beta_t``.
    """


__all__ = [
    "VanillaUCB",
    "AlphaPiUCB",
    "LambdaPriorUCB",
]
