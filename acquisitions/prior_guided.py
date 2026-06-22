"""Prior-guided acquisition functions for Experiment B.

All acquisitions assume the **maximization** convention on benchmarks kept on
their **raw (natural) scale** — only oriented to a maximization problem, never
rescaled to ``[0, 1]`` (see :class:`benchmarks.oriented.OrientedBenchmark`) —
and a GP trained on raw noisy observations ``y_obs`` of the objective ``f``.
Sequential (q=1) only.

Three distinct objects are kept separate (do not conflate them):

- ``pi(x) in [0, 1]``: the spatial Gaussian prior, used by ``AlphaPiUCB`` as a
  dimensionless acquisition weight;
- ``f_prior(x) = baseline + amplitude * pi(x)``: a prior **mean** on the ``y``
  scale, carried in the GP mean by the ``lambda_pi_*`` methods (see
  :class:`common.priors.PriorMeanModule`);
- ``f(x)``: the raw oriented objective the GP models and the regret measures.

Three methods support the prior-injection comparison:

``VanillaUCB``
    GP-UCB without prior::

        a(x) = mu(x) + sqrt(beta) * sigma(x)

``AlphaPiUCB``  (SoC-compliant)
    The GP stays generic; the Gaussian domain prior ``pi(x)`` reshapes the
    acquisition only, by weighting the exploration term.  The weight is
    normalized by ``1 - alpha`` so the bonus never drops below vanilla::

        w(x) = (1 - alpha) + alpha * pi(x)
        a(x) = mu(x) + [w(x) / (1 - alpha)] * sqrt(beta) * sigma(x)
             = mu(x) + [1 + (alpha / (1 - alpha)) * pi(x)] * sqrt(beta) * sigma(x)

    This guarantees ``a(x) >= mu(x) + sqrt(beta) * sigma(x)`` pointwise
    (the vanilla UCB), which the theoretical guarantee relies on: low-prior
    regions keep at least the vanilla exploration bonus, high-prior regions
    get up to twice it (for ``alpha = 1/2``).  ``alpha = 0`` recovers
    ``VanillaUCB``; ``alpha`` must be ``< 1`` (the normalization diverges).

``LambdaPriorUCB``  (SoC-violating baselines)
    Standard UCB on the posterior of a GP whose **prior mean** already
    encodes the prior, ``f ~ GP(lambda * f_prior, k)`` with the prior mean on
    the ``y`` scale (see :class:`common.priors.PriorMeanModule`).  No residual
    reconstruction is needed: the posterior returned by the model represents
    ``f`` directly.

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
    r"""SoC-compliant prior-weighted GP-UCB (normalized / theoretically safe).

    The prior weights the exploration term only, with the weight normalized
    by ``1 - alpha``:

    .. math::

        w(x) &= (1 - \alpha) + \alpha \, \pi(x) \\
        a_t(x) &= \mu_t(x)
            + \frac{w(x)}{1 - \alpha} \, \sqrt{\beta_t} \, \sigma_t(x)
            = \mu_t(x)
            + \Big(1 + \tfrac{\alpha}{1 - \alpha}\,\pi(x)\Big)
              \sqrt{\beta_t} \, \sigma_t(x)

    Since ``w(x) / (1 - \alpha) \ge 1``, the acquisition dominates vanilla
    UCB pointwise, :math:`a_t(x) \ge \mu_t(x) + \sqrt{\beta_t}\,\sigma_t(x)`,
    which the regret guarantee relies on.  For ``alpha = 1/2`` the
    multiplier is ``1 + pi(x)``, in ``[1, 2]``.

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
        alpha: Prior-strength weight in ``[0, 1)``.  ``alpha = 1`` is
            rejected because the ``1 / (1 - alpha)`` normalization diverges.
        beta: Nonnegative exploration parameter ``beta_t``.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        prior: GaussianPrior,
        alpha: float = 0.25,
        beta: float = 2.0,
        ) -> None:
        super().__init__(model=model, beta=beta)
        if not 0.0 <= alpha < 1.0:
            raise ValueError(
                f"alpha must be in [0, 1) (the 1/(1 - alpha) normalization "
                f"diverges at 1), got {alpha}"
            )
        self.alpha: float = float(alpha)
        # Prior gain alpha / (1 - alpha): multiplier of sigma is 1 + gain * pi.
        self._pi_gain: float = self.alpha / (1.0 - self.alpha)
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
        # Normalized weight w(x) / (1 - alpha) = 1 + (alpha / (1 - alpha)) * pi,
        # >= 1 everywhere so the bonus never drops below the vanilla one.
        weight = 1.0 + self._pi_gain * pi
        return mu + weight * math.sqrt(self.beta) * sig


class LambdaPriorUCB(VanillaUCB):
    r"""Standard UCB on a prior-mean GP posterior.

    Mathematically identical to :class:`VanillaUCB`,

    .. math::

        a_t(x) = \mu_t(x) + \sqrt{\beta_t} \, \sigma_t(x),

    but documented as a distinct class because its contract differs: the
    model is expected to carry the prior in its **mean module**
    (``f ~ GP(lambda * f_prior, k)`` with ``f_prior = baseline + amplitude *
    pi`` on the ``y`` scale, fixed or adaptive ``lambda`` — see
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
