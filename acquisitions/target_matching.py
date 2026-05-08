"""Target-matching acquisition functions for Experiment A (SoC-compliant).

The GP is trained on raw noisy observations ``y_obs ≈ f(x)``.
The target ``T`` appears only here, in the acquisition.

This separation is the core principle of the SoC experimental design:
the model represents the unknown phenomenon; the decision objective
(proximity to target) lives exclusively in the acquisition.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import t_batch_mode_transform

_MIN_VARIANCE: float = 1e-10


class TargetMatchingUCB(AnalyticAcquisitionFunction):
    """SoC-compliant target-matching acquisition based on GP-UCB.

    The GP posterior over the raw function f gives predictive mean mu_f,t(x)
    and standard deviation sigma_f,t(x). The acquisition maximized by BoTorch is

        alpha_right_t(x) = - |mu_f,t(x) - T| + sqrt(beta_t) * sigma_f,t(x).

    Equivalently, this maximizes optimism for target matching by minimizing

        a_right_t(x) = |mu_f,t(x) - T| - sqrt(beta_t) * sigma_f,t(x).

    The first term favors points whose posterior mean is close to the target T.
    The second term encourages exploration in regions of high posterior
    uncertainty.

    SoC contract:
        The model must be trained on raw observations y_obs, i.e. noisy
        observations of f(x). It must not be trained on |y_obs - T|. The target T
        appears only in the acquisition, never in the training data.

    Args:
        model: BoTorch GP model fitted on raw y_obs. Must have one output.
        target: Scalar target value T.
        beta: Nonnegative exploration parameter beta_t. Larger values encourage
            exploration.

    Example:
        model = fit_single_task_gp(train_X, obs.y_obs)

        acq = TargetMatchingUCB(model=model, target=10.0, beta=2.0)

        candidate, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
    """

    def __init__(
        self,
        model: SingleTaskGP,
        target: float | torch.Tensor,
        beta: float = 2.0,
        ) -> None:
        super().__init__(model=model)

        if isinstance(target, torch.Tensor):
            self.register_buffer("target", target.to(torch.double).squeeze())
        else:
            self.register_buffer("target", torch.tensor(target, dtype=torch.double))

        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.beta: float = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the target-matching UCB acquisition at ``X``.

        Args:
            X: Candidate tensor of shape ``batch_shape x 1 x d``.
                The ``q`` dimension must equal 1; this acquisition is
                sequential only.  The decorator :func:`t_batch_mode_transform`
                promotes a plain ``(n, d)`` input to ``(n, 1, d)``
                automatically.

        Returns:
            Tensor of shape ``batch_shape`` — one acquisition value per
            candidate point.

        Raises:
            ValueError: If the ``q`` dimension of ``X`` is not 1.
        """
        if X.shape[-2] != 1:
            raise ValueError(
                f"TargetMatchingUCB supports q=1 only, got q={X.shape[-2]}. "
                "Use a sequential strategy or a batch-compatible acquisition "
                "for q > 1."
            )

        posterior = self.model.posterior(X)

        # shapes after posterior: batch_shape x q x 1  (single output)
        mean = posterior.mean  # (..., 1, 1)
        variance = posterior.variance.clamp_min(_MIN_VARIANCE)  # (..., 1, 1)
        sigma = variance.sqrt()

        sqrt_beta = math.sqrt(self.beta)
        distance_to_target = (mean - self.target).abs()

        # acquisition value: -|mu - T| + sqrt(beta) * sigma
        value = -distance_to_target + sqrt_beta * sigma  # (..., 1, 1)

        # squeeze q=1 and output=1 dimensions -> shape batch_shape
        return value.squeeze(-1).squeeze(-1)
