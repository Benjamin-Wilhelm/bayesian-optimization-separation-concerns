"""Gaussian search-domain priors and prior-mean GP construction.

Shared infrastructure for prior-guided BO (Experiment B and, later,
Experiment C).  Three building blocks:

- :class:`GaussianPrior` — an unnormalized Gaussian bump on the search
  domain::

      pi_c(x) = exp( -||x - c||^2 / (2 l^2) )           (isotropic form)

  with optional per-dimension lengthscales::

      pi_c(x) = exp( -0.5 * sum_j ((x_j - c_j) / l_j)^2 )

  The same object is consumed in two different ways:

  * SoC-compliant (``alpha_pi_ucb``): the prior enters the **acquisition**
    only (see :mod:`acquisitions.prior_guided`); the GP stays generic.
  * SoC-violating baselines (``lambda_pi_*``): the prior becomes the GP
    prior mean ``f'(x) = pi_c(x)`` via :class:`PriorMeanModule`.

- :class:`PriorMeanModule` — a GPyTorch mean module ``m(x) = lambda * pi(x)``
  with ``lambda`` registered as a (optionally trainable) torch parameter, so
  it is optimized **jointly** with the kernel hyperparameters by the standard
  BoTorch / GPyTorch marginal-likelihood fitting procedure, and serialized
  through ``state_dict`` like every other GP parameter.

- :class:`PriorGPModelBuilder` — an extension (subclass) of
  :class:`common.models.GPModelBuilder` that attaches the prior mean module
  at construction time.  ``common/models.py`` itself stays untouched and
  task-agnostic.

Scientific contract: the GP always models the original objective ``f`` and
trains on raw noisy ``y_obs``.  The prior mean shifts the GP prior; it never
transforms the training observations.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import torch
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.means import Mean

from common.models import GPModelBuilder, GPModelConfig, validate_training_data


# ---------------------------------------------------------------------------
# GaussianPrior
# ---------------------------------------------------------------------------

class GaussianPrior:
    r"""Unnormalized Gaussian prior bump on the search domain.

    .. math::

        \pi_c(x) = \exp\Big( -\tfrac{1}{2} \sum_j
            \big( (x_j - c_j) / l_j \big)^2 \Big) \in (0, 1]

    With a scalar lengthscale this reduces to the isotropic form
    ``exp(-||x - c||^2 / (2 l^2))``.

    Args:
        center: Prior center ``c``, shape ``(dim,)`` (list or tensor),
            expressed in the problem's native domain coordinates.
        lengthscale: Scalar ``l`` or per-dimension lengthscales, shape
            ``(dim,)``.  Must be strictly positive.
    """

    def __init__(
        self,
        center: Union[list, torch.Tensor],
        lengthscale: Union[float, list, torch.Tensor],
        ) -> None:
        center_t = torch.as_tensor(center, dtype=torch.double).flatten()
        if center_t.numel() == 0:
            raise ValueError("center must contain at least one coordinate.")
        ls_t = torch.as_tensor(lengthscale, dtype=torch.double).flatten()
        if ls_t.numel() == 1:
            ls_t = ls_t.expand(center_t.numel()).clone()
        if ls_t.numel() != center_t.numel():
            raise ValueError(
                f"lengthscale must be a scalar or have the same length as "
                f"center ({center_t.numel()}), got {ls_t.numel()} entries."
            )
        if not torch.all(ls_t > 0):
            raise ValueError(f"lengthscale must be strictly positive, got {ls_t.tolist()}.")
        self.center = center_t
        self.lengthscale = ls_t

    @property
    def dim(self) -> int:
        return int(self.center.numel())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate ``pi(X)`` for ``X`` of shape ``(..., dim)``.

        Returns a tensor of shape ``(...)`` with values in ``(0, 1]``.
        """
        if X.shape[-1] != self.dim:
            raise ValueError(
                f"X last dimension must be {self.dim}, got {X.shape[-1]}."
            )
        c = self.center.to(dtype=X.dtype, device=X.device)
        l = self.lengthscale.to(dtype=X.dtype, device=X.device)
        z = (X - c) / l
        return torch.exp(-0.5 * z.pow(2).sum(dim=-1))

    __call__ = evaluate

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def transformed_to_unit_cube(self, bounds: torch.Tensor) -> "GaussianPrior":
        """Re-express the prior in unit-cube coordinates.

        If ``u = (x - lb) / (ub - lb)`` is the input seen by a model with a
        fixed-bounds ``Normalize`` transform, the prior with center
        ``(c - lb) / (ub - lb)`` and per-dimension lengthscales
        ``l_j / (ub_j - lb_j)`` satisfies ``pi_unit(u) = pi(x)`` exactly.

        Args:
            bounds: Box bounds, shape ``(2, dim)``.

        Returns:
            New :class:`GaussianPrior` in unit-cube coordinates.
        """
        bounds = torch.as_tensor(bounds, dtype=torch.double)
        if bounds.shape != (2, self.dim):
            raise ValueError(
                f"bounds must have shape (2, {self.dim}), got {tuple(bounds.shape)}."
            )
        lb, ub = bounds[0], bounds[1]
        span = ub - lb
        return GaussianPrior(
            center=(self.center - lb) / span,
            lengthscale=self.lengthscale / span,
        )

    # ------------------------------------------------------------------
    # (De)serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "lengthscale": self.lengthscale.tolist(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        dim: Optional[int] = None,
        ) -> "GaussianPrior":
        """Build a prior from a YAML/JSON sub-config.

        Expected keys: ``center`` (list of length ``dim``) and
        ``lengthscale`` (scalar or list).  Extra descriptive keys such as
        ``label`` are ignored here and may be stored separately by callers.
        """
        if "center" not in config or "lengthscale" not in config:
            raise ValueError(
                "Prior config requires 'center' and 'lengthscale' keys, got "
                f"{sorted(config)}."
            )
        prior = cls(center=config["center"], lengthscale=config["lengthscale"])
        if dim is not None and prior.dim != dim:
            raise ValueError(
                f"Prior center has {prior.dim} coordinates but the problem "
                f"dimension is {dim}."
            )
        return prior

    def __repr__(self) -> str:
        return (
            f"GaussianPrior(center={self.center.tolist()}, "
            f"lengthscale={self.lengthscale.tolist()})"
        )


# ---------------------------------------------------------------------------
# PriorMeanModule
# ---------------------------------------------------------------------------

class PriorMeanModule(Mean):
    r"""GPyTorch mean module ``m(x) = lambda * pi(x)``.

    Implements the prior-mean GP

    .. math::

        f \sim \mathcal{GP}(\lambda \, f', \, k_\theta),
        \qquad f'(x) = \pi_c(x),

    where ``lambda`` is a plain torch parameter (identity parameterization,
    unconstrained).  With ``learn_lambda=True`` it is optimized jointly with
    the kernel hyperparameters by marginal-likelihood maximization; with
    ``learn_lambda=False`` it stays frozen at ``lambda_init`` (e.g. ``1.0``
    for the fixed-injection baseline).  Either way the parameter lives in
    ``state_dict()`` under ``raw_lambda``, so fixed and adaptive models share
    identical state-dict keys and serialize/deserialize uniformly through
    :func:`common.models.serialize_state_dict` /
    :func:`common.models.load_state_dict_into`.

    The prior's ``center`` and ``lengthscale`` are registered as buffers:
    they are saved in the state dict for self-description but are not
    trainable.

    .. warning::

        The mean operates on whatever input representation the surrounding
        model feeds it.  When the GP uses a ``Normalize`` input transform,
        pass a prior already expressed in unit-cube coordinates (see
        :meth:`GaussianPrior.transformed_to_unit_cube`).
    """

    def __init__(
        self,
        prior: GaussianPrior,
        lambda_init: float = 1.0,
        learn_lambda: bool = True,
        ) -> None:
        super().__init__()
        self.register_buffer("prior_center", prior.center.clone())
        self.register_buffer("prior_lengthscale", prior.lengthscale.clone())
        self.register_parameter(
            "raw_lambda",
            torch.nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.double)),
        )
        if not learn_lambda:
            self.raw_lambda.requires_grad_(False)
        self.learn_lambda = bool(learn_lambda)

    @property
    def lambda_value(self) -> float:
        """Current value of ``lambda`` as a Python float."""
        return float(self.raw_lambda.detach().item())

    def _pi(self, x: torch.Tensor) -> torch.Tensor:
        c = self.prior_center.to(dtype=x.dtype, device=x.device)
        l = self.prior_lengthscale.to(dtype=x.dtype, device=x.device)
        z = (x - c) / l
        return torch.exp(-0.5 * z.pow(2).sum(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean ``lambda * pi(x)`` for ``x`` of shape ``(..., n, d)``.

        Returns shape ``(..., n)`` as required by GPyTorch mean modules.
        """
        return self.raw_lambda * self._pi(x)


# ---------------------------------------------------------------------------
# PriorGPModelBuilder
# ---------------------------------------------------------------------------

class PriorGPModelBuilder(GPModelBuilder):
    """GP builder with optional fixed-bounds input normalization and prior mean.

    Extends :class:`common.models.GPModelBuilder` without modifying it.  Two
    additions relative to the parent:

    1. When ``bounds`` is provided and ``config.normalize_input`` is true,
       the ``Normalize`` transform uses **fixed** domain bounds instead of
       the data-dependent default.  This makes the input representation
       deterministic across iterations, which is required for an exact
       placement of the prior mean (and is applied to *all* Experiment B
       methods so model configurations stay comparable).
    2. When ``prior`` is provided, the model's mean module is replaced by a
       :class:`PriorMeanModule` ``lambda * pi(x)`` (prior re-expressed in
       unit-cube coordinates if input normalization is active).

    Reconstruction contract (``process.py``): building with the same
    ``(config, bounds, prior, lambda_init, learn_lambda)`` arguments and the
    same training data yields a state-dict-compatible model, so
    :func:`common.models.load_state_dict_into` restores snapshots without
    refitting.

    Args:
        config: Standard :class:`~common.models.GPModelConfig`.
        bounds: Problem bounds ``(2, dim)`` for the fixed ``Normalize``
            transform.  ``None`` falls back to the parent's data-dependent
            normalization.
        prior: Optional :class:`GaussianPrior` in **native domain
            coordinates**.  ``None`` keeps the default constant mean.
        lambda_init: Initial value of ``lambda``.
        learn_lambda: Whether ``lambda`` is trained by MLL fitting.
    """

    def __init__(
        self,
        config: Optional[GPModelConfig] = None,
        *,
        bounds: Optional[torch.Tensor] = None,
        prior: Optional[GaussianPrior] = None,
        lambda_init: float = 1.0,
        learn_lambda: bool = True,
        ) -> None:
        super().__init__(config)
        self.bounds = (
            torch.as_tensor(bounds, dtype=torch.double) if bounds is not None else None
        )
        self.prior = prior
        self.lambda_init = float(lambda_init)
        self.learn_lambda = bool(learn_lambda)
        if prior is not None and self.config.standardize_output:
            raise ValueError(
                "PriorGPModelBuilder with a prior mean requires "
                "standardize_output=False: a Standardize outcome transform "
                "would rescale the targets while the mean module "
                "lambda * pi(x) stays on the raw observation scale.  "
                "Experiment B uses output-normalized benchmarks, so "
                "standardization is unnecessary."
            )

    def construct(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        ) -> SingleTaskGP:
        """Build (without fitting) a GP, attaching the prior mean if configured.

        Mirrors :meth:`GPModelBuilder.construct`, with fixed-bounds input
        normalization and the optional :class:`PriorMeanModule`.
        """
        train_X, train_Y = validate_training_data(train_X, train_Y)

        kwargs: dict[str, Any] = {"train_X": train_X, "train_Y": train_Y}
        if self.config.normalize_input:
            if self.bounds is not None:
                kwargs["input_transform"] = Normalize(
                    d=train_X.shape[-1],
                    bounds=self.bounds.to(train_X.device),
                )
            else:
                kwargs["input_transform"] = Normalize(d=train_X.shape[-1])
        if self.config.standardize_output:
            kwargs["outcome_transform"] = Standardize(m=train_Y.shape[-1])

        if self.config.noise_std is not None:
            kwargs["train_Yvar"] = torch.full_like(train_Y, self.config.noise_std ** 2)
            model: SingleTaskGP = FixedNoiseGP(**kwargs)
        else:
            model = SingleTaskGP(**kwargs)

        if self.prior is not None:
            mean_prior = self.prior
            if self.config.normalize_input and self.bounds is not None:
                mean_prior = self.prior.transformed_to_unit_cube(self.bounds)
            model.mean_module = PriorMeanModule(
                prior=mean_prior,
                lambda_init=self.lambda_init,
                learn_lambda=self.learn_lambda,
            ).to(torch.double)

        model.to(train_X.device)
        model.eval()
        return model

    def extract_hyperparameters(self, model: Any) -> dict[str, Any]:
        """Parent extraction plus the prior-mean ``lambda`` when present."""
        params = super().extract_hyperparameters(model)
        mean = getattr(model, "mean_module", None)
        if isinstance(mean, PriorMeanModule):
            params["lambda"] = mean.lambda_value
            params["lambda_learned"] = mean.learn_lambda
        return params


__all__ = [
    "GaussianPrior",
    "PriorMeanModule",
    "PriorGPModelBuilder",
]
