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

- :class:`PriorMeanModule` — a GPyTorch mean module
  ``m(x) = lambda * f_prior(x)`` with ``f_prior(x) = baseline + amplitude *
  pi(x)`` living on the observation (``y``) scale, and ``lambda`` registered
  as a (optionally trainable) torch parameter, so it is optimized **jointly**
  with the kernel hyperparameters by the standard BoTorch / GPyTorch
  marginal-likelihood fitting procedure, and serialized through ``state_dict``
  like every other GP parameter.

  The ``(baseline, amplitude)`` pair is what keeps the prior mean on the same
  scale as the targets ``y``: the spatial weight ``pi`` is dimensionless in
  ``[0, 1]``, whereas ``f_prior`` is an estimate of the objective ``f``.  For
  the raw-scale Experiment B benchmarks these come from
  :meth:`benchmarks.oriented.OrientedBenchmark.prior_mean_scale`.  The legacy
  default ``(baseline, amplitude) = (0, 1)`` reduces to ``m(x) = lambda *
  pi(x)``.

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
from torch.quasirandom import SobolEngine
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.means import Mean

from common.models import (
    GPModelBuilder,
    GPModelConfig,
    make_covar_module,
    validate_training_data,
)


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
    r"""GPyTorch mean module ``m(x) = lambda * f_prior(x)``.

    Implements the prior-mean GP

    .. math::

        f \sim \mathcal{GP}(\lambda \, f_{\mathrm{prior}}, \, k_\theta),
        \qquad f_{\mathrm{prior}}(x) = b + a \, \pi_c(x),

    where ``b`` (``baseline``) and ``a`` (``amplitude``) place the prior mean
    on the **observation (y) scale** — the same scale as the training targets
    — while the spatial weight ``pi_c(x)`` stays dimensionless in ``[0, 1]``.
    With the default ``(b, a) = (0, 1)`` this reduces to ``m(x) = lambda *
    pi(x)``.

    ``lambda`` is a plain torch parameter (identity parameterization,
    unconstrained).  With ``learn_lambda=True`` it is optimized jointly with
    the kernel hyperparameters by marginal-likelihood maximization; with
    ``learn_lambda=False`` it stays frozen at ``lambda_init`` (e.g. ``1.0``
    for the fixed-injection baseline).  Either way the parameter lives in
    ``state_dict()`` under ``raw_lambda``, so fixed and adaptive models share
    identical state-dict keys and serialize/deserialize uniformly through
    :func:`common.models.serialize_state_dict` /
    :func:`common.models.load_state_dict_into`.

    The prior's ``center``, ``lengthscale`` and the ``(baseline, amplitude)``
    scaling are registered as buffers: they are saved in the state dict for
    self-description but are not trainable.

    .. warning::

        The mean operates on whatever input representation the surrounding
        model feeds it.  When the GP uses a ``Normalize`` input transform,
        pass a prior already expressed in unit-cube coordinates (see
        :meth:`GaussianPrior.transformed_to_unit_cube`).  ``baseline`` and
        ``amplitude`` are on the ``y`` scale and are therefore independent of
        any input transform.
    """

    def __init__(
        self,
        prior: GaussianPrior,
        lambda_init: float = 1.0,
        learn_lambda: bool = True,
        baseline: float = 0.0,
        amplitude: float = 1.0,
        standardize_prior: bool = False,
        prior_loc: float = 0.0,
        prior_scale: float = 1.0,
        standardization_fallback: bool = False,
        ) -> None:
        super().__init__()
        self.register_buffer("prior_center", prior.center.clone())
        self.register_buffer("prior_lengthscale", prior.lengthscale.clone())
        self.register_buffer(
            "baseline", torch.tensor(float(baseline), dtype=torch.double)
        )
        self.register_buffer(
            "amplitude", torch.tensor(float(amplitude), dtype=torch.double)
        )
        self.register_parameter(
            "raw_lambda",
            torch.nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.double)),
        )
        if not learn_lambda:
            self.raw_lambda.requires_grad_(False)
        self.learn_lambda = bool(learn_lambda)
        # Optional P-BO-style dynamic standardization of the prior weight:
        # f_prior(x) = (pi(x) - prior_loc) / prior_scale  instead of
        # baseline + amplitude * pi(x).  The extra buffers are registered ONLY
        # in this mode, so the raw (baseline/amplitude) methods keep their exact
        # state_dict keys and previously-saved snapshots still reconstruct.
        self.standardize_prior = bool(standardize_prior)
        if self.standardize_prior:
            self.register_buffer(
                "prior_loc", torch.tensor(float(prior_loc), dtype=torch.double)
            )
            self.register_buffer(
                "prior_scale", torch.tensor(float(prior_scale), dtype=torch.double)
            )
            self.register_buffer(
                "standardization_fallback",
                torch.tensor(
                    1.0 if standardization_fallback else 0.0, dtype=torch.double
                ),
            )

    @property
    def lambda_value(self) -> float:
        """Current value of ``lambda`` as a Python float."""
        return float(self.raw_lambda.detach().item())

    def _pi(self, x: torch.Tensor) -> torch.Tensor:
        c = self.prior_center.to(dtype=x.dtype, device=x.device)
        l = self.prior_lengthscale.to(dtype=x.dtype, device=x.device)
        z = (x - c) / l
        return torch.exp(-0.5 * z.pow(2).sum(dim=-1))

    def _f_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Prior-mean function on the GP's target scale.

        Two mutually-exclusive parameterizations:

        - raw:          ``f_prior(x) = baseline + amplitude * pi(x)`` (y scale);
        - standardized: ``f_prior(x) = (pi(x) - prior_loc) / prior_scale``, a
          z-scored weight used together with a ``Standardize(Y)`` outcome
          transform so ``lambda`` couples two O(1) standardized quantities
          rather than absorbing the raw objective scale.
        """
        pi = self._pi(x)
        if self.standardize_prior:
            loc = self.prior_loc.to(dtype=x.dtype, device=x.device)
            scale = self.prior_scale.to(dtype=x.dtype, device=x.device)
            return (pi - loc) / scale
        b = self.baseline.to(dtype=x.dtype, device=x.device)
        a = self.amplitude.to(dtype=x.dtype, device=x.device)
        return b + a * pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean ``lambda * f_prior(x)`` for ``x`` of shape ``(..., n, d)``.

        Returns shape ``(..., n)`` as required by GPyTorch mean modules.
        """
        return self.raw_lambda * self._f_prior(x)


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
        prior_baseline: ``baseline`` of the prior-mean function
            ``f_prior(x) = baseline + amplitude * pi(x)``, on the ``y`` scale.
        prior_amplitude: ``amplitude`` of the prior-mean function, on the
            ``y`` scale.  The default ``(0, 1)`` reduces to ``m = lambda *
            pi``.
    """

    # Numerical floor for the prior-weight standard deviation; also the
    # threshold below which the per-train-point estimate is judged degenerate
    # and the dense-grid fallback is used instead.
    _PRIOR_STD_EPS: float = 1e-6
    # Deterministic dense grid used for the standardization fallback.
    _FALLBACK_GRID_SIZE: int = 4096

    def __init__(
        self,
        config: Optional[GPModelConfig] = None,
        *,
        bounds: Optional[torch.Tensor] = None,
        prior: Optional[GaussianPrior] = None,
        lambda_init: float = 1.0,
        learn_lambda: bool = True,
        prior_baseline: float = 0.0,
        prior_amplitude: float = 1.0,
        standardize_prior_mean: bool = False,
        ) -> None:
        super().__init__(config)
        self.bounds = (
            torch.as_tensor(bounds, dtype=torch.double) if bounds is not None else None
        )
        self.prior = prior
        self.lambda_init = float(lambda_init)
        self.learn_lambda = bool(learn_lambda)
        self.prior_baseline = float(prior_baseline)
        self.prior_amplitude = float(prior_amplitude)
        self.standardize_prior_mean = bool(standardize_prior_mean)
        # The raw (baseline/amplitude) prior mean lives on the y scale and is
        # incompatible with a Standardize outcome transform.  The standardized
        # prior mode is the exception: it *requires* Standardize(Y) because the
        # prior weight pi(x) is itself z-scored, so both sides of lambda are
        # standardized and lambda stays dimensionless.
        if (
            prior is not None
            and self.config.standardize_output
            and not self.standardize_prior_mean
        ):
            raise ValueError(
                "PriorGPModelBuilder with a (raw) prior mean requires "
                "standardize_output=False: a Standardize outcome transform "
                "would rescale the targets while the mean module "
                "lambda * f_prior(x) stays on the raw observation scale, so "
                "lambda would have to absorb the scale mismatch instead of "
                "learning a genuine prior/objective alignment.  The "
                "raw prior-to-the-mean methods therefore disable "
                "standardization and rely on f_prior being built on the y "
                "scale.  Use standardize_prior_mean=True (with "
                "standardize_output=True) for the P-BO-style standardized "
                "variant."
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
        covar_module = make_covar_module(self.config.kernel, train_X.shape[-1])
        if covar_module is not None:
            kwargs["covar_module"] = covar_module

        if self.config.noise_std is not None:
            kwargs["train_Yvar"] = torch.full_like(train_Y, self.config.noise_std ** 2)
            model: SingleTaskGP = FixedNoiseGP(**kwargs)
        else:
            model = SingleTaskGP(**kwargs)

        if self.prior is not None:
            mean_prior = self.prior
            if self.config.normalize_input and self.bounds is not None:
                mean_prior = self.prior.transformed_to_unit_cube(self.bounds)
            if self.standardize_prior_mean:
                loc, scale, fallback = self._prior_standardization(train_X)
                model.mean_module = PriorMeanModule(
                    prior=mean_prior,
                    lambda_init=self.lambda_init,
                    learn_lambda=self.learn_lambda,
                    standardize_prior=True,
                    prior_loc=loc,
                    prior_scale=scale,
                    standardization_fallback=fallback,
                ).to(torch.double)
            else:
                model.mean_module = PriorMeanModule(
                    prior=mean_prior,
                    lambda_init=self.lambda_init,
                    learn_lambda=self.learn_lambda,
                    baseline=self.prior_baseline,
                    amplitude=self.prior_amplitude,
                ).to(torch.double)

        model.to(train_X.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Prior-weight standardization (P-BO-style)
    # ------------------------------------------------------------------

    def _fallback_grid(self) -> torch.Tensor:
        """Deterministic dense grid over the domain for standardization.

        Used when the prior weight has (near-)zero spread across the training
        inputs (e.g. a single initial point, or all points far from the prior
        center), so ``pi(X_train).std()`` cannot define a usable scale.
        Native-domain coordinates, shape ``(G, dim)``.
        """
        d = self.prior.dim
        engine = SobolEngine(dimension=d, scramble=True, seed=0)
        u = engine.draw(self._FALLBACK_GRID_SIZE).to(torch.double)  # in [0, 1]^d
        if self.bounds is not None:
            lb, ub = self.bounds[0], self.bounds[1]
            return lb + (ub - lb) * u
        return u

    def _prior_standardization(
        self,
        train_X: torch.Tensor,
        ) -> tuple[float, float, bool]:
        """``(prior_loc, prior_scale, fallback_used)`` for the standardized prior.

        ``pi`` is z-scored over the training inputs.  When that spread is below
        :attr:`_PRIOR_STD_EPS` (degenerate, e.g. one initial point) the
        statistics are taken from a deterministic dense grid instead, and the
        fallback flag is set.  ``prior_scale`` is floored at the same eps.
        """
        p = self.prior.evaluate(train_X.to(torch.double)).reshape(-1)
        loc = p.mean() if p.numel() else torch.zeros((), dtype=torch.double)
        scale = (
            p.std(unbiased=False)
            if p.numel() >= 2
            else torch.zeros((), dtype=torch.double)
        )
        fallback = bool(scale < self._PRIOR_STD_EPS)
        if fallback:
            pg = self.prior.evaluate(self._fallback_grid()).reshape(-1)
            loc = pg.mean()
            scale = pg.std(unbiased=False)
        scale = scale.clamp_min(self._PRIOR_STD_EPS)
        return float(loc), float(scale), fallback

    def extract_hyperparameters(self, model: Any) -> dict[str, Any]:
        """Parent extraction plus prior-mean / standardization diagnostics.

        Adds ``lambda`` (and ``lambda_learned``) for any prior-mean GP, and,
        for the standardized variant, the prior z-scoring stats
        (``prior_loc``, ``prior_scale``, ``standardization_fallback``) and the
        outcome-transform stats (``y_mean``, ``y_std``).  The kernel
        ``outputscale`` / ``lengthscale`` and ``noise_variance_{raw,
        standardized}`` already come from the parent extractor.
        """
        params = super().extract_hyperparameters(model)
        mean = getattr(model, "mean_module", None)
        if isinstance(mean, PriorMeanModule):
            params["lambda"] = mean.lambda_value
            params["lambda_learned"] = mean.learn_lambda
            if mean.standardize_prior:
                params["prior_loc"] = float(mean.prior_loc.item())
                params["prior_scale"] = float(mean.prior_scale.item())
                params["standardization_fallback"] = bool(
                    mean.standardization_fallback.item() > 0.5
                )
        outcome = getattr(model, "outcome_transform", None)
        if outcome is not None and hasattr(outcome, "means") and hasattr(outcome, "stdvs"):
            params["y_mean"] = float(outcome.means.detach().mean().item())
            params["y_std"] = float(outcome.stdvs.detach().mean().item())
        return params


__all__ = [
    "GaussianPrior",
    "PriorMeanModule",
    "PriorGPModelBuilder",
]
