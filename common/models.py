"""Generic GP model utilities.

Provides construction, fitting, prediction, hyperparameter extraction, and
deterministic reconstruction for BoTorch GPs.  No task-specific or
experiment-specific logic lives here.

Scientific contract: the BO loop trains on ``obs.y_obs`` only.  This module
operates on plain tensors; the caller is responsible for passing the right
quantity and must never pass ``f_noiseless`` as training targets.

Two construction paths are exposed:

- :meth:`GPModelBuilder.fit`        — fresh model + MLL hyperparameter fitting.
- :meth:`GPModelBuilder.construct`  — fresh model, **no** MLL fitting.  Used
  by ``process.py`` together with :func:`load_state_dict_into` to rebuild a
  GP from stored hyperparameters without any retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except ImportError:
    from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from linear_operator.utils.cholesky import psd_safe_cholesky


# ---------------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------------
#
# ``matern52`` reproduces the BoTorch default (Matern 5/2 with Gamma priors on
# the lengthscale and outputscale); it is left to BoTorch so that previously
# generated runs stay bit-for-bit reproducible.  The other kernels are built
# explicitly here, with the SAME Gamma hyperpriors as the BoTorch default, so
# a kernel comparison changes only the covariance function, not the
# regularisation.

SUPPORTED_KERNELS: tuple[str, ...] = ("matern52", "matern32", "matern12", "rbf")

# BoTorch's default hyperpriors (botorch <= 0.7 SingleTaskGP), reused for the
# non-default kernels so they remain comparable.
_LENGTHSCALE_PRIOR = (3.0, 6.0)   # GammaPrior(concentration, rate)
_OUTPUTSCALE_PRIOR = (2.0, 0.15)


def make_covar_module(kernel: str, ard_num_dims: int) -> Optional[Kernel]:
    """Build the ``covar_module`` for a chosen kernel name.

    Args:
        kernel: One of :data:`SUPPORTED_KERNELS`.
        ard_num_dims: Number of input dimensions (per-dimension lengthscales).

    Returns:
        ``None`` for ``"matern52"`` (use the BoTorch default unchanged), or an
        explicit :class:`~gpytorch.kernels.ScaleKernel` for the other kernels.

    Raises:
        ValueError: If ``kernel`` is not supported.
    """
    if kernel == "matern52":
        return None
    ls_prior = GammaPrior(*_LENGTHSCALE_PRIOR)
    os_prior = GammaPrior(*_OUTPUTSCALE_PRIOR)
    if kernel == "matern32":
        base: Kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims,
                                    lengthscale_prior=ls_prior)
    elif kernel == "matern12":
        base = MaternKernel(nu=0.5, ard_num_dims=ard_num_dims,
                            lengthscale_prior=ls_prior)
    elif kernel == "rbf":
        base = RBFKernel(ard_num_dims=ard_num_dims, lengthscale_prior=ls_prior)
    else:
        raise ValueError(
            f"Unsupported kernel '{kernel}'.  Supported: {SUPPORTED_KERNELS}."
        )
    return ScaleKernel(base, outputscale_prior=os_prior)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_training_data(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalise training tensors.

    - Casts both tensors to ``torch.double``.
    - Ensures ``train_X`` is 2-D ``(n, d)``.
    - Ensures ``train_Y`` is 2-D ``(n, m)``, promoting ``(n,)`` to ``(n, 1)``.
    - Checks that both tensors have the same number of rows.

    Args:
        train_X: Input tensor, shape ``(n, d)``.
        train_Y: Target tensor, shape ``(n, m)`` or ``(n,)``.

    Returns:
        Validated ``(train_X, train_Y)`` pair, both ``torch.double``.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    train_X = train_X.to(torch.double)
    train_Y = train_Y.to(torch.double)

    if train_X.ndim != 2:
        raise ValueError(
            f"train_X must be 2-D (n, d), got shape {tuple(train_X.shape)}"
        )
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2:
        raise ValueError(
            f"train_Y must be 1-D (n,) or 2-D (n, m), got shape {tuple(train_Y.shape)}"
        )
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError(
            f"train_X and train_Y must have the same number of rows, "
            f"got {train_X.shape[0]} and {train_Y.shape[0]}"
        )
    return train_X, train_Y


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPModelConfig:
    """Configuration for GP model construction and fitting.

    Attributes:
        noise_std: Fixed observation noise standard deviation.  If ``None``,
            noise variance is inferred by marginal log-likelihood optimisation
            (``SingleTaskGP``); if given, it is fixed (``FixedNoiseGP``).
        normalize_input: Apply a ``Normalize`` input transform.  Recommended
            when input bounds are not ``[0, 1]^d``.
        standardize_output: Apply a ``Standardize`` outcome transform.
            Recommended to keep the GP prior mean and variance well-scaled.
        kernel: Covariance kernel name, one of :data:`SUPPORTED_KERNELS`
            (``"matern52"`` default, ``"matern32"``, ``"matern12"``,
            ``"rbf"``).  ``"matern52"`` uses the BoTorch default unchanged;
            the others are built explicitly with the same Gamma hyperpriors
            (see :func:`make_covar_module`).
        fixed_lengthscale: Optional positive scalar or ARD vector, in
            input-transform coordinates. Freezes only the lengthscale;
            other trainable parameters still undergo MLL optimization.
        input_bounds: Optional fixed (2, d) bounds for input normalization.
            None preserves the existing data-dependent normalization.
    """

    noise_std: Optional[float] = None
    normalize_input: bool = True
    standardize_output: bool = True
    kernel: str = "matern52"
    # Lengthscales are in input-transform coordinates. None retains MLL fitting.
    fixed_lengthscale: Optional[float | list[float]] = None
    input_bounds: Optional[list[list[float]]] = None

    def __post_init__(self) -> None:
        if self.fixed_lengthscale is not None:
            values = torch.as_tensor(self.fixed_lengthscale, dtype=torch.double)
            if values.numel() == 0 or not torch.isfinite(values).all() or (values <= 0).any():
                raise ValueError("fixed_lengthscale must contain positive finite values")
        if self.kernel not in SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{self.kernel}'.  "
                f"Supported: {SUPPORTED_KERNELS}."
            )


# ---------------------------------------------------------------------------
# State-dict (de)serialisation
# ---------------------------------------------------------------------------

def serialize_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """Return a JSON-serialisable copy of ``model.state_dict()``.

    Each tensor parameter / buffer is converted via ``.tolist()``.  Scalar
    (0-D) tensors are returned as Python ``float``; higher-rank tensors as
    nested lists.  Shapes are preserved implicitly by the tolist round-trip.

    Args:
        model: Any ``torch.nn.Module``; in practice a fitted BoTorch GP.

    Returns:
        Dict ``{parameter_name: list | float}`` suitable for ``json.dump``.
    """
    return {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}


def load_state_dict_into(model: torch.nn.Module, serial: dict[str, Any]) -> None:
    """Inverse of :func:`serialize_state_dict`.

    Coerces each stored entry to the dtype and shape of the corresponding
    parameter in the freshly-constructed ``model``, then calls
    ``model.load_state_dict(...)``.

    Strict mode is enforced: every key in the stored dict must exist in the
    target model and vice versa.  This catches silent botorch / kernel changes
    between run and process.

    Args:
        model: Freshly-constructed GP with the same shape as the saved model
            (same ``train_X``/``train_Y`` shapes, same transforms, same
            kernel).  Typically produced by :meth:`GPModelBuilder.construct`.
        serial: Dict produced by :func:`serialize_state_dict`.
    """
    template = model.state_dict()
    if set(template.keys()) != set(serial.keys()):
        missing = set(template.keys()) - set(serial.keys())
        extra   = set(serial.keys()) - set(template.keys())
        raise ValueError(
            f"state_dict key mismatch.  Missing in stored state: {sorted(missing)}.  "
            f"Extra in stored state: {sorted(extra)}."
        )
    new_state: dict[str, torch.Tensor] = {}
    for k, v in serial.items():
        target = template[k]
        new_state[k] = torch.tensor(v, dtype=target.dtype).reshape(target.shape)
    model.load_state_dict(new_state)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def effective_rkhs_norm(model: SingleTaskGP) -> dict[str, Any]:
    """Norm of the centered posterior mean of a scalar, unbatched GP.

    In model coordinates, r = train_targets - prior_mean(X),
    alpha = (K + diag(noise))^{-1} r, and B² = alphaᵀ K alpha.
    K includes outputscale; noise is the likelihood's actual training noise
    (including any library floor), not the requested noise or latent variance.
    Evaluation-mode train_inputs already include the input transform.

    With Standardize(mean=a, std=s), the raw centered mean is s K(x,X) alpha
    and its raw covariance is s² K, so its norm is also B. The prior mean
    a + s*m(x) is excluded: it need not belong to the kernel's RKHS.
    For a ScaleKernel, also return the raw centered mean's norm under the
    unit-outputscale base kernel, s*sqrt(outputscale)*B. This removes changing
    amplitude conventions, but still depends on lengthscale/input coordinates.
    No fitting or explicit matrix inverse is used.
    """
    if model.training:
        raise ValueError("effective_rkhs_norm requires an evaluation-mode GP")
    X = model.train_inputs[0]
    if X.ndim != 2 or model.train_targets.ndim != 1:
        raise ValueError("effective_rkhs_norm supports scalar unbatched GPs only")
    K = model.covar_module(X).to_dense()
    residual = model.train_targets - model.mean_module(X)
    noise = model.likelihood.noise.reshape(-1).expand_as(residual)
    system = K + torch.diag(noise)
    chol, info = torch.linalg.cholesky_ex(system)
    solve_jitter = 0.0
    if info.any():
        chol = psd_safe_cholesky(system)
        solve_jitter = float((chol @ chol.T - system).diagonal().max().clamp_min(0))
    alpha = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
    norm_squared = (alpha @ K @ alpha).clamp_min(0)
    transform = getattr(model, "outcome_transform", None)
    if transform is not None and not isinstance(transform, Standardize):
        raise ValueError("RKHS output conversion supports Standardize only")
    outcome_std = float(transform.stdvs.item()) if transform is not None else 1.0
    outcome_mean = float(transform.means.item()) if transform is not None else 0.0
    outputscale = float(model.covar_module.outputscale)
    return dict(
        effective_rkhs_norm=float(norm_squared.sqrt()),
        effective_rkhs_norm_squared=float(norm_squared),
        effective_rkhs_norm_unit_outputscale=float(norm_squared.sqrt()) * outcome_std * outputscale**.5,
        rkhs_context=dict(
            convention="centered_posterior_mean_in_fitted_kernel",
            outcome_mean=outcome_mean, outcome_std=outcome_std,
            prior_mean_raw=(outcome_mean + outcome_std * model.mean_module(X)).tolist(),
            outputscale_raw=outcome_std**2 * outputscale,
            noise_variance_raw_min=float(noise.min()) * outcome_std**2,
            noise_variance_raw_max=float(noise.max()) * outcome_std**2,
            # Nonzero only if the Cholesky routine added numerical jitter.
            solve_jitter=solve_jitter,
        ),
    )


class GPModelBuilder:
    """Centralises GP construction, fitting, prediction, and diagnostics.

    Wraps a :class:`GPModelConfig` and provides a consistent interface for
    the BO loop.  The builder itself holds no mutable state and is safe to
    reuse across iterations.

    Scientific contract: ``train_Y`` must always be ``obs.y_obs`` — the noisy
    observations returned by the problem — never ``f_noiseless``.

    Args:
        config: Model configuration.  Defaults to ``GPModelConfig()`` if
            not provided.
    """

    def __init__(self, config: Optional[GPModelConfig] = None) -> None:
        self.config: GPModelConfig = config if config is not None else GPModelConfig()

    # --- construction & fitting ---

    def construct(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> SingleTaskGP:
        """Build a GP **without** fitting its hyperparameters.

        Returns a fresh model in ``eval`` mode whose kernel / mean / likelihood
        parameters carry their default initialisation.  Intended for use by
        ``process.py`` to rebuild a saved GP: call :func:`load_state_dict_into`
        on the returned model to restore the trained state.

        Uses ``FixedNoiseGP`` when ``config.noise_std`` is set, otherwise
        ``SingleTaskGP``.

        Args:
            train_X: Training inputs, shape ``(n, dim)``.
            train_Y: Training targets, shape ``(n, 1)`` or ``(n,)``.

        Returns:
            Unfitted model in ``eval`` mode.
        """
        train_X, train_Y = validate_training_data(train_X, train_Y)

        kwargs: dict[str, Any] = {"train_X": train_X, "train_Y": train_Y}
        if self.config.normalize_input:
            bounds = (torch.as_tensor(self.config.input_bounds, dtype=train_X.dtype,
                                      device=train_X.device)
                      if self.config.input_bounds is not None else None)
            if bounds is not None and (bounds.shape != (2, train_X.shape[-1])
                                       or not torch.isfinite(bounds).all()
                                       or not torch.all(bounds[0] < bounds[1])):
                raise ValueError("input_bounds must be finite ordered (2, d) bounds")
            kwargs["input_transform"] = Normalize(d=train_X.shape[-1], bounds=bounds)
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
        model.to(train_X.device)
        if self.config.fixed_lengthscale is not None:
            base = model.covar_module.base_kernel
            lengthscale = torch.as_tensor(self.config.fixed_lengthscale,
                                         dtype=train_X.dtype, device=train_X.device)
            if lengthscale.ndim > 1 or lengthscale.numel() not in (1, train_X.shape[-1]):
                raise ValueError("fixed_lengthscale must be scalar or have one value per input dimension")
            base.lengthscale = lengthscale
            base.raw_lengthscale.requires_grad_(False)
        model.eval()
        return model

    def fit(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> SingleTaskGP:
        """Construct and fit a GP on noisy training observations.

        ``train_Y`` must be ``obs.y_obs``, never ``f_noiseless`` (RIGHT), or
        the method-specific transformed target (WRONG).

        Args:
            train_X: Training inputs, shape ``(n, dim)``.
            train_Y: Training targets, shape ``(n, 1)`` or ``(n,)``.

        Returns:
            Fitted model in ``eval`` mode.
        """
        model = self.construct(train_X, train_Y)
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()
        return model

    def fit_model_list(
        self,
        train_Xs: list[torch.Tensor],
        train_Ys: list[torch.Tensor],
        noise_stds: Optional[list[Optional[float]]] = None,
    ) -> ModelListGP:
        """Fit one GP per task and return a :class:`~botorch.models.ModelListGP`.

        Each output is modelled independently.  Per-task noise overrides can
        be supplied via ``noise_stds``; otherwise ``config.noise_std`` is used
        for every task.

        All ``train_Ys`` entries must be ``obs.y_obs``, never ``f_noiseless``.

        Args:
            train_Xs: Input tensors, one per task.
            train_Ys: Noisy observations ``y_obs``, one per task.
            noise_stds: Per-task noise std overrides.  ``None`` entries inherit
                ``config.noise_std``.  Defaults to ``config.noise_std`` for all.

        Returns:
            :class:`~botorch.models.ModelListGP` in ``eval`` mode.

        Raises:
            ValueError: If list lengths are inconsistent.
        """
        n_tasks = len(train_Xs)
        if len(train_Ys) != n_tasks:
            raise ValueError(
                f"train_Xs and train_Ys must have the same length, "
                f"got {n_tasks} and {len(train_Ys)}"
            )
        if noise_stds is None:
            noise_stds = [self.config.noise_std] * n_tasks
        if len(noise_stds) != n_tasks:
            raise ValueError(
                f"noise_stds must have length {n_tasks}, got {len(noise_stds)}"
            )

        models = []
        for X, Y, std in zip(train_Xs, train_Ys, noise_stds):
            task_cfg = GPModelConfig(
                noise_std=std,
                normalize_input=self.config.normalize_input,
                standardize_output=self.config.standardize_output,
                kernel=self.config.kernel,
                fixed_lengthscale=self.config.fixed_lengthscale,
                input_bounds=self.config.input_bounds,
            )
            models.append(GPModelBuilder(task_cfg).fit(X, Y))
        return ModelListGP(*models)

    # --- prediction ---

    def predict(
        self,
        model: SingleTaskGP,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and marginal variance at test points.

        Intended for diagnostics, plotting, and acquisition debugging.

        Args:
            model: A fitted GP in ``eval`` mode.
            X: Test inputs, shape ``(n, dim)``.

        Returns:
            Tuple ``(mean, var)`` where both tensors have shape ``(n, 1)``.
        """
        X = X.to(torch.double)
        with torch.no_grad():
            posterior = model.posterior(X)
        return posterior.mean, posterior.variance

    # --- hyperparameter extraction ---

    def extract_hyperparameters(
        self,
        model: SingleTaskGP | ModelListGP,
    ) -> dict[str, Any]:
        """Extract learned hyperparameters as a JSON-serialisable dict.

        Handles both :class:`~botorch.models.SingleTaskGP` /
        :class:`~botorch.models.FixedNoiseGP` and
        :class:`~botorch.models.ModelListGP`.  For a ``ModelListGP`` the
        result contains one entry per sub-model keyed ``"model_0"``,
        ``"model_1"``, etc.

        Extracted fields (where available): ``lengthscale``, ``outputscale``,
        ``noise_variance``.

        Args:
            model: A fitted GP model.

        Returns:
            Dict with scalar or list values; safe to pass to ``json.dumps``.
        """
        if isinstance(model, ModelListGP):
            return {
                f"model_{i}": self._extract_single(m)
                for i, m in enumerate(model.models)
            }
        return self._extract_single(model)

    # --- private helpers ---

    @staticmethod
    def _extract_single(model: SingleTaskGP) -> dict[str, Any]:
        params: dict[str, Any] = {}

        try:
            noise = model.likelihood.noise.detach().squeeze()
            params["noise_variance_standardized"] = float(noise.mean().item())
            std = model.outcome_transform.stdvs.detach().flatten()
            params["outcome_std"] = float(std.mean().item())
            params["noise_variance_raw"] = params["noise_variance_standardized"] * params["outcome_std"] ** 2
        
        except AttributeError:
            pass

        covar = model.covar_module
        if hasattr(covar, "outputscale"):
            params["outputscale"] = covar.outputscale.item()
            base_kernel = covar.base_kernel
        else:
            base_kernel = covar

        if hasattr(base_kernel, "lengthscale"):
            ls = base_kernel.lengthscale.detach().squeeze()
            params["lengthscale"] = ls.tolist() if ls.numel() > 1 else ls.item()

        return params
