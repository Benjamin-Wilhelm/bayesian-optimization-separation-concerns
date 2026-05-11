"""Generic GP model utilities.

Provides construction, fitting, prediction, and hyperparameter extraction
for BoTorch GPs.  No task-specific or experiment-specific logic lives here.

Scientific contract: the BO loop trains on ``obs.y_obs`` only.  This module
operates on plain tensors; the caller is responsible for passing the right
quantity and must never pass ``f_noiseless`` as training targets.

Usage::

    config = GPModelConfig(noise_std=0.05)
    builder = GPModelBuilder(config)
    model = builder.fit(obs.X, obs.y_obs)
    mean, var = builder.predict(model, X_candidate)
    state = builder.capture_state(model, iteration=t)
    params = state.hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


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
        kernel: Kernel name.  Only ``"matern52"`` is currently supported;
            this field is reserved for future extensibility.
    """

    noise_std: Optional[float] = None
    normalize_input: bool = True
    standardize_output: bool = True
    kernel: str = "matern52"


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

@dataclass
class ModelState:
    """Serialisable snapshot of a fitted GP at one BO iteration.

    Stores lightweight metadata and hyperparameters so that model diagnostics
    can be reconstructed after the BO run without keeping the full model
    objects in memory.

    Attributes:
        iteration: BO iteration index (0-based).
        model_type: Class name of the fitted model, e.g. ``"SingleTaskGP"``
            or ``"FixedNoiseGP"``.
        kernel_name: Kernel identifier from the config, e.g. ``"matern52"``.
        hyperparameters: JSON-serialisable dict of learned hyperparameters
            (lengthscale, outputscale, noise_variance).
    """

    iteration: int
    model_type: str
    kernel_name: str
    hyperparameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "iteration": self.iteration,
            "model_type": self.model_type,
            "kernel_name": self.kernel_name,
            "hyperparameters": self.hyperparameters,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

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

    def fit(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> SingleTaskGP:
        """Construct and fit a GP on raw noisy observations.

        Uses ``FixedNoiseGP`` when ``config.noise_std`` is set, otherwise
        ``SingleTaskGP`` with inferred noise.

        ``train_Y`` must be ``obs.y_obs``, never ``f_noiseless``.

        Args:
            train_X: Training inputs, shape ``(n, dim)``.
            train_Y: Noisy observations ``y_obs``, shape ``(n, 1)`` or ``(n,)``.

        Returns:
            Fitted model in ``eval`` mode.
        """
        train_X, train_Y = validate_training_data(train_X, train_Y)

        kwargs: dict[str, Any] = {"train_X": train_X, "train_Y": train_Y}
        if self.config.normalize_input:
            kwargs["input_transform"] = Normalize(d=train_X.shape[-1])
        if self.config.standardize_output:
            kwargs["outcome_transform"] = Standardize(m=train_Y.shape[-1])

        if self.config.noise_std is not None:
            kwargs["train_Yvar"] = torch.full_like(train_Y, self.config.noise_std ** 2)
            model: SingleTaskGP = FixedNoiseGP(**kwargs)
        else:
            model = SingleTaskGP(**kwargs)

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

    # --- model state capture ---

    def capture_state(
        self,
        model: SingleTaskGP | ModelListGP,
        iteration: int,
    ) -> ModelState:
        """Capture a serialisable snapshot of the model at a given BO iteration.

        The returned :class:`ModelState` can be appended to a list across
        iterations and later serialised or used for posterior diagnostics.

        Args:
            model: A fitted GP model.
            iteration: Current BO iteration index (0-based).

        Returns:
            :class:`ModelState` with learned hyperparameters and metadata.
        """
        return ModelState(
            iteration=iteration,
            model_type=type(model).__name__,
            kernel_name=self.config.kernel,
            hyperparameters=self.extract_hyperparameters(model),
        )

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
