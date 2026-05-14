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
            kwargs["input_transform"] = Normalize(d=train_X.shape[-1])
        if self.config.standardize_output:
            kwargs["outcome_transform"] = Standardize(m=train_Y.shape[-1])

        if self.config.noise_std is not None:
            kwargs["train_Yvar"] = torch.full_like(train_Y, self.config.noise_std ** 2)
            model: SingleTaskGP = FixedNoiseGP(**kwargs)
        else:
            model = SingleTaskGP(**kwargs)
        model.to(train_X.device)
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
