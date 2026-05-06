"""Generic GP model utilities.

Provides construction, fitting, and prediction helpers for BoTorch GPs.
No task-specific or experiment-specific logic lives here.

Scientific contract: the BO loop trains on ``obs.y_obs`` only.  The
functions in this module enforce this by operating on plain tensors; the
caller is responsible for passing the right quantity.

Usage::

    model = fit_single_task_gp(obs.X, obs.y_obs, noise_std=problem.default_noise_std)
    mean, var = get_posterior_mean_var(model, X_candidate)
    params = get_hyperparameters(model)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
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
# Model construction and fitting
# ---------------------------------------------------------------------------

def fit_single_task_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    noise_std: Optional[float] = None,
) -> SingleTaskGP:
    """Construct and fit a SingleTaskGP on raw noisy observations.

    ``train_Y`` must be ``obs.y_obs`` — the noisy observations returned by
    the problem — never ``f_noiseless``.

    Input normalisation (``Normalize``) and output standardisation
    (``Standardize``) are passed explicitly.

    Args:
        train_X: Training inputs, shape ``(n, dim)``.
        train_Y: Noisy observations ``y_obs``, shape ``(n, 1)``.
        noise_std: If given, fixes the observation noise to this value.
            If ``None``, the noise variance is inferred by maximising
            the marginal log-likelihood.

    Returns:
        Fitted :class:`~botorch.models.SingleTaskGP` in ``eval`` mode.
    """
    train_X, train_Y = validate_training_data(train_X, train_Y)

    input_tf = Normalize(d=train_X.shape[-1])
    outcome_tf = Standardize(m=train_Y.shape[-1])

    if noise_std is not None:
        train_Yvar = torch.full_like(train_Y, noise_std ** 2)
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_tf,
            outcome_transform=outcome_tf,
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_tf,
            outcome_transform=outcome_tf,
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def fit_model_list_gp(
    train_Xs: list[torch.Tensor],
    train_Ys: list[torch.Tensor],
    noise_stds: Optional[list[Optional[float]]] = None,
) -> ModelListGP:
    """Fit one SingleTaskGP per task and return a ModelListGP.

    Each output is modelled independently.  Useful as a building block for
    multi-objective or constrained BO, where acquisition functions need a
    joint model object.  No acquisition logic lives here.

    Args:
        train_Xs: List of input tensors, one per task.
        train_Ys: List of target tensors ``y_obs``, one per task.
        noise_stds: Noise standard deviations, one per task.  A ``None``
            entry means noise is inferred for that task.  Defaults to
            inferring noise for all tasks.

    Returns:
        :class:`~botorch.models.ModelListGP` wrapping the fitted models,
        in ``eval`` mode.

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
        noise_stds = [None] * n_tasks
    if len(noise_stds) != n_tasks:
        raise ValueError(
            f"noise_stds must have the same length as train_Xs, "
            f"got {len(noise_stds)} and {n_tasks}"
        )

    models = [
        fit_single_task_gp(X, Y, noise_std=std)
        for X, Y, std in zip(train_Xs, train_Ys, noise_stds)
    ]
    return ModelListGP(*models)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def get_posterior_mean_var(
    model: SingleTaskGP,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return posterior mean and marginal variance at test points.

    Intended for diagnostics, plotting, and acquisition debugging.

    Args:
        model: A fitted GP in ``eval`` mode.
        X: Test inputs, shape ``(n, dim)``, ``torch.double``.

    Returns:
        Tuple ``(mean, var)`` where:

        - ``mean``: shape ``(n, 1)``, posterior mean.
        - ``var``: shape ``(n, 1)``, marginal posterior variance (diagonal
          of the full predictive covariance).
    """
    X = X.to(torch.double)
    with torch.no_grad():
        posterior = model.posterior(X)
    return posterior.mean, posterior.variance


# ---------------------------------------------------------------------------
# Hyperparameter extraction
# ---------------------------------------------------------------------------

def _single_gp_hyperparameters(model: SingleTaskGP) -> dict[str, Any]:
    """Extract hyperparameters from one SingleTaskGP as a plain dict."""
    params: dict[str, Any] = {}

    # Observation noise variance
    try:
        params["noise_variance"] = model.likelihood.noise.item()
    except AttributeError:
        pass

    # Output scale and base kernel (handles ScaleKernel wrapping a base kernel)
    covar = model.covar_module
    if hasattr(covar, "outputscale"):
        params["outputscale"] = covar.outputscale.item()
        base_kernel = covar.base_kernel
    else:
        base_kernel = covar

    # Lengthscale: scalar for isotropic, list for ARD
    if hasattr(base_kernel, "lengthscale"):
        ls = base_kernel.lengthscale.detach().squeeze()
        params["lengthscale"] = ls.tolist() if ls.numel() > 1 else ls.item()

    return params


def get_hyperparameters(
    model: SingleTaskGP | ModelListGP,
) -> dict[str, Any]:
    """Extract learned GP hyperparameters as a serialisable dict.

    Handles both :class:`~botorch.models.SingleTaskGP` and
    :class:`~botorch.models.ModelListGP`.  For a ``ModelListGP`` the result
    contains one entry per sub-model keyed ``"model_0"``, ``"model_1"``, etc.

    Extracted fields (where available): ``lengthscale``, ``outputscale``,
    ``noise_variance``.

    Args:
        model: A fitted GP model.

    Returns:
        Dict with scalar or list values; safe to pass to ``json.dumps``.
    """
    if isinstance(model, ModelListGP):
        return {
            f"model_{i}": _single_gp_hyperparameters(m)
            for i, m in enumerate(model.models)
        }
    return _single_gp_hyperparameters(model)