"""Shared GP / acquisition step logic for Experiment C.

Both runners (``run_fixed_scalarization`` and ``run_pareto_front``) build GPs
and optimize acquisitions the same way; only *where* the scalarization is
applied differs (RIGHT scalarizes the acquisition vector, WRONG scalarizes the
observations).  This module centralises that shared logic and the RIGHT/WRONG
pairing hash.

RIGHT model
    One generic single-output GP per objective (a
    :class:`~botorch.models.ModelListGP`), trained on the raw **oriented**
    objectives ``F``.  The preference lambda never enters the targets.

WRONG model
    A single scalar GP trained on ``g_lambda(x) = rho_lambda(F(x))`` for a
    *fixed* lambda (used only in ``fixed_scalarization`` and
    ``wrong_algorithm_2``, where lambda is constant within a GP).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import torch
from botorch.models import ModelListGP, SingleTaskGP

from acquisitions.prior_guided import VanillaUCB
from common.acquisition_utils import optimize_acquisition
from common.models import GPModelBuilder, GPModelConfig
from acquisitions.multiobjective_acquisitions import RightScalarizedUCB
from common.scalarizations import scalarize

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# GP builder
# ---------------------------------------------------------------------------

def make_gp_builder(config: dict[str, Any], noise_std: float) -> GPModelBuilder:
    """Build the shared :class:`GPModelBuilder` for one run.

    Noise policy: a strictly positive ``noise.std`` with ``use_fixed_noise``
    (default) yields a ``FixedNoiseGP`` with known variance; ``noise.std == 0``
    uses an inferred-noise ``SingleTaskGP`` (avoids a degenerate zero-variance
    ``FixedNoiseGP``).  Input normalization and outcome standardization are
    internal GP transforms that never touch the scientific objective.
    """
    model_cfg = config.get("model", {}) or {}
    use_fixed_noise = bool(model_cfg.get("use_fixed_noise", True))
    gp_noise: Optional[float]
    if noise_std and noise_std > 0.0 and use_fixed_noise:
        gp_noise = float(noise_std)
    else:
        gp_noise = None
    gp_config = GPModelConfig(
        noise_std=gp_noise,
        normalize_input=bool(model_cfg.get("normalize_input", True)),
        standardize_output=bool(model_cfg.get("standardize_output", True)),
        kernel=str(model_cfg.get("kernel", "matern52")),
    )
    return GPModelBuilder(gp_config)


# ---------------------------------------------------------------------------
# RIGHT / WRONG model fitting
# ---------------------------------------------------------------------------

def fit_right_models(
    builder: GPModelBuilder,
    X: Tensor,
    oriented_Y: Tensor,
) -> ModelListGP:
    """Fit one generic GP per objective; return a :class:`ModelListGP`."""
    m = oriented_Y.shape[-1]
    train_Xs = [X for _ in range(m)]
    train_Ys = [oriented_Y[:, j : j + 1] for j in range(m)]
    return builder.fit_model_list(train_Xs, train_Ys)


def fit_wrong_model(
    builder: GPModelBuilder,
    X: Tensor,
    g_values: Tensor,
) -> SingleTaskGP:
    """Fit the scalar WRONG GP on the transformed targets ``g_lambda(F(x))``."""
    return builder.fit(X, g_values.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Acquisition steps
# ---------------------------------------------------------------------------

def right_next_point(
    model: ModelListGP,
    lambdas: Tensor,
    beta: float,
    scalarization: str,
    reference_point: Optional[Tensor],
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
) -> tuple[Tensor, Tensor]:
    """Optimize the RIGHT scalarized UCB; return ``(x_next (1, d), acq_value)``."""
    acq = RightScalarizedUCB(
        model=model,
        lambdas=lambdas,
        beta=beta,
        scalarization=scalarization,
        reference_point=reference_point,
    )
    return optimize_acquisition(
        acq, bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples
    )


def wrong_next_point(
    model: SingleTaskGP,
    beta: float,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
) -> tuple[Tensor, Tensor]:
    """Optimize the WRONG scalar UCB; return ``(x_next (1, d), acq_value)``."""
    acq = VanillaUCB(model=model, beta=beta)
    return optimize_acquisition(
        acq, bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples
    )


def wrong_targets(
    oriented_Y: Tensor,
    lambdas: Tensor,
    scalarization: str,
    reference_point: Optional[Tensor],
) -> Tensor:
    """Transformed WRONG targets ``g_lambda(F(x)) = rho_lambda(F(x))``, ``(n, 1)``."""
    g = scalarize(oriented_Y, lambdas, reference_point, scalarization)
    return g.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Pairing hash
# ---------------------------------------------------------------------------

def pairing_hash(lambdas: Tensor, X_init: Tensor, ndigits: int = 10) -> str:
    """Deterministic hash of a preference + initial design, for pairing checks.

    RIGHT and WRONG that must be paired share the same lambda and initial
    design, hence the same hash.  Rounding makes the hash robust to trivial
    float formatting differences while still catching genuine mismatches.
    """
    payload = {
        "lambda": [round(float(v), ndigits) for v in lambdas.reshape(-1).tolist()],
        "X_init": [
            [round(float(v), ndigits) for v in row]
            for row in X_init.reshape(X_init.shape[0], -1).tolist()
        ],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = [
    "make_gp_builder",
    "fit_right_models",
    "fit_wrong_model",
    "right_next_point",
    "wrong_next_point",
    "wrong_targets",
    "pairing_hash",
]
