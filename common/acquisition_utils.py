"""Generic acquisition function optimisation utilities.

Provides thin wrappers around BoTorch's ``optimize_acqf`` that work with
*any* :class:`~botorch.acquisition.AcquisitionFunction`.

No experiment-specific or task-specific logic lives here.  All objective
structure (targets, constraints, costs, priors) belongs in the
``acquisitions/`` directory.

Usage::

    from botorch.acquisition import UpperConfidenceBound
    from common.acquisition_utils import optimize_acquisition

    acq = UpperConfidenceBound(model, beta=2.0)
    candidates, acq_value = optimize_acquisition(acq, bounds, q=1)
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_NUM_RESTARTS: int = 10
_DEFAULT_RAW_SAMPLES: int = 256


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def optimize_acquisition(
    acq_fn: AcquisitionFunction,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = _DEFAULT_NUM_RESTARTS,
    raw_samples: int = _DEFAULT_RAW_SAMPLES,
    fixed_features: Optional[dict[int, float]] = None,
    options: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise an acquisition function over a box-constrained domain.

    Wraps :func:`botorch.optim.optimize_acqf` with sensible defaults and
    returns detached tensors suitable for downstream use without keeping
    the computational graph alive.

    Args:
        acq_fn: Any BoTorch acquisition function.  Must be in ``eval`` mode;
            the model it wraps should already be fitted.
        bounds: Box constraints, shape ``(2, d)``.  ``bounds[0]`` are lower
            bounds and ``bounds[1]`` are upper bounds.
        q: Number of candidates to return (batch size).  Use ``q=1`` for
            sequential BO.
        num_restarts: Number of random restarts for the L-BFGS-B optimiser.
            Higher values reduce the chance of converging to a poor local
            optimum at the cost of more compute.
        raw_samples: Number of quasi-random samples used to initialise the
            multi-start optimisation.
        fixed_features: Optional dict mapping dimension indices to fixed
            values, forwarded to
            :func:`~botorch.optim.optimize_acqf_mixed` semantics.  Currently
            passed directly to ``optimize_acqf``; if ``None``, all dimensions
            are free.
        options: Optional dict of L-BFGS-B solver options, e.g.
            ``{"maxiter": 200}``.

    Returns:
        Tuple ``(candidates, acq_value)`` where:

        - ``candidates``: Detached tensor of shape ``(q, d)`` containing the
          optimised candidate points.
        - ``acq_value``: Detached scalar tensor, the acquisition value at the
          returned candidates.

    Raises:
        ValueError: If ``bounds`` does not have shape ``(2, d)`` or if ``q``
            is not a positive integer.

    Example::

        candidates, value = optimize_acquisition(
            acq_fn=ucb,
            bounds=problem.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
        next_x = candidates  # shape (1, d)
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(
            f"bounds must have shape (2, d), got {tuple(bounds.shape)}"
        )
    if q < 1:
        raise ValueError(f"q must be a positive integer, got {q}")

    bounds = bounds.to(torch.double)

    candidates, acq_value = optimize_acqf(
        acq_function=acq_fn,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features=fixed_features,
        options=options or {},
    )

    return candidates.detach(), acq_value.detach()


# ---------------------------------------------------------------------------
# Sequential convenience wrapper
# ---------------------------------------------------------------------------

def suggest_next_point(
    acq_fn: AcquisitionFunction,
    bounds: torch.Tensor,
    num_restarts: int = _DEFAULT_NUM_RESTARTS,
    raw_samples: int = _DEFAULT_RAW_SAMPLES,
    options: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the single best next candidate under ``acq_fn``.

    Convenience wrapper around :func:`optimize_acquisition` for the common
    sequential BO case where ``q=1``.

    Args:
        acq_fn: Any BoTorch acquisition function.
        bounds: Box constraints, shape ``(2, d)``.
        num_restarts: Number of random restarts.
        raw_samples: Number of raw samples for initialisation.
        options: Optional L-BFGS-B solver options.

    Returns:
        Tuple ``(x_next, acq_value)`` where:

        - ``x_next``: Detached tensor of shape ``(1, d)``.
        - ``acq_value``: Detached scalar tensor.
    """
    return optimize_acquisition(
        acq_fn=acq_fn,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )


# ---------------------------------------------------------------------------
# Numerical safeguard
# ---------------------------------------------------------------------------

def clamp_variance(
    variance: torch.Tensor,
    min_variance: float = 1e-12,
    ) -> torch.Tensor:
    """Clamp predictive variance to a positive lower bound.

    Posterior variance can become numerically negative due to floating-point
    cancellation in the Cholesky solve.  Clamping prevents downstream
    issues such as ``sqrt`` of a negative number in UCB-style acquisitions.

    Args:
        variance: Predictive variance tensor, any shape.
        min_variance: Minimum allowed value.  Defaults to ``1e-6``.

    Returns:
        Clamped variance tensor, same shape as input.
    """
    return variance.clamp(min=min_variance)


# ---------------------------------------------------------------------------
# UCB exploration schedule (beta_t)
# ---------------------------------------------------------------------------

class BetaSchedule:
    r"""Iteration-dependent UCB exploration parameter ``beta_t``.

    UCB-style acquisitions score ``a_t(x) = mu(x) + sqrt(beta_t) * sigma(x)``.
    This object returns ``beta_t`` (**not** ``sqrt(beta_t)``) as a function of
    the 1-based BO iteration ``t`` and the problem dimension ``d``, so a run
    can switch from a constant exploration weight to a growing schedule purely
    from the YAML config.

    Generic and task-agnostic: shared by Experiment A and Experiment B (any
    UCB acquisition that takes a scalar ``beta``).

    Config resolution (:meth:`from_config`):

    - ``beta_schedule:`` present  -> use it (takes priority over ``beta``);
    - else ``beta:`` present      -> constant schedule reproducing the scalar;
    - else                        -> ``ValueError``.

    Supported ``beta_schedule.type`` values:

    ``constant``
        ``beta_t = value``.  Exactly reproduces the legacy scalar ``beta``.

    ``log``
        ``beta_t = beta0 + c * log(t + 1)``.

    ``matern_gamma``
        GP-UCB / Matern-inspired theoretical schedule.  The maximum
        information gain is approximated by

        .. math::

            \gamma_t^{\mathrm{proxy}} = \mathrm{scale}\cdot (t+1)^{p}\,
                \log(t+1), \qquad p = \frac{d(d+1)}{2\nu + d(d+1)},

        and the exploration parameter is

        .. math::

            \sqrt{\beta_t} = B + R\sqrt{2\big(\gamma_{t-1}^{\mathrm{proxy}}
                + 1 + \log(1/\delta)\big)}, \qquad \beta_t = (\sqrt{\beta_t})^2.

        ``gamma`` is evaluated one iteration back (``t - 1``), matching the
        "information gained up to the previous round" convention; at ``t = 1``
        it is ``0``.  ``scale`` rescales ``gamma^proxy`` without changing the
        functional form (e.g. ``0.1`` to soften the conservative bound,
        ``1.0`` to keep it).
    """

    _VALID_TYPES = ("constant", "log", "matern_gamma")

    def __init__(self, kind: str, params: Mapping[str, float]) -> None:
        if kind not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown beta_schedule type '{kind}'.  Known: {self._VALID_TYPES}."
            )
        self.kind = kind
        self.params = {k: float(v) for k, v in params.items()}
        self._validate()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BetaSchedule":
        """Build from a full run config (``beta_schedule`` wins over ``beta``)."""
        sched = config.get("beta_schedule")
        if sched is not None:
            return cls.from_schedule_dict(sched)
        beta = config.get("beta")
        if beta is not None:
            return cls("constant", {"value": float(beta)})
        raise ValueError(
            "config must define either 'beta' (scalar) or 'beta_schedule'."
        )

    @classmethod
    def from_schedule_dict(cls, d: Mapping[str, Any]) -> "BetaSchedule":
        """Build from the ``beta_schedule`` sub-config alone.

        Also accepts the dict produced by :meth:`to_dict` (round-trip), so a
        schedule stored in metadata reconstructs identically.
        """
        kind = d.get("type")
        if kind == "constant":
            if d.get("value") is None:
                raise ValueError("beta_schedule 'constant' requires 'value'.")
            params: dict[str, float] = {"value": float(d["value"])}
        elif kind == "log":
            if d.get("beta0") is None:
                raise ValueError("beta_schedule 'log' requires 'beta0'.")
            params = {"beta0": float(d["beta0"]), "c": float(d.get("c", 1.0))}
        elif kind == "matern_gamma":
            params = {
                "B":     float(d.get("B", 1.0)),
                "R":     float(d.get("R", 1.0)),
                "delta": float(d.get("delta", 0.1)),
                "nu":    float(d.get("nu", 2.5)),
                "scale": float(d.get("scale", 1.0)),
            }
        else:
            raise ValueError(
                f"Unknown beta_schedule type '{kind}'.  Known: {cls._VALID_TYPES}."
            )
        return cls(kind, params)

    def _validate(self) -> None:
        """Ensure the schedule yields ``beta_t >= 0`` (required by UCB)."""
        p = self.params
        if self.kind == "constant":
            if p["value"] < 0:
                raise ValueError(f"beta 'value' must be >= 0, got {p['value']}.")
        elif self.kind == "log":
            if p["beta0"] < 0:
                raise ValueError(
                    f"beta_schedule 'beta0' must be >= 0, got {p['beta0']}."
                )
        elif self.kind == "matern_gamma":
            if p["B"] < 0 or p["R"] < 0:
                raise ValueError("beta_schedule 'B' and 'R' must be >= 0.")
            if not 0.0 < p["delta"] < 1.0:
                raise ValueError(
                    f"beta_schedule 'delta' must be in (0, 1), got {p['delta']}."
                )
            if p["nu"] <= 0:
                raise ValueError(f"beta_schedule 'nu' must be > 0, got {p['nu']}.")
            if p["scale"] < 0:
                raise ValueError(
                    f"beta_schedule 'scale' must be >= 0, got {p['scale']}."
                )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _gamma_proxy(self, s: int, dim: int) -> float:
        """Information-gain proxy ``scale * (s+1)^p * log(s+1)`` (``0`` at s=0)."""
        p = self.params
        exponent = (dim * (dim + 1)) / (2.0 * p["nu"] + dim * (dim + 1))
        return p["scale"] * ((s + 1) ** exponent) * math.log(s + 1)

    def beta_at(self, t: int, dim: int) -> float:
        """Return ``beta_t`` at 1-based BO iteration ``t`` (dimension ``dim``).

        ``dim`` is only used by the ``matern_gamma`` schedule; the others
        ignore it.
        """
        if t < 1:
            raise ValueError(f"t must be a 1-based iteration >= 1, got {t}.")
        if self.kind == "constant":
            return self.params["value"]
        if self.kind == "log":
            return self.params["beta0"] + self.params["c"] * math.log(t + 1)
        # matern_gamma: gamma evaluated one step back (gamma_proxy(0) = 0).
        p = self.params
        gamma_prev = self._gamma_proxy(t - 1, dim)
        sqrt_beta = p["B"] + p["R"] * math.sqrt(
            2.0 * (gamma_prev + 1.0 + math.log(1.0 / p["delta"]))
        )
        return sqrt_beta ** 2

    def is_constant(self) -> bool:
        """Whether the schedule is a plain constant (legacy ``beta`` behavior)."""
        return self.kind == "constant"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Config-shaped dict ``{"type": ..., **params}`` for metadata storage."""
        return {"type": self.kind, **self.params}

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"BetaSchedule(type={self.kind}, {inner})"
