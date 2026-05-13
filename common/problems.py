"""Generic BO problem abstraction.

Central contract for the BO loop:

    obs = problem.observe(X)

The BO loop trains only on ``obs.y_obs``.

Scientific distinction:
- ``y_obs`` is the noisy observation returned to the BO algorithm.
  The algorithm may only use this quantity for model training and acquisition.
- ``f_noiseless`` is the true underlying value, only available in synthetic
  benchmark experiments where the ground truth is known.
  It must never be used by the algorithm itself; only for evaluation metrics
  and diagnostics computed after the fact.
- For real-data problems, ``f_noiseless`` will usually be ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Shape helper
# ---------------------------------------------------------------------------

def ensure_2d_y(y: torch.Tensor) -> torch.Tensor:
    """Convert a 1-D tensor of shape (n,) to (n, 1).

    Leaves tensors already shaped (n, 1) unchanged.

    Args:
        y: Tensor of shape (n,) or (n, 1).

    Returns:
        Tensor of shape (n, 1).

    Raises:
        ValueError: If y has more than 2 dimensions or is already 2-D with a second dimension other than 1.
    """
    if y.dim() == 1:
        return y.unsqueeze(-1)
    if y.dim() == 2 and y.shape[1] == 1:
        return y
    raise ValueError(
        f"ensure_2d_y expects shape (n,) or (n, 1), got {tuple(y.shape)}"
    )


# ---------------------------------------------------------------------------
# Observation Class (Collection of the data)
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Result of one call to ``problem.observe(X)``.

    Attributes:
        X: Queried points, shape (n, dim).
        y_obs: Noisy observations returned to the BO algorithm, shape
            (n, 1).  This is the only quantity the algorithm should
            use for model training and acquisition optimisation.
        f_noiseless: True underlying values, shape (n, 1), or None.
            Available only for synthetic benchmarks where the ground truth is
            known.  Must not be used by the BO algorithm; reserved for
            post-hoc evaluation metrics and diagnostics.
        metadata: Optional dict carrying auxiliary information such as noise
            realisation, measurement id, data source, wall-clock time, etc.
    """

    X: torch.Tensor
    y_obs: torch.Tensor
    f_noiseless: Optional[torch.Tensor] = None
    metadata: Optional[dict[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        self.y_obs = ensure_2d_y(self.y_obs)

        if self.f_noiseless is not None:
            self.f_noiseless = ensure_2d_y(self.f_noiseless)
        if self.y_obs.shape != self.X.shape[:1] + (1,):
            raise ValueError(
                f"y_obs must have shape {self.X.shape[:1] + (1,)}, got {tuple(self.y_obs.shape)} "
                f"for X of shape {tuple(self.X.shape)}"
            )
        if self.f_noiseless is not None and self.f_noiseless.shape != self.y_obs.shape:
            raise ValueError(
                f"f_noiseless shape {tuple(self.f_noiseless.shape)} must match "
                f"y_obs shape {tuple(self.y_obs.shape)}"
            )


# ---------------------------------------------------------------------------
# ProblemSpec abstract base class
# ---------------------------------------------------------------------------

class ProblemSpec(ABC):
    """Abstract base class for all BO problems.

    Subclasses must implement : meth:`observe`.

    For synthetic benchmarks, meth:`has_noiseless` should be overridden to
    return ``True`` and ``observe`` should populate ``f_noiseless``.

    For real-data problems, ``has_noiseless`` returns ``False`` (default) and
    ``observe`` leaves ``f_noiseless`` as ``None``.

    Attributes:
        name: Human-readable problem identifier.
        dim: Input dimension.
        bounds: Box bounds, shape ``(2, dim)``.  ``bounds[0]`` is the lower
            bound and ``bounds[1]`` is the upper bound. (For instance if in dimension 3 : [0,1]*[-1,2]*[1,5] for bounds [0,-1,1]*[1,2,5])
        target: Scalar target value used in target-matching experiments.
        problem_type: String tag, e.g. ``"synthetic"`` or ``"real_data"``.
        default_noise_std: Default observation noise standard deviation, or "None" if noise is not fixed at problem level.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        bounds: torch.Tensor,
        problem_type: str,
        default_noise_std: Optional[float] = None,) -> None:

        if bounds.shape != (2, dim):
            raise ValueError(f"bounds must have shape (2, dim)={(2, dim)}, got {tuple(bounds.shape)}")
        self.name = name
        self.dim = dim
        self.bounds = bounds.to(torch.double)
        self.problem_type = problem_type
        self.default_noise_std = default_noise_std

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    def observe(self, X: torch.Tensor, **kwargs: Any) -> Observation:
        """Query the problem at points ``X`` and return an Observation.
        This public method is shared by all problems. It validates and normalises
        X before delegating the problem-specific observation logic to
        ``_observe_prepared``.
        """
        X = self.prepare_X(X)
        return self._observe_prepared(X, **kwargs)


    @abstractmethod
    def _observe_prepared(self, X: torch.Tensor, **kwargs: Any) -> Observation:
        """Problem-specific observation logic.

        Here X is guaranteed to have shape ``(n, dim)`` and dtype ``torch.double``.
        Concrete subclasses must implement this method.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def prepare_X(self, X: torch.Tensor) -> torch.Tensor:
        """Validate and normalise query points.

        - Promotes a single point of shape ``(dim,)`` to ``(1, dim)``.
        - Casts to ``torch.double``.
        - Checks that the last dimension equals ``self.dim``.

        Args:
            X: Candidate points, shape ``(n, dim)`` or ``(dim,)``.

        Returns:
            Validated tensor of shape ``(n, dim)`` in ``torch.double``.

        Raises:
            ValueError: If the last dimension does not match ``self.dim``.
        """
        X = X.to(torch.double)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.shape[-1] != self.dim:
            raise ValueError(
                f"X last dimension must be {self.dim}, got {X.shape[-1]}"
            )
        return X

    # ------------------------------------------------------------------
    # Overridable helpers
    # ------------------------------------------------------------------

    def has_noiseless(self) -> bool:
        """Return ``True`` if this problem provides ``f_noiseless`` values.

        Synthetic benchmarks should override this to return ``True``.
        Real-data problems return ``False`` (the default).
        """
        return False

    def oracle_target_distance(self, target: float) -> Optional[float]:
        """Return ``min_x |f(x) - target|`` over the problem domain, if known.

        Used by :mod:`common.metrics` to compute the target-matching simple
        regret ``min_t |f(x_t) - T| - min_x |f(x) - T|``.

        - If the target is attainable (lies in the range of ``f``), returns ``0.0``.
        - If the target is not attainable but the analytic minimum distance is
          known, returns it.
        - Returns ``None`` when the oracle value is not analytically available.
          Callers should then either treat the simple regret as
          ``min_t |f(x_t) - T|`` directly or estimate the oracle empirically.

        The default implementation returns ``None``.
        """
        return None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Return a serialisable dict describing the problem configuration."""
        return {
            "name": self.name,
            "dim": self.dim,
            "bounds": self.bounds.tolist(),
            "problem_type": self.problem_type,
            "default_noise_std": self.default_noise_std,
        }
