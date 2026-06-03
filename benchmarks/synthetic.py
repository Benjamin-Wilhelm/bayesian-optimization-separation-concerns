"""Abstract base class for synthetic analytic BO benchmarks.

All synthetic benchmarks inherit from :class:`SyntheticBenchmark`, which in
turn inherits from :class:`~common.problems.ProblemSpec`.

Concrete subclasses must implement :meth:`f`, the noiseless function.

The central observation contract is unchanged::

    obs = benchmark.observe(X)           # y_obs = f(X) + noise
    f_vals = benchmark.f(X)              # noiseless, for diagnostics only
    X_test = benchmark.test_set(n=512)   # held-out Sobol grid
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping, Optional

import torch

from common.problems import Observation, ProblemSpec


# Keys accepted at the top of every benchmark configuration block.
_COMMON_CONFIG_KEYS: frozenset[str] = frozenset({"name", "noise_std", "bounds"})


def _normalise_bounds_config(bounds: Any) -> torch.Tensor:
    """Convert a YAML ``bounds`` entry into a ``(2, dim)`` ``torch.double`` tensor.

    Accepted forms::

        # canonical (tensor-compatible) list-of-lists
        bounds:
          - [lb_1, ..., lb_d]
          - [ub_1, ..., ub_d]

        # equivalent mapping form (legacy)
        bounds:
          lower: [lb_1, ..., lb_d]
          upper: [ub_1, ..., ub_d]

    Full validation (shape, ``lower < upper``) is performed downstream by
    :class:`~common.problems.ProblemSpec`.
    """
    if isinstance(bounds, torch.Tensor):
        return bounds.to(torch.double)
    if isinstance(bounds, Mapping):
        if "lower" not in bounds or "upper" not in bounds:
            raise ValueError(
                "Mapping form of 'bounds' requires both 'lower' and 'upper' keys."
            )
        lower = list(bounds["lower"])
        upper = list(bounds["upper"])
        if len(lower) != len(upper):
            raise ValueError(
                f"'bounds.lower' and 'bounds.upper' must have the same length, "
                f"got {len(lower)} vs {len(upper)}."
            )
        return torch.as_tensor([lower, upper], dtype=torch.double)
    return torch.as_tensor(bounds, dtype=torch.double)


class SyntheticBenchmark(ProblemSpec):
    """Abstract base for synthetic analytic benchmarks.

    Subclasses implement :meth:`f` (noiseless evaluation).
    This class handles noise injection and :class:`~common.problems.Observation`
    construction.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        bounds: torch.Tensor,
        default_noise_std: float = 0.0,
        ) -> None:

        super().__init__(
            name=name,
            dim=dim,
            bounds=bounds,
            problem_type="synthetic",
            default_noise_std=default_noise_std,
        )

    @staticmethod
    def resolve_bounds(
        default_bounds: torch.Tensor,
        override: Optional[Any],
        ) -> torch.Tensor:
        """Return the bounds to use given a benchmark default and an optional override.

        If ``override`` is ``None``, ``default_bounds`` is returned (cast to double).
        Otherwise ``override`` is converted to a ``torch.double`` tensor of shape
        ``(2, dim)`` via :func:`_normalise_bounds_config`.  Full validation (shape,
        ``lower < upper``, dtype) is performed downstream by
        :class:`~common.problems.ProblemSpec`.
        """
        if override is None:
            return default_bounds.to(torch.double)
        return _normalise_bounds_config(override)

    # ------------------------------------------------------------------
    # YAML configuration entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SyntheticBenchmark":
        """Build a benchmark instance from its YAML sub-configuration.

        The default implementation supports the keys :attr:`_COMMON_CONFIG_KEYS`
        (``name``, ``noise_std``, ``bounds``).  Subclasses with additional
        intrinsic parameters (e.g. ``rate`` for :class:`Exponential1D` or
        geometric parameters for :class:`GaussianTrap1D`) must override this
        method and explicitly list the keys they accept.

        Unknown keys raise :class:`ValueError` so that typos and misplaced
        keys (e.g. ``beta`` accidentally added under ``benchmark:``) cannot be
        silently ignored.

        The ``name`` key is informational only — dispatch is performed by the
        caller (see :func:`benchmarks.build_benchmark`).  When present, it is
        ignored at this level.
        """
        cls._check_allowed_keys(config, _COMMON_CONFIG_KEYS)
        kwargs = cls._extract_common_kwargs(config)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Helpers for subclasses' from_config implementations
    # ------------------------------------------------------------------

    @classmethod
    def _check_allowed_keys(
        cls,
        config: Mapping[str, Any],
        allowed: frozenset[str],
        ) -> None:
        unknown = set(config) - set(allowed)
        if unknown:
            raise ValueError(
                f"Unknown keys in benchmark config for {cls.__name__}: "
                f"{sorted(unknown)}. Allowed keys: {sorted(allowed)}."
            )

    @staticmethod
    def _extract_common_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
        """Pick the common ``noise_std`` and ``bounds`` kwargs from ``config``.

        ``bounds`` is normalised via :func:`_normalise_bounds_config`; ``None``
        values are dropped so that the benchmark constructor falls back to its
        default.
        """
        kwargs: dict[str, Any] = {}
        if "noise_std" in config and config["noise_std"] is not None:
            kwargs["noise_std"] = float(config["noise_std"])
        if config.get("bounds") is not None:
            kwargs["bounds"] = _normalise_bounds_config(config["bounds"])
        return kwargs

    # ------------------------------------------------------------------
    # Noiseless function (subclasses must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def f(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the noiseless benchmark function.

        Args:
            X: Tensor of shape ``(n, dim)``, already validated double.

        Returns:
            Tensor of shape ``(n, 1)`` with noiseless function values.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # ProblemSpec interface
    # ------------------------------------------------------------------

    def has_noiseless(self) -> bool:
        """Synthetic benchmarks always provide ``f_noiseless``."""
        return True

    def _observe_prepared(
        self,
        X: torch.Tensor,
        noise_std: Optional[float] = None,
        **kwargs: Any,
        ) -> Observation:
        """Add i.i.d. Gaussian noise on top of the noiseless function.
        
        y_obs = f(X) + epsilon,
        epsilon ~ N(0, sigma^2 I)
        Args:
            X: Tensor of shape ``(n, dim)``, validated double.
            noise_std: Noise standard deviation.  Overrides
                ``self.default_noise_std`` when provided.

        Returns:
            :class:`~common.problems.Observation` with ``f_noiseless`` set.
        """
        std = noise_std if noise_std is not None else (self.default_noise_std or 0.0)
        f_vals = self.f(X)
        noise = torch.randn_like(f_vals) * std
        y_obs = f_vals + noise
        return Observation(X=X, y_obs=y_obs, f_noiseless=f_vals)

    # ------------------------------------------------------------------
    # Diagnostics helper
    # ------------------------------------------------------------------

    def test_set(self, n: int, seed: int = 0) -> torch.Tensor:
        """Generate a deterministic quasi-uniform test grid for GP diagnostics.

        Uses a scrambled Sobol sequence so coverage is quasi-uniform.  The
        caller is responsible for evaluating the benchmark at these points::

            X_test = benchmark.test_set(n=512)
            f_test = benchmark.f(X_test)          # noiseless ground truth
            obs    = benchmark.observe(X_test)    # noisy observations

        Args:
            n: Number of test points.
            seed: Sobol scramble seed for reproducibility.

        Returns:
            Tensor of shape ``(n, dim)`` in the problem's native bounds.
        """
        sobol = torch.quasirandom.SobolEngine(
            dimension=self.dim, scramble=True, seed=seed
        )
        X_unit = sobol.draw(n).to(torch.double)
        lb, ub = self.bounds[0], self.bounds[1]
        return lb + (ub - lb) * X_unit

    # ------------------------------------------------------------------
    # Near-target grid
    # ------------------------------------------------------------------

    # Generic rejection-sampling parameters.  Subclasses may override
    # ``_NEAR_TARGET_BATCH`` for benchmarks where small batches are too
    # inefficient (e.g. tight epsilon in high dimension).
    _NEAR_TARGET_BATCH: int = 4096
    _NEAR_TARGET_MAX_CANDIDATES: int = 4_000_000

    def near_target_test_set(
        self,
        *,
        target: float,
        epsilon: float,
        n: int,
        seed: int,
    ) -> torch.Tensor:
        r"""Generate exactly ``n`` test points satisfying ``|f(x) - T| <= eps``.

        Strict contract::

            X_near = problem.near_target_test_set(target=T, epsilon=eps,
                                                  n=n, seed=s)
            assert X_near.shape == (n, problem.dim)
            assert torch.all(torch.abs(problem.f(X_near).squeeze(-1) - T) <= eps)

        The generic implementation runs deterministic rejection sampling on
        a single continuous Sobol stream over ``self.bounds``: draw a batch,
        evaluate ``self.f``, keep candidates inside the near-target band,
        repeat until ``n`` valid points have been accumulated.

        Subclasses may override this method when rejection on the unit cube
        is too inefficient (e.g. :class:`Hartmann6D` with a tight ``epsilon``
        around a deep target) — in that case a local-perturbation strategy
        around level-set anchors is typically faster.

        Args:
            target: Target value ``T``.
            epsilon: Strict near-target tolerance; must be positive.
            n: Required number of returned points.
            seed: Sobol scramble seed; fully determines the output.

        Returns:
            Tensor of shape ``(n, self.dim)`` of dtype ``torch.double``.

        Raises:
            ValueError: ``n < 0`` or ``epsilon <= 0``.
            RuntimeError: Safety cap on the number of Sobol candidates
                (``_NEAR_TARGET_MAX_CANDIDATES``) was reached without
                collecting ``n`` valid points.  Typically signals a target
                outside the range of ``f`` rather than a transient pool
                shortage; callers should re-validate the configuration or
                provide a benchmark-specific override.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}.")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}.")
        if n == 0:
            return torch.empty((0, self.dim), dtype=torch.double)

        sobol = torch.quasirandom.SobolEngine(
            dimension=self.dim, scramble=True, seed=seed
        )
        lb, ub = self.bounds[0], self.bounds[1]

        collected: list[torch.Tensor] = []
        n_collected = 0
        n_drawn = 0
        batch = max(int(self._NEAR_TARGET_BATCH), 1)
        max_drawn = max(int(self._NEAR_TARGET_MAX_CANDIDATES), batch)

        while n_collected < n:
            X_unit = sobol.draw(batch).to(torch.double)
            X = lb + (ub - lb) * X_unit
            f_vals = self.f(X).squeeze(-1)
            mask = (f_vals - target).abs() <= epsilon
            if mask.any():
                accepted = X[mask]
                collected.append(accepted)
                n_collected += int(accepted.shape[0])
            n_drawn += batch
            if n_drawn >= max_drawn and n_collected < n:
                raise RuntimeError(
                    f"near_target_test_set: drew {n_drawn} Sobol candidates "
                    f"without collecting {n} points satisfying "
                    f"|f - {target}| <= {epsilon} on '{self.name}'. "
                    "This usually means the target lies outside the range "
                    "of f or the band is too thin for the generic rejection "
                    "sampler; consider validating the configuration or "
                    "providing a benchmark-specific override."
                )

        pool = torch.cat(collected, dim=0)
        return pool[:n]
