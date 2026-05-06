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
from typing import Any, Optional

import torch

from common.problems import Observation, ProblemSpec


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
