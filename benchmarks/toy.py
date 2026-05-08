"""Concrete synthetic BO benchmarks.

Implements four standard benchmark functions:

- :class:`SineTrend1D`  — Pedagogical 1-D function for interpretability.
- :class:`Forrester1D`  — Forrester et al. (2008) multi-modal 1-D stress test.
- :class:`Branin2D`     — Classic 2-D Branin–Hoo function.
- :class:`Hartmann6D`   — 6-D Hartmann function.

Default noise levels are chosen so that the signal-to-noise ratio is
meaningful for target-matching experiments.
"""

from __future__ import annotations

import math

import torch

from benchmarks.synthetic import SyntheticBenchmark


# ---------------------------------------------------------------------------
# SineTrend 1-D  (pedagogical)
# ---------------------------------------------------------------------------

DEFAULT_NOISE_STD_SINE_TREND1D: float = 0.05


class SineTrend1D(SyntheticBenchmark):
    """Pedagogical 1-D benchmark for interpretability and visualisation.

        f(x) = sin(6x) + 0.3x, x in [0, 1].

    Smooth and unimodal enough to read easily in a figure, yet non-trivial
    due to the sinusoidal oscillation on top of a linear trend.
    Range approximately [-0.77, 1.08].

    Args:
        noise_std: Observation noise standard deviation.
    """

    def __init__(self, noise_std: float = DEFAULT_NOISE_STD_SINE_TREND1D) -> None:
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="sine_trend_1d",
            dim=1,
            bounds=bounds,
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:, 0]
        return (torch.sin(6.0 * x) + 0.3 * x).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Forrester 1-D  (BO stress test)
# ---------------------------------------------------------------------------

DEFAULT_NOISE_STD_FORRESTER1D: float = 0.5


class Forrester1D(SyntheticBenchmark):
    """Forrester et al. (2008) 1-D BO stress-test function.

    .. math::
        f(x) = (6x - 2)^2 * sin(12x - 4),  x in [0, 1].

    Forrester 1D benchmark function.

    - Multi-modal function on [0, 1]
    - Local minimum near x ≈ 0.1
    - Global minimum at x* ≈ 0.757 with f* ≈ -6.02
    - Function range approximately [-6.02, 15.8]

    Args:
        noise_std: Observation noise standard deviation.
    """

    def __init__(self, noise_std: float = DEFAULT_NOISE_STD_FORRESTER1D) -> None:
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="forrester_1d",
            dim=1,
            bounds=bounds,
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:, 0]
        return ((6.0 * x - 2.0).pow(2) * torch.sin(12.0 * x - 4.0)).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Branin 2-D
# ---------------------------------------------------------------------------

DEFAULT_NOISE_STD_BRANIN: float = 2.0


class Branin2D(SyntheticBenchmark):
    """Classic Branin–Hoo benchmark function.

    - 2D multi-modal function with three identical global minima
    - Global minimum value: f* ≈ 0.397887
    - Domain:
        x1 ∈ [-5, 10]
        x2 ∈ [0, 15]

    Definition:
        f(x1, x2) = a (x2 - b x1^2 + c x1 - r)^2
                    + s (1 - t) cos(x1) + s

    Parameters:
        a = 1
        b = 5.1 / (4π^2)
        c = 5 / π
        r = 6
        s = 10
        t = 1 / (8π)

    Args:
        noise_std: Observation noise standard deviation.
    """

    _a: float = 1.0
    _b: float = 5.1 / (4.0 * math.pi ** 2)
    _c: float = 5.0 / math.pi
    _r: float = 6.0
    _s: float = 10.0
    _t: float = 1.0 / (8.0 * math.pi)

    def __init__(self, noise_std: float = DEFAULT_NOISE_STD_BRANIN) -> None:
        bounds = torch.tensor(
            [[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double
        )
        super().__init__(
            name="branin_2d",
            dim=2,
            bounds=bounds,
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        term1 = self._a * (x2 - self._b * x1.pow(2) + self._c * x1 - self._r).pow(2)
        term2 = self._s * (1.0 - self._t) * torch.cos(x1)
        return (term1 + term2 + self._s).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Hartmann 6-D
# ---------------------------------------------------------------------------

DEFAULT_NOISE_STD_HARTMANN6: float = 0.05

# Module-level constants to avoid re-allocation on each forward call.
_H6_ALPHA = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.double)

_H6_A = torch.tensor(
    [
        [10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
        [ 0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
        [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
        [17.0,  8.0,  0.05, 10.0,  0.1, 14.0],
    ],
    dtype=torch.double,
)

_H6_P = 1e-4 * torch.tensor(
    [
        [1312.0, 1696.0, 5569.0,  124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0,  381.0],
    ],
    dtype=torch.double,
)


class Hartmann6D(SyntheticBenchmark):
    """6-D Hartmann benchmark function.

    - 6-dimensional multi-modal function on [0, 1]^6
    - Highly non-convex with several local minima
    - Global minimum value: f* ≈ -3.32237
    - Global minimizer approximately:
        x* ≈ (0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573)

    Definition:
        f(x) = - sum_{i=1}^4 alpha_i
                exp( - sum_{j=1}^6 A_ij (x_j - P_ij)^2 )

    Args:
        noise_std: Observation noise standard deviation.
    """

    def __init__(self, noise_std: float = DEFAULT_NOISE_STD_HARTMANN6) -> None:
        bounds = torch.zeros(2, 6, dtype=torch.double)
        bounds[1] = 1.0
        super().__init__(
            name="hartmann_6d",
            dim=6,
            bounds=bounds,
            default_noise_std=noise_std,)

    def f(self, X: torch.Tensor) -> torch.Tensor:
        # Move constants to the same device as X.
        alpha = _H6_ALPHA.to(X.device)
        A = _H6_A.to(X.device)
        P = _H6_P.to(X.device)

        # X: (n, 6)  ->  diff: (n, 4, 6)
        diff = X.unsqueeze(1) - P.unsqueeze(0)
        # inner: (n, 4)
        inner = (A.unsqueeze(0) * diff.pow(2)).sum(dim=-1)
        # result: (n,)
        result = -(alpha.unsqueeze(0) * torch.exp(-inner)).sum(dim=-1)
        return result.unsqueeze(-1)
