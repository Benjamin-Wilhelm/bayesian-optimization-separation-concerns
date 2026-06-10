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
from typing import Any, Mapping, Optional

import torch

from benchmarks.synthetic import SyntheticBenchmark, _normalise_bounds_config

class Linear1D(SyntheticBenchmark):
    """One-dimensional linear monotone benchmark.

    This is the simplest possible target-matching benchmark.

    The noiseless function is

        f(x) = x,   x in [0, 1].

    For any target T in [0, 1], the target set {x : f(x) = T}
    contains exactly one point, namely x* = T.

    This benchmark is useful because the raw objective is perfectly smooth
    and globally bijective, while the transformed target-matching objective

        g(x) = |f(x) - T| = |x - T|

    has a non-smooth cusp at the target point.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        default_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="linear_1d",
            dim=1,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate f(x) = x.

        Args:
            X: Tensor of shape (..., 1), with values in [0, 1].

        Returns:
            Tensor of shape (..., 1).
        """
        return X[..., :1]
    

class Exponential1D(SyntheticBenchmark):
    """One-dimensional normalized exponential monotone benchmark.

    The noiseless function is

        f(x) = (exp(a x) - 1) / (exp(a) - 1),   x in [0, 1],

    where a > 0 controls the curvature.

    The function is smooth, strictly increasing, and maps [0, 1] to [0, 1].
    Therefore, for every target T in [0, 1], there is a unique target point.

    This benchmark is slightly richer than Linear1D because the raw objective
    is nonlinear while remaining geometrically simple. The transformed
    objective

        g(x) = |f(x) - T|

    still introduces a cusp at the unique target point.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        rate: float = 3.0,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}.")

        self.rate = float(rate)

        default_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="exponential_1d",
            dim=1,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the normalized exponential benchmark.

        Args:
            X: Tensor of shape (..., 1), with values in [0, 1].

        Returns:
            Tensor of shape (..., 1).
        """
        a = torch.as_tensor(self.rate, dtype=X.dtype, device=X.device)
        numerator = torch.exp(a * X[..., :1]) - 1.0
        denominator = torch.exp(a) - 1.0
        return numerator / denominator

    _CONFIG_KEYS: frozenset[str] = frozenset({"name", "noise_std", "bounds", "rate"})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Exponential1D":
        cls._check_allowed_keys(config, cls._CONFIG_KEYS)
        kwargs = cls._extract_common_kwargs(config)
        if "rate" in config and config["rate"] is not None:
            kwargs["rate"] = float(config["rate"])
        return cls(**kwargs)


class GaussianTrap1D(SyntheticBenchmark):
    """One-dimensional target-matching benchmark with a local near-target trap.

    The noiseless function is

        f(x) = x - A exp(-(x - c)^2 / (2 s^2))

    on a configurable one-dimensional domain.

    Parameters are expressed through the intended trap geometry:

        trap_location:
            Position of the local near-target minimum of |f(x) - T|.

        width:
            Width s of the Gaussian dip.

        shape:
            Value u > 1 controlling the shape of the dip.

        trap_gap:
            Positive local distance |f(x_trap) - T| at the false minimum.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        trap_location: float = 0.40,
        width: float = 0.08,
        shape: float = 2.0,
        trap_gap: float = 0.02,
        bounds: torch.Tensor | None = None,
    ) -> None:
        if width <= 0:
            raise ValueError("width must be strictly positive.")

        if shape <= 1:
            raise ValueError(
                "shape must be greater than 1 to define the left local maximum."
            )

        if trap_gap <= 0:
            raise ValueError("trap_gap must be strictly positive.")

        self.trap_location = float(trap_location)
        self.width = float(width)
        self.shape = float(shape)
        self.trap_gap = float(trap_gap)

        self.amplitude = (
            self.width
            * math.exp(self.shape**2 / 2.0)
            / self.shape
        )

        self.center = (
            self.trap_location
            + self.width * self.shape
        )

        self.target = (
            self.trap_location
            - self.width / self.shape
            + self.trap_gap
        )

        default_bounds = torch.tensor(
            [[0.0], [1.0]],
            dtype=torch.double,
        )

        resolved_bounds = SyntheticBenchmark.resolve_bounds(default_bounds, bounds)

        if not (
            resolved_bounds[0, 0].item() <= self.trap_location
            <= resolved_bounds[1, 0].item()
        ):
            raise ValueError("trap_location must lie inside the domain.")

        if not (
            resolved_bounds[0, 0].item() <= self.center
            <= resolved_bounds[1, 0].item()
        ):
            raise ValueError("Gaussian center must lie inside the domain.")

        super().__init__(
            name="gaussian_trap_1d",
            dim=1,
            bounds=resolved_bounds,
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0]

        return x - self.amplitude * torch.exp(
            -((x - self.center) ** 2)
            / (2.0 * self.width**2)
        )

    _CONFIG_KEYS: frozenset[str] = frozenset({
        "name",
        "noise_std",
        "bounds",
        "trap_location",
        "width",
        "shape",
        "trap_gap",
    })

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GaussianTrap1D":
        cls._check_allowed_keys(config, cls._CONFIG_KEYS)
        kwargs = cls._extract_common_kwargs(config)
        for key in ("trap_location", "width", "shape", "trap_gap"):
            if key in config and config[key] is not None:
                kwargs[key] = float(config[key])
        return cls(**kwargs)


class DampedOscillatoryTrap1D(SyntheticBenchmark):
    """One-dimensional damped oscillatory benchmark.

    The noiseless function is

        f(x) = x + a exp(-lambda x) sin(2 pi m x),   x in [0, 1].

    The linear trend gives a global direction, while the damped oscillation
    creates local near-target regions. For suitable targets, the transformed
    objective

        g(x) = |f(x) - T|

    has several local valleys, which can mislead a GP fitted directly on g.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        self.amplitude = 0.22
        self.decay = 2.5
        self.frequency = 5.0

        default_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="damped_oscillatory_trap_1d",
            dim=1,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., :1]

        a = torch.as_tensor(self.amplitude, dtype=X.dtype, device=X.device)
        decay = torch.as_tensor(self.decay, dtype=X.dtype, device=X.device)
        freq = torch.as_tensor(self.frequency, dtype=X.dtype, device=X.device)
        two_pi = torch.as_tensor(2.0 * torch.pi, dtype=X.dtype, device=X.device)

        oscillation = a * torch.exp(-decay * x) * torch.sin(two_pi * freq * x)
        return x + oscillation
# ---------------------------------------------------------------------------
# SineTrend 1-D  (pedagogical)
# ---------------------------------------------------------------------------

DEFAULT_NOISE_STD_SINE_TREND1D: float = 0.05

# Numerical extrema of f(x) = sin(6x) + 0.3x on [0, 1], computed on a
# dense grid (200_001 points).  Used by :meth:`SineTrend1D.oracle_target_distance`
# to decide whether a target is attainable and, if not, return the closest
# achievable target distance.
_SINE_TREND1D_F_MIN: float = -0.7656308116
_SINE_TREND1D_F_MAX: float =  1.0797900769

class SineTrend1D(SyntheticBenchmark):
    """Pedagogical 1-D benchmark for interpretability and visualisation.

        f(x) = sin(6x) + 0.3x, x in [0, 1].

    Smooth and unimodal enough to read easily in a figure, yet non-trivial
    due to the sinusoidal oscillation on top of a linear trend.
    Range approximately [-0.766, 1.080].

    Args:
        noise_std: Observation noise standard deviation.
    """

    def __init__(
        self,
        noise_std: float = DEFAULT_NOISE_STD_SINE_TREND1D,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        default_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="sine_trend_1d",
            dim=1,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:, 0]
        return (torch.sin(6.0 * x) + 0.3 * x).unsqueeze(-1)

    def oracle_target_distance(self, target: float) -> float:
        """``min_x |f(x) - target|`` on ``[0, 1]``.

        Zero whenever ``target`` lies in the range of ``f``, otherwise the
        distance to the nearer end of the range.
        """
        if _SINE_TREND1D_F_MIN <= target <= _SINE_TREND1D_F_MAX:
            return 0.0
        return min(abs(target - _SINE_TREND1D_F_MIN),
                   abs(target - _SINE_TREND1D_F_MAX))


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

    def __init__(
        self,
        noise_std: float = DEFAULT_NOISE_STD_FORRESTER1D,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        default_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        super().__init__(
            name="forrester_1d",
            dim=1,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
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

    def __init__(
        self,
        noise_std: float = DEFAULT_NOISE_STD_BRANIN,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        default_bounds = torch.tensor(
            [[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double
        )
        super().__init__(
            name="branin_2d",
            dim=2,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
            default_noise_std=noise_std,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        term1 = self._a * (x2 - self._b * x1.pow(2) + self._c * x1 - self._r).pow(2)
        term2 = self._s * (1.0 - self._t) * torch.cos(x1)
        return (term1 + term2 + self._s).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Hartmann 3-D
# ---------------------------------------------------------------------------

DEFAULT_NOISE_STD_HARTMANN3: float = 0.05

# Module-level constants to avoid re-allocation on each forward call.
_H3_ALPHA = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.double)

_H3_A = torch.tensor(
    [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ],
    dtype=torch.double,
)

_H3_P = 1e-4 * torch.tensor(
    [
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [ 381.0, 5743.0, 8828.0],
    ],
    dtype=torch.double,
)


class Hartmann3D(SyntheticBenchmark):
    """3-D Hartmann benchmark function.

    - 3-dimensional multi-modal function on [0, 1]^3
    - Four local minima
    - Global minimum value: f* ≈ -3.86278
    - Global minimizer approximately:
        x* ≈ (0.114614, 0.555649, 0.852547)

    Definition:
        f(x) = - sum_{i=1}^4 alpha_i
                exp( - sum_{j=1}^3 A_ij (x_j - P_ij)^2 )

    Args:
        noise_std: Observation noise standard deviation.
    """

    def __init__(
        self,
        noise_std: float = DEFAULT_NOISE_STD_HARTMANN3,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        default_bounds = torch.zeros(2, 3, dtype=torch.double)
        default_bounds[1] = 1.0
        super().__init__(
            name="hartmann_3d",
            dim=3,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
            default_noise_std=noise_std,)

    def f(self, X: torch.Tensor) -> torch.Tensor:
        # Move constants to the same device as X.
        alpha = _H3_ALPHA.to(X.device)
        A = _H3_A.to(X.device)
        P = _H3_P.to(X.device)

        # X: (n, 3)  ->  diff: (n, 4, 3)
        diff = X.unsqueeze(1) - P.unsqueeze(0)
        # inner: (n, 4)
        inner = (A.unsqueeze(0) * diff.pow(2)).sum(dim=-1)
        # result: (n,)
        result = -(alpha.unsqueeze(0) * torch.exp(-inner)).sum(dim=-1)
        return result.unsqueeze(-1)


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

    def __init__(
        self,
        noise_std: float = DEFAULT_NOISE_STD_HARTMANN6,
        bounds: Optional[torch.Tensor] = None,
        ) -> None:
        default_bounds = torch.zeros(2, 6, dtype=torch.double)
        default_bounds[1] = 1.0
        super().__init__(
            name="hartmann_6d",
            dim=6,
            bounds=SyntheticBenchmark.resolve_bounds(default_bounds, bounds),
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
