"""Synthetic BO benchmarks for SoC Bayesian optimization experiments."""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

import torch

from benchmarks.synthetic import SyntheticBenchmark
from benchmarks.toy import (
    Branin2D,
    DEFAULT_NOISE_STD_BRANIN,
    DEFAULT_NOISE_STD_FORRESTER1D,
    DEFAULT_NOISE_STD_HARTMANN6,
    DEFAULT_NOISE_STD_SINE_TREND1D,
    Forrester1D,
    Hartmann6D,
    SineTrend1D,
    Linear1D,
    Exponential1D,
    GaussianTrap1D,
    DampedOscillatoryTrap1D,
)

# Look-up table used by ``process.py`` to re-instantiate a benchmark from its
# metadata string (``benchmark.name``). Keep keys in sync with each class's
# ``name`` attribute.
BENCHMARK_REGISTRY: dict[str, type[SyntheticBenchmark]] = {
    "sine_trend_1d": SineTrend1D,
    "forrester_1d": Forrester1D,
    "branin_2d": Branin2D,
    "hartmann_6d": Hartmann6D,

    # Minimal / diagnostic 1D target-matching benchmarks
    "linear_1d": Linear1D,
    "exponential_1d": Exponential1D,
    "gaussian_trap_1d": GaussianTrap1D,
    "damped_oscillatory_trap_1d": DampedOscillatoryTrap1D,
}


def build_benchmark_from_metadata(
    name: str,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    noise_std: Optional[float] = None,
    bounds: Optional[torch.Tensor] = None,
) -> SyntheticBenchmark:
    """Instantiate a benchmark, preferring bounds saved in metadata.

    Resolution order for the bounds:

    1. ``bounds`` keyword argument (highest priority).
    2. ``metadata["bounds"]`` if present.
    3. The benchmark class default (with a :mod:`warnings` warning so that the
       caller knows the run did not record its bounds — typical of older runs
       produced before bounds were configurable).

    The ``noise_std`` is resolved similarly: explicit keyword first, then
    ``metadata["noise_std"]`` if present.
    """
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'.  Known: {sorted(BENCHMARK_REGISTRY)}"
        )

    kwargs: dict[str, Any] = {}
    if noise_std is not None:
        kwargs["noise_std"] = float(noise_std)
    elif metadata is not None and metadata.get("noise_std") is not None:
        kwargs["noise_std"] = float(metadata["noise_std"])

    if bounds is None and metadata is not None:
        bounds = metadata.get("bounds")

    if bounds is None:
        warnings.warn(
            f"No bounds found in metadata for benchmark '{name}'. "
            "Falling back to the current default bounds in the benchmark class; "
            "if defaults have changed since this run was produced, derived "
            "grids and figures may not match the original configuration.",
            stacklevel=2,
        )
    else:
        kwargs["bounds"] = torch.as_tensor(bounds, dtype=torch.double)

    return BENCHMARK_REGISTRY[name](**kwargs)


__all__ = [
    "SyntheticBenchmark",
    "SineTrend1D",
    "Forrester1D",
    "Branin2D",
    "Hartmann6D",
    "Linear1D",
    "Exponential1D",
    "GaussianTrap1D",
    "DampedOscillatoryTrap1D",
    "DEFAULT_NOISE_STD_SINE_TREND1D",
    "DEFAULT_NOISE_STD_FORRESTER1D",
    "DEFAULT_NOISE_STD_BRANIN",
    "DEFAULT_NOISE_STD_HARTMANN6",
    "BENCHMARK_REGISTRY",
    "build_benchmark_from_metadata",
]