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


def _benchmark_section(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract ``(name, sub_config)`` from a top-level YAML config.

    Two layouts are supported, in this order of preference:

    Nested (recommended)::

        benchmark:
          name: gaussian_trap_1d
          bounds: [[0.0], [1.0]]
          noise_std: 0.0
          width: 0.08

    Flat (legacy)::

        benchmark: gaussian_trap_1d
        noise_std: 0.0
        bounds:
          lower: [0.0]
          upper: [1.0]
    """
    bench = config.get("benchmark")
    if isinstance(bench, Mapping):
        if "name" not in bench:
            raise KeyError(
                "config['benchmark'] mapping must contain a 'name' key."
            )
        name = str(bench["name"])
        sub = {k: v for k, v in bench.items() if k != "name"}
        return name, sub
    if isinstance(bench, str):
        sub: dict[str, Any] = {}
        if config.get("noise_std") is not None:
            sub["noise_std"] = config["noise_std"]
        if config.get("bounds") is not None:
            sub["bounds"] = config["bounds"]
        return bench, sub
    raise TypeError(
        "config['benchmark'] must be a string (legacy flat layout) or a "
        "mapping with a 'name' key (nested layout)."
    )


def build_benchmark(config: Mapping[str, Any]) -> SyntheticBenchmark:
    """Construct a :class:`SyntheticBenchmark` from a top-level YAML config.

    The benchmark is the single source of truth for its domain and intrinsic
    parameters.  BO-strategy parameters (``beta``, ``budget``,
    ``initial_design_size``, ``num_restarts``, ``raw_samples``, ...) must not
    appear in the ``benchmark`` sub-config — :meth:`SyntheticBenchmark.from_config`
    rejects unknown keys explicitly to avoid silently ignoring typos.

    See :func:`_benchmark_section` for the two accepted YAML layouts.
    """
    name, sub = _benchmark_section(config)
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'.  Known: {sorted(BENCHMARK_REGISTRY)}"
        )
    return BENCHMARK_REGISTRY[name].from_config(sub)


def build_benchmark_from_metadata(
    name: str,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    noise_std: Optional[float] = None,
    bounds: Optional[Any] = None,
) -> SyntheticBenchmark:
    """Instantiate a benchmark from a per-seed metadata dict.

    Kept as a convenience for callers that already have the resolved bounds
    in hand (e.g. ``process.py`` reading ``SeedMetadata``).  Prefer
    :func:`build_benchmark` when starting from a YAML/JSON top-level config.

    Resolution order for the bounds:

    1. ``bounds`` keyword argument.
    2. ``metadata["bounds"]`` if present.
    3. The benchmark class default (with a :mod:`warnings` warning).
    """
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'.  Known: {sorted(BENCHMARK_REGISTRY)}"
        )

    sub: dict[str, Any] = {}
    if noise_std is not None:
        sub["noise_std"] = float(noise_std)
    elif metadata is not None and metadata.get("noise_std") is not None:
        sub["noise_std"] = float(metadata["noise_std"])

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
        sub["bounds"] = bounds

    return BENCHMARK_REGISTRY[name].from_config(sub)


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
    "build_benchmark",
    "build_benchmark_from_metadata",
]