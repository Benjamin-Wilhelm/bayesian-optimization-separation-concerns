"""Synthetic BO benchmarks for SoC Bayesian optimization experiments."""

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
]