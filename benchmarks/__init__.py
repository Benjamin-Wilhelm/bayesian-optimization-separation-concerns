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
)

__all__ = [
    "SyntheticBenchmark",
    "SineTrend1D",
    "Forrester1D",
    "Branin2D",
    "Hartmann6D",
    "DEFAULT_NOISE_STD_SINE_TREND1D",
    "DEFAULT_NOISE_STD_FORRESTER1D",
    "DEFAULT_NOISE_STD_BRANIN",
    "DEFAULT_NOISE_STD_HARTMANN6",
]
