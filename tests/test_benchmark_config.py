"""Tests for benchmark YAML/dict configuration.

Covers the public ``from_config`` / ``build_benchmark`` plumbing introduced
to make benchmark intrinsic parameters (domain bounds, observation noise,
geometric parameters) configurable from the YAML, with the benchmark instance
itself as the single source of truth at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks import (
    BENCHMARK_REGISTRY,
    Exponential1D,
    Forrester1D,
    GaussianTrap1D,
    build_benchmark,
)
from common.initial_design import sobol_design


# ---------------------------------------------------------------------------
# from_config: defaults
# ---------------------------------------------------------------------------

def test_forrester_from_config_defaults_match_constructor():
    cfg_default = Forrester1D.from_config({})
    direct = Forrester1D()
    assert torch.equal(cfg_default.bounds, direct.bounds)
    assert cfg_default.default_noise_std == direct.default_noise_std


def test_forrester_from_config_with_explicit_noise():
    bench = Forrester1D.from_config({"noise_std": 0.0})
    assert bench.default_noise_std == 0.0
    # Bounds default to [0, 1].
    assert torch.equal(
        bench.bounds, torch.tensor([[0.0], [1.0]], dtype=torch.double)
    )


# ---------------------------------------------------------------------------
# from_config: custom bounds
# ---------------------------------------------------------------------------

def test_gaussian_trap_from_config_with_custom_bounds():
    bench = GaussianTrap1D.from_config({
        "noise_std": 0.0,
        "bounds": [[-20.0], [20.0]],
        "trap_location": 0.40,
        "width": 0.08,
        "shape": 2.0,
        "trap_gap": 0.02,
    })
    assert torch.equal(
        bench.bounds,
        torch.tensor([[-20.0], [20.0]], dtype=torch.double),
    )


def test_bounds_mapping_form_supported():
    bench = Exponential1D.from_config({
        "noise_std": 0.0,
        "bounds": {"lower": [0.0], "upper": [1.0]},
    })
    assert torch.equal(
        bench.bounds, torch.tensor([[0.0], [1.0]], dtype=torch.double)
    )


# ---------------------------------------------------------------------------
# Deterministic grid reconstruction
# ---------------------------------------------------------------------------

def test_same_config_yields_identical_sobol_grid():
    cfg = {
        "benchmark": {
            "name": "exponential_1d",
            "noise_std": 0.0,
            "bounds": [[-5.0], [5.0]],
            "rate": 3.0,
        }
    }
    b1 = build_benchmark(cfg)
    b2 = build_benchmark(cfg)
    grid1 = sobol_design(b1.bounds, n=64, seed=123)
    grid2 = sobol_design(b2.bounds, n=64, seed=123)
    assert torch.equal(grid1, grid2)
    assert grid1.min().item() >= -5.0
    assert grid1.max().item() <= 5.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_bounds_lower_greater_than_upper():
    with pytest.raises(ValueError, match="lower < upper"):
        Exponential1D.from_config({"bounds": [[1.0], [-1.0]]})


def test_invalid_bounds_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(2, dim\)"):
        Exponential1D.from_config({"bounds": [[0.0, 1.0]]})


def test_gaussian_trap_rejects_invalid_width():
    with pytest.raises(ValueError, match="width must be strictly positive"):
        GaussianTrap1D.from_config({
            "noise_std": 0.0,
            "width": 0.0,
        })


def test_unknown_key_is_rejected():
    # 'beta' is a BO-strategy parameter and must not appear under benchmark.
    with pytest.raises(ValueError, match="Unknown keys"):
        Exponential1D.from_config({"beta": 2.0})


# ---------------------------------------------------------------------------
# build_benchmark: layout dispatch
# ---------------------------------------------------------------------------

def test_build_benchmark_nested_layout():
    cfg = {
        "benchmark": {
            "name": "gaussian_trap_1d",
            "noise_std": 0.0,
            "bounds": [[-1.0], [2.0]],
            "trap_location": 0.40,
            "width": 0.08,
            "shape": 2.0,
            "trap_gap": 0.02,
        }
    }
    bench = build_benchmark(cfg)
    assert isinstance(bench, GaussianTrap1D)
    assert torch.equal(
        bench.bounds,
        torch.tensor([[-1.0], [2.0]], dtype=torch.double),
    )


def test_build_benchmark_flat_layout_legacy():
    cfg = {
        "benchmark": "exponential_1d",
        "noise_std": 0.0,
        "bounds": {"lower": [0.0], "upper": [1.0]},
    }
    bench = build_benchmark(cfg)
    assert bench.name == "exponential_1d"
    assert torch.equal(
        bench.bounds, torch.tensor([[0.0], [1.0]], dtype=torch.double)
    )


def test_build_benchmark_unknown_name():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        build_benchmark({"benchmark": "nonexistent_xxx"})


def test_registry_round_trip_covers_default_benchmarks():
    # Every registered benchmark must build from an empty config (defaults only).
    for name, cls in BENCHMARK_REGISTRY.items():
        if name == "gaussian_trap_1d":
            # GaussianTrap1D has no zero-arg construction; uses default geometry.
            bench = cls.from_config({})
        else:
            bench = cls.from_config({})
        assert bench.name == name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
