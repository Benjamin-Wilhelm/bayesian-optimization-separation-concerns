"""Tests for ``SyntheticBenchmark.near_target_test_set``.

The contract is strict:

- exactly ``n`` points are returned,
- every point satisfies ``|f(x) - target| <= epsilon``,
- output is deterministic in ``seed``,
- points are distinct,
- the method works on Hartmann 6D for a sensible target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks import Exponential1D, Hartmann6D, SineTrend1D


# ---------------------------------------------------------------------------
# Test 1 — Exact size and strict band membership on a 1-D benchmark
# ---------------------------------------------------------------------------

def test_size_and_band_strict_1d():
    problem = Exponential1D(noise_std=0.0)
    X_near = problem.near_target_test_set(target=0.5, epsilon=0.05, n=128, seed=7)
    assert X_near.shape == (128, problem.dim)
    f_vals = problem.f(X_near).squeeze(-1)
    assert torch.all((f_vals - 0.5).abs() <= 0.05)


# ---------------------------------------------------------------------------
# Test 2 — Determinism: same config -> same tensor (bit-exact)
# ---------------------------------------------------------------------------

def test_deterministic_for_same_seed():
    problem = SineTrend1D(noise_std=0.0)
    a = problem.near_target_test_set(target=0.4, epsilon=0.05, n=64, seed=11)
    b = problem.near_target_test_set(target=0.4, epsilon=0.05, n=64, seed=11)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Test 3 — Different seeds typically yield different grids
# ---------------------------------------------------------------------------

def test_different_seeds_yield_different_grids():
    problem = SineTrend1D(noise_std=0.0)
    a = problem.near_target_test_set(target=0.4, epsilon=0.05, n=64, seed=11)
    b = problem.near_target_test_set(target=0.4, epsilon=0.05, n=64, seed=12)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# Test 4 — Returned points are distinct
# ---------------------------------------------------------------------------

def test_no_duplicate_points():
    problem = Exponential1D(noise_std=0.0)
    X = problem.near_target_test_set(target=0.5, epsilon=0.05, n=200, seed=3)
    flat = X.squeeze(-1).tolist()
    assert len(set(flat)) == len(flat)


# ---------------------------------------------------------------------------
# Test 5 — Hartmann 6D returns exactly n points satisfying the band
# ---------------------------------------------------------------------------

def test_hartmann6d_near_target_band():
    problem = Hartmann6D(noise_std=0.0)
    # A target inside the function range with a reasonable epsilon so the
    # generic rejection sampler converges in well under the safety cap.
    X = problem.near_target_test_set(target=-2.0, epsilon=0.05, n=128, seed=1)
    assert X.shape == (128, 6)
    f_vals = problem.f(X).squeeze(-1)
    assert torch.all((f_vals + 2.0).abs() <= 0.05)


# ---------------------------------------------------------------------------
# Test 6 — Joint NLPD: global and near-target share the same helper schema
# ---------------------------------------------------------------------------

def test_joint_nlpd_schema_shared_global_and_near_target():
    """The same helper produces the same dict schema for both grids."""
    from common.diagnostics import posterior_joint_calibration_summary

    class _FakeMVN:
        def __init__(self, n):
            self.covariance_matrix = torch.eye(n, dtype=torch.double)

    class _FakePosterior:
        def __init__(self, n):
            self.mean = torch.zeros(n, 1, dtype=torch.double)
            self.variance = torch.ones(n, 1, dtype=torch.double)
            self.mvn = _FakeMVN(n)

    class _FakeModel:
        def posterior(self, X):
            return _FakePosterior(X.shape[0])

    model = _FakeModel()
    X_global = torch.randn(16, 2, dtype=torch.double)
    X_near = torch.randn(8, 2, dtype=torch.double)
    y_global = torch.zeros(16, dtype=torch.double)
    y_near = torch.zeros(8, dtype=torch.double)

    out_global = posterior_joint_calibration_summary(
        model=model, X=X_global, y_true=y_global, jitter=1e-6,
    )
    out_near = posterior_joint_calibration_summary(
        model=model, X=X_near, y_true=y_near, jitter=1e-6,
    )
    expected_keys = {"m_test", "jitter", "nlpd_joint", "mahalanobis_sq", "logdet"}
    assert expected_keys.issubset(out_global.keys())
    assert expected_keys.issubset(out_near.keys())
    assert out_global["m_test"] == 16
    assert out_near["m_test"] == 8


# ---------------------------------------------------------------------------
# Test 7 — ProcessedSeedData round-trip preserves near-target grid metadata
# ---------------------------------------------------------------------------

def test_processed_seed_data_round_trip_preserves_test_grid():
    from data.processed import (
        CalibrationReport,
        ConvergenceCurves,
        FinalScalars,
        MethodProcessed,
        ProcessedSeedData,
        TestGrid,
    )

    tg = TestGrid(
        n_test=512, seed=0,
        near_target_epsilon=0.05,
        near_target_mode="benchmark_near_target_test_set",
        n_test_near_target=128,
        near_target_seed=1,
        m_test_joint_distrib=64,
        m_test_joint_near_target=64,
    )
    psd = ProcessedSeedData(
        metadata={"seed": 0},
        target=0.5,
        epsilon=0.01,
        failure_threshold=0.05,
        oracle_best=0.0,
        test_grid=tg,
        methods={},
    )
    d = psd.to_dict()
    psd2 = ProcessedSeedData.from_dict(d)
    assert psd2.test_grid.to_dict() == tg.to_dict()


# ---------------------------------------------------------------------------
# Test 8 — Legacy processed JSON without new fields still loads
# ---------------------------------------------------------------------------

def test_legacy_test_grid_loads_without_near_target_fields():
    from data.processed import TestGrid
    tg = TestGrid.from_dict({"n_test": 64, "seed": 0})
    assert tg.near_target_mode is None
    assert tg.n_test_near_target is None
    assert tg.near_target_seed is None
    assert tg.m_test_joint_near_target is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_negative_n():
    problem = SineTrend1D(noise_std=0.0)
    with pytest.raises(ValueError, match="n must be non-negative"):
        problem.near_target_test_set(target=0.4, epsilon=0.05, n=-1, seed=0)


def test_rejects_nonpositive_epsilon():
    problem = SineTrend1D(noise_std=0.0)
    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        problem.near_target_test_set(target=0.4, epsilon=0.0, n=8, seed=0)


def test_zero_n_returns_empty():
    problem = SineTrend1D(noise_std=0.0)
    X = problem.near_target_test_set(target=0.4, epsilon=0.05, n=0, seed=0)
    assert X.shape == (0, problem.dim)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
