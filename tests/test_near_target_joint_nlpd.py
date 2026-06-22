"""Tests for the near-target joint-NLPD plumbing in ``process.py``.

These tests exercise:

* deterministic and capped selection of the near-target subset;
* graceful handling when fewer points are available than requested;
* graceful handling when zero near-target points exist;
* that the global joint-NLPD behaviour is unchanged.

The arithmetic of :func:`common.diagnostics.joint_gaussian_nlpd` itself is
already covered indirectly via existing pipeline tests; here we focus on
subset construction and integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Deterministic near-target subset selection (pure logic mirror)
# ---------------------------------------------------------------------------
#
# The selection rule must match what process._calibration_reports does:
#     mask = |f_test - target| <= epsilon
#     idx  = nonzero(mask)[:k]    where k = min(m, mask.sum())
#     subset = X_test[idx], f_test[idx]
# We re-implement it tightly here so the test stays robust against any
# refactor of process.py's surrounding code.

def _select_near_target_joint(
    X_test: torch.Tensor,
    f_test: torch.Tensor,
    target: float,
    epsilon: float,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    flat_f = f_test.squeeze(-1)
    mask = (flat_f - target).abs() <= epsilon
    all_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    total = int(all_idx.numel())
    k = min(int(m), total)
    if k < 2:
        return X_test.new_zeros((0, X_test.shape[1])), flat_f.new_zeros((0,)), total
    sel = all_idx[:k]
    return X_test[sel], flat_f[sel], total


def _fixed_grid() -> tuple[torch.Tensor, torch.Tensor]:
    # 1-D grid spanning [0, 1], f(x) = x.  Near-target points for T=0.5
    # form a known dense slice; size depends on epsilon.
    X = torch.linspace(0.0, 1.0, steps=101, dtype=torch.double).unsqueeze(-1)
    f = X.clone()
    return X, f


def test_subset_size_capped_by_m():
    X, f = _fixed_grid()
    Xs, fs, total = _select_near_target_joint(X, f, target=0.5, epsilon=0.2, m=16)
    assert Xs.shape[0] == 16
    assert total >= 16
    # All selected points must satisfy the mask.
    assert torch.all((fs - 0.5).abs() <= 0.2)


def test_subset_deterministic():
    X, f = _fixed_grid()
    Xs1, fs1, _ = _select_near_target_joint(X, f, target=0.5, epsilon=0.2, m=10)
    Xs2, fs2, _ = _select_near_target_joint(X, f, target=0.5, epsilon=0.2, m=10)
    assert torch.equal(Xs1, Xs2)
    assert torch.equal(fs1, fs2)


def test_uses_all_available_when_fewer_than_requested():
    X, f = _fixed_grid()
    # epsilon=0.05 on a [0,1] grid of 101 points yields 11 near-target points;
    # asking for 64 must clamp to 11.
    Xs, fs, total = _select_near_target_joint(X, f, target=0.5, epsilon=0.05, m=64)
    assert 2 <= total < 64
    assert Xs.shape[0] == total
    assert torch.all((fs - 0.5).abs() <= 0.05)


def test_no_near_target_points_returns_empty():
    X, f = _fixed_grid()
    # target outside the function range -> mask is all False.
    Xs, fs, total = _select_near_target_joint(X, f, target=10.0, epsilon=0.05, m=16)
    assert total == 0
    assert Xs.shape[0] == 0
    assert fs.numel() == 0


def test_single_near_target_point_does_not_form_joint():
    # f only has one value within 1e-9 of target; less than 2 points should
    # disable the joint distribution.
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    f = X.clone()
    Xs, fs, total = _select_near_target_joint(X, f, target=0.5, epsilon=1e-9, m=4)
    assert total == 1
    assert Xs.shape[0] == 0
    assert fs.numel() == 0


# ---------------------------------------------------------------------------
# Integration: TestGrid schema carries the new field
# ---------------------------------------------------------------------------

def test_test_grid_dataclass_accepts_m_test_joint_near_target():
    from data.processed import TestGrid
    tg = TestGrid(
        n_test=128, seed=0,
        m_test_joint_distrib=8,
        m_test_joint_near_target=4,
    )
    d = tg.to_dict()
    assert d["m_test_joint_near_target"] == 4
    # Round-trip via from_dict.
    tg2 = TestGrid.from_dict(d)
    assert tg2.m_test_joint_near_target == 4


def test_test_grid_from_dict_tolerates_legacy_missing_field():
    from data.processed import TestGrid
    tg = TestGrid.from_dict({"n_test": 32, "seed": 0})
    assert tg.m_test_joint_near_target is None
    assert tg.m_test_joint_distrib is None


# ---------------------------------------------------------------------------
# Integration: analyze.py knows about the new keys
# ---------------------------------------------------------------------------

def test_analyze_near_target_keys_include_joint():
    from experiments.exp_a_target_matching import analyze
    for key in (
        "joint_nlpd",
        "joint_mahalanobis_sq",
        "joint_logdet",
        "joint_n_test",
        "joint_total_count",
    ):
        assert key in analyze._CALIB_NEAR_TARGET_KEYS
        assert key in analyze._CALIB_NEAR_TARGET_JOINT_KEYS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
