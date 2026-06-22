"""Minimal checks that bootstrap bands honour the requested central statistic.

Run with ``pytest tests/test_statistics_bands.py`` from the repository root,
or directly with ``python tests/test_statistics_bands.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.statistics import aggregate_convergence_curves, bootstrap_uniform_band


def _skewed_curves(seed: int = 0) -> np.ndarray:
    """Curves whose per-step mean and median differ noticeably (skewed rows)."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=0.0, sigma=0.8, size=(11, 7))
    # Add a step-dependent ramp so the curves look like trajectories.
    return base + np.linspace(0.0, 2.0, 7)[None, :]


def test_uniform_band_median_is_centred_on_median():
    """``statistic="median"`` must centre the uniform band on the median."""
    curves = _skewed_curves()
    band = bootstrap_uniform_band(curves, n_bootstrap=2000, statistic="median")

    expected = np.median(curves, axis=0)
    np.testing.assert_allclose(band["center"], expected)
    # The band must NOT silently fall back to the mean.
    assert not np.allclose(band["center"], np.mean(curves, axis=0))
    assert band["statistic"] == "median"
    # Backward-compatible alias holds the median (and no stray "mean" key).
    np.testing.assert_allclose(band["median"], expected)
    assert "mean" not in band


def test_uniform_band_mean_is_centred_on_mean():
    """``statistic="mean"`` (default) keeps the historical mean-centred band."""
    curves = _skewed_curves()
    band = bootstrap_uniform_band(curves, n_bootstrap=2000, statistic="mean")

    expected = np.mean(curves, axis=0)
    np.testing.assert_allclose(band["center"], expected)
    assert band["statistic"] == "mean"
    np.testing.assert_allclose(band["mean"], expected)  # backward-compat alias


def test_aggregate_uniform_band_propagates_median():
    """The analyze.py entry point must not announce median then return a mean band."""
    curves = _skewed_curves()
    out = aggregate_convergence_curves(
        {"m": curves},
        statistic="median",
        band_type="uniform",
        n_bootstrap=2000,
    )["m"]

    assert out["band_type"] == "uniform"
    assert out["statistic"] == "median"
    np.testing.assert_allclose(np.asarray(out["center"]), np.median(curves, axis=0))


if __name__ == "__main__":
    test_uniform_band_median_is_centred_on_median()
    test_uniform_band_mean_is_centred_on_mean()
    test_aggregate_uniform_band_propagates_median()
    print("OK: uniform bands honour the requested central statistic.")
