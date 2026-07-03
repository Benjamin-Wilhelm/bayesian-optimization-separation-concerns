"""Tests for the UCB exploration schedule :class:`common.acquisition_utils.BetaSchedule`.

Shared by Experiment A and Experiment B.  The central guarantee is backward
compatibility: ``beta: 4.0`` and ``beta_schedule: {type: constant, value: 4.0}``
must yield exactly the same ``beta_t`` at every iteration.
"""

from __future__ import annotations

import math

import pytest

from common.acquisition_utils import BetaSchedule


# ---------------------------------------------------------------------------
# Backward compatibility: constant reproduces the legacy scalar beta
# ---------------------------------------------------------------------------

def test_constant_schedule_reproduces_scalar_beta():
    """`beta: 4.0` and an explicit constant schedule must be identical."""
    legacy = BetaSchedule.from_config({"beta": 4.0})
    explicit = BetaSchedule.from_config(
        {"beta_schedule": {"type": "constant", "value": 4.0}}
    )
    for t in range(1, 51):
        assert legacy.beta_at(t, dim=2) == 4.0
        assert explicit.beta_at(t, dim=2) == 4.0
        assert legacy.beta_at(t, dim=2) == explicit.beta_at(t, dim=2)


def test_constant_is_dimension_independent():
    sched = BetaSchedule.from_config({"beta": 8.0})
    assert sched.beta_at(1, dim=1) == 8.0
    assert sched.beta_at(10, dim=6) == 8.0


def test_schedule_takes_priority_over_scalar_beta():
    """When both are present, `beta_schedule` wins over `beta`."""
    sched = BetaSchedule.from_config(
        {"beta": 4.0, "beta_schedule": {"type": "constant", "value": 9.0}}
    )
    assert sched.kind == "constant"
    assert sched.beta_at(1, dim=2) == 9.0


# ---------------------------------------------------------------------------
# log schedule
# ---------------------------------------------------------------------------

def test_log_schedule_formula():
    sched = BetaSchedule.from_config(
        {"beta_schedule": {"type": "log", "beta0": 4.0, "c": 1.0}}
    )
    for t in (1, 2, 5, 20):
        assert sched.beta_at(t, dim=3) == pytest.approx(4.0 + math.log(t + 1))


def test_log_schedule_default_c_is_one():
    sched = BetaSchedule.from_schedule_dict({"type": "log", "beta0": 2.0})
    assert sched.params["c"] == 1.0
    assert sched.beta_at(3, dim=1) == pytest.approx(2.0 + math.log(4))


def test_log_schedule_is_increasing():
    sched = BetaSchedule.from_config(
        {"beta_schedule": {"type": "log", "beta0": 4.0, "c": 1.0}}
    )
    vals = [sched.beta_at(t, dim=2) for t in range(1, 30)]
    assert all(b < a for b, a in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# matern_gamma schedule
# ---------------------------------------------------------------------------

def _matern_beta(t, dim, B, R, delta, nu, scale):
    exponent = (dim * (dim + 1)) / (2.0 * nu + dim * (dim + 1))
    s = t - 1
    gamma = scale * ((s + 1) ** exponent) * math.log(s + 1)
    return (B + R * math.sqrt(2.0 * (gamma + 1.0 + math.log(1.0 / delta)))) ** 2


def test_matern_gamma_matches_reference_formula():
    cfg = {"type": "matern_gamma", "B": 1.0, "R": 1.0,
           "delta": 0.1, "nu": 2.5, "scale": 1.0}
    sched = BetaSchedule.from_schedule_dict(cfg)
    for t in (1, 2, 5, 25):
        expected = _matern_beta(t, 2, 1.0, 1.0, 0.1, 2.5, 1.0)
        assert sched.beta_at(t, dim=2) == pytest.approx(expected)


def test_matern_gamma_gamma_is_zero_at_first_iteration():
    """At t=1 the information-gain proxy gamma_{t-1} = gamma(0) = 0."""
    cfg = {"type": "matern_gamma", "B": 1.0, "R": 1.0, "delta": 0.1, "nu": 2.5}
    sched = BetaSchedule.from_schedule_dict(cfg)
    expected = (1.0 + 1.0 * math.sqrt(2.0 * (0.0 + 1.0 + math.log(10.0)))) ** 2
    assert sched.beta_at(1, dim=4) == pytest.approx(expected)


def test_matern_gamma_scale_monotonic_effect():
    """Larger `scale` inflates gamma^proxy, hence beta_t, for t > 1."""
    base = {"type": "matern_gamma", "B": 1.0, "R": 1.0, "delta": 0.1, "nu": 2.5}
    low = BetaSchedule.from_schedule_dict({**base, "scale": 0.1})
    high = BetaSchedule.from_schedule_dict({**base, "scale": 1.0})
    assert high.beta_at(10, dim=2) > low.beta_at(10, dim=2)


def test_matern_gamma_defaults():
    sched = BetaSchedule.from_schedule_dict({"type": "matern_gamma"})
    assert sched.params == {"B": 1.0, "R": 1.0, "delta": 0.1, "nu": 2.5, "scale": 1.0}


# ---------------------------------------------------------------------------
# Round-trip and validation
# ---------------------------------------------------------------------------

def test_to_dict_round_trip():
    for cfg in (
        {"type": "constant", "value": 4.0},
        {"type": "log", "beta0": 4.0, "c": 0.5},
        {"type": "matern_gamma", "B": 1.0, "R": 1.0, "delta": 0.1, "nu": 2.5, "scale": 0.1},
    ):
        sched = BetaSchedule.from_schedule_dict(cfg)
        restored = BetaSchedule.from_schedule_dict(sched.to_dict())
        assert restored.to_dict() == sched.to_dict()
        assert restored.beta_at(7, dim=3) == pytest.approx(sched.beta_at(7, dim=3))


def test_beta_at_rejects_non_positive_iteration():
    sched = BetaSchedule.from_config({"beta": 4.0})
    with pytest.raises(ValueError):
        sched.beta_at(0, dim=2)


def test_from_config_requires_beta_or_schedule():
    with pytest.raises(ValueError):
        BetaSchedule.from_config({"budget": 50})


def test_unknown_schedule_type_rejected():
    with pytest.raises(ValueError):
        BetaSchedule.from_schedule_dict({"type": "quadratic", "a": 1.0})


@pytest.mark.parametrize("cfg", [
    {"type": "constant", "value": -1.0},
    {"type": "log", "beta0": -0.5},
    {"type": "matern_gamma", "delta": 0.0},
    {"type": "matern_gamma", "delta": 1.0},
    {"type": "matern_gamma", "nu": 0.0},
])
def test_invalid_parameters_rejected(cfg):
    with pytest.raises(ValueError):
        BetaSchedule.from_schedule_dict(cfg)


def test_all_schedules_yield_nonnegative_beta():
    for cfg in (
        {"type": "constant", "value": 0.0},
        {"type": "log", "beta0": 0.0, "c": 1.0},
        {"type": "matern_gamma", "B": 0.0, "R": 1.0, "delta": 0.5, "nu": 1.5, "scale": 1.0},
    ):
        sched = BetaSchedule.from_schedule_dict(cfg)
        assert all(sched.beta_at(t, dim=2) >= 0.0 for t in range(1, 30))
