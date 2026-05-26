"""Model-quality and calibration diagnostics for Gaussian process predictions.

All functions are **pure and stateless**.  They accept tensors, numpy arrays,
or plain Python lists, and normalise inputs to 1-D float64 numpy arrays
internally.

Scientific note
---------------
These diagnostics measure how well the GP posterior matches held-out data.

For the **RIGHT** method (GP trained on raw ``y_obs``), use raw ``f_noiseless``
as ``y_true`` in synthetic experiments.

For the **WRONG** method (GP trained on ``|y_obs - T|``), diagnostics may be
computed against either the transformed quantity ``|f_noiseless - T|`` or the
raw ``f_noiseless``, depending on what the calling analysis script is
measuring.  This module does **not** hard-code that distinction; the caller is
responsible for passing the appropriate ``y_true``.

Inputs
------
All vector arguments accept:

- 1-D ``numpy.ndarray`` of shape ``(n,)``
- 2-D array of shape ``(n, 1)`` — flattened automatically
- PyTorch tensors of the above shapes
- Python lists of scalars

Returns
-------
Scalar summaries are Python ``float``.  Array outputs are ``numpy.ndarray``.
Dictionary outputs contain only JSON-serialisable values (``float``, ``int``).
``NaN`` is used to signal that a quantity is undefined (e.g. empty subset).
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy
# ---------------------------------------------------------------------------

_SCIPY_AVAILABLE = False
try:
    import scipy.stats as _scipy_stats  # noqa: F401 (used via name below)
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

# Type alias
ArrayLike = Union[np.ndarray, list, "torch.Tensor"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_1d(values: ArrayLike, name: str = "values") -> np.ndarray:
    """Coerce any array-like input to a 1-D float64 numpy array."""
    if hasattr(values, "detach"):          # torch.Tensor
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (n,) or 2-D (n,1), got shape {arr.shape}"
        )
    return arr


def _resolve_std(
    std: Optional[ArrayLike],
    var: Optional[ArrayLike],
    eps: float,
) -> np.ndarray:
    """Return a clamped 1-D std array from either ``std`` or ``var``."""
    if std is not None and var is not None:
        raise ValueError("Provide either std or var, not both.")
    if std is None and var is None:
        raise ValueError("Must provide either std or var.")
    if var is not None:
        s = np.sqrt(np.maximum(_to_numpy_1d(var, "var"), 0.0))
    else:
        s = _to_numpy_1d(std, "std")
    return np.maximum(s, eps)


# ---------------------------------------------------------------------------
# Fallback normal distribution functions (no scipy)
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal CDF via Python's ``math.erf`` (exact)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_quantile(p: float) -> float:
    """Approximate inverse standard normal CDF.

    Uses the rational approximation from Abramowitz & Stegun 26.2.17.
    Maximum absolute error < 4.5e-4.  Only called when scipy is unavailable.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    if p > 0.5:
        q, sign = 1.0 - p, 1.0
    else:
        q, sign = p, -1.0
    t = math.sqrt(-2.0 * math.log(q))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    approx = t - (c[0] + c[1] * t + c[2] * t ** 2) / (
        1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
    )
    return sign * approx


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------

def gaussian_nlpd(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    eps: float = 1e-9,
) -> float:
    """Mean negative log predictive density under independent Gaussian predictions.

    .. math::

        \\mathrm{NLPD} = \\frac{1}{n} \\sum_{i=1}^{n}
            \\left[
                \\frac{(y_i - \\mu_i)^2}{2 \\sigma_i^2}
                + \\frac{1}{2} \\ln(2 \\pi \\sigma_i^2)
            \\right]

    Args:
        y_true: Observed values, shape ``(n,)``.
        mean: Predictive means, shape ``(n,)``.
        std: Predictive standard deviations, shape ``(n,)``.
            Provide either ``std`` or ``var``.
        var: Predictive variances, shape ``(n,)``.
            Provide either ``std`` or ``var``.
        eps: Minimum value to clamp ``std`` for numerical stability.

    Returns:
        Scalar float — mean NLPD across all ``n`` test points.
    """
    y = _to_numpy_1d(y_true, "y_true")
    mu = _to_numpy_1d(mean, "mean")
    s = _resolve_std(std, var, eps)
    nlpd_per_point = 0.5 * ((y - mu) ** 2 / s ** 2 + np.log(2.0 * math.pi * s ** 2))
    return float(np.mean(nlpd_per_point))


def standardized_residuals(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    eps: float = 1e-9,
) -> np.ndarray:
    r"""Standardized residuals :math:`z_i = (y_i - \mu_i) / \sigma_i`.

    For a well-specified model with correct noise, these should be
    approximately ``N(0, 1)``.

    Args:
        y_true: Observed values, shape ``(n,)``.
        mean: Predictive means, shape ``(n,)``.
        std: Predictive standard deviations, shape ``(n,)``.
        var: Predictive variances, shape ``(n,)``.
        eps: Minimum std clamp.

    Returns:
        Array of shape ``(n,)`` — standardized residuals.
    """
    y = _to_numpy_1d(y_true, "y_true")
    mu = _to_numpy_1d(mean, "mean")
    s = _resolve_std(std, var, eps)
    return (y - mu) / s


def empirical_coverage(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    level: float = 0.95,
    eps: float = 1e-9,
) -> float:
    r"""Fraction of ``y_true`` inside the ``level`` predictive credible interval.

    The symmetric Gaussian interval at level :math:`(1-\alpha)` is
    :math:`[\mu - z_{\alpha/2} \sigma,\; \mu + z_{\alpha/2} \sigma]`
    where :math:`z_{\alpha/2} = \Phi^{-1}((1 + \text{level})/2)`.

    Args:
        y_true: Observed values, shape ``(n,)``.
        mean: Predictive means, shape ``(n,)``.
        std: Predictive standard deviations, shape ``(n,)``.
        var: Predictive variances, shape ``(n,)``.
        level: Nominal coverage level, e.g. ``0.95``.
        eps: Minimum std clamp.

    Returns:
        Scalar float in ``[0, 1]`` — empirical coverage fraction.
    """
    y = _to_numpy_1d(y_true, "y_true")
    mu = _to_numpy_1d(mean, "mean")
    s = _resolve_std(std, var, eps)
    if _SCIPY_AVAILABLE:
        import scipy.stats
        z = scipy.stats.norm.ppf((1.0 + level) / 2.0)
    else:
        z = _normal_quantile((1.0 + level) / 2.0)
    inside = np.abs(y - mu) <= z * s
    return float(np.mean(inside))


def pit_values(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    eps: float = 1e-9,
) -> np.ndarray:
    r"""Probability integral transform (PIT) values.

    :math:`\mathrm{PIT}_i = \Phi(z_i)` where
    :math:`z_i = (y_i - \mu_i) / \sigma_i`.

    For a well-calibrated model these should be approximately
    :math:`\mathrm{Uniform}(0, 1)`.

    Args:
        y_true: Observed values, shape ``(n,)``.
        mean: Predictive means, shape ``(n,)``.
        std: Predictive standard deviations, shape ``(n,)``.
        var: Predictive variances, shape ``(n,)``.
        eps: Minimum std clamp.

    Returns:
        Array of shape ``(n,)`` — PIT values in ``[0, 1]``.
    """
    z = standardized_residuals(y_true, mean, std=std, var=var, eps=eps)
    if _SCIPY_AVAILABLE:
        import scipy.stats
        return scipy.stats.norm.cdf(z)
    return np.array([_normal_cdf(float(zi)) for zi in z])


def confidence_bound_violation_rate(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    beta: float = 2.0,
    eps: float = 1e-9,
) -> float:
    r"""Fraction of points where :math:`|y - \mu| > \sqrt{\beta}\,\sigma`.

    Uses the UCB convention: ``beta`` scales variance, so the confidence
    half-width is :math:`\sqrt{\beta} \cdot \sigma`.

    Args:
        y_true: Observed values, shape ``(n,)``.
        mean: Predictive means, shape ``(n,)``.
        std: Predictive standard deviations, shape ``(n,)``.
        var: Predictive variances, shape ``(n,)``.
        beta: Confidence scaling parameter (applied to variance).
        eps: Minimum std clamp.

    Returns:
        Scalar float in ``[0, 1]`` — fraction of bound violations.
    """
    y = _to_numpy_1d(y_true, "y_true")
    mu = _to_numpy_1d(mean, "mean")
    s = _resolve_std(std, var, eps)
    violations = np.abs(y - mu) > math.sqrt(beta) * s
    return float(np.mean(violations))


def calibration_summary(
    y_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    level: float = 0.95,
    beta: float = 2.0,
) -> dict:
    """Full calibration summary dictionary.

    Combines NLPD, empirical coverage, standardized-residual statistics,
    PIT statistics, and confidence-bound violation rate into a single
    JSON-serialisable dictionary.

    Args:
        y_true: Observed values.
        mean: Predictive means.
        std: Predictive standard deviations.  Provide either ``std`` or ``var``.
        var: Predictive variances.  Provide either ``std`` or ``var``.
        level: Nominal coverage level for ``empirical_coverage``.
        beta: Confidence scaling for ``confidence_bound_violation_rate``.

    Returns:
        Dictionary with keys (all ``float``):

        - ``nlpd``: mean negative log predictive density
        - ``coverage``: empirical coverage at ``level``
        - ``mean_standardized_residual``: mean of :math:`z`-scores
        - ``std_standardized_residual``: sample std of :math:`z`-scores
          (``NaN`` if ``n=1``)
        - ``mean_abs_standardized_residual``: mean of :math:`|z|`
        - ``pit_mean``: mean of PIT values
        - ``pit_std``: sample std of PIT values (``NaN`` if ``n=1``)
        - ``confidence_violation_rate``: fraction of :math:`\\beta`-band
          violations
    """
    z = standardized_residuals(y_true, mean, std=std, var=var)
    pits = pit_values(y_true, mean, std=std, var=var)
    n = len(z)
    return {
        "nlpd": gaussian_nlpd(y_true, mean, std=std, var=var),
        "coverage": empirical_coverage(y_true, mean, std=std, var=var, level=level),
        "mean_standardized_residual": float(np.mean(z)),
        "std_standardized_residual": float(np.std(z, ddof=1)) if n > 1 else float("nan"),
        "mean_abs_standardized_residual": float(np.mean(np.abs(z))),
        "pit_mean": float(np.mean(pits)),
        "pit_std": float(np.std(pits, ddof=1)) if n > 1 else float("nan"),
        "confidence_violation_rate": confidence_bound_violation_rate(
            y_true, mean, std=std, var=var, beta=beta
        ),
    }


# ---------------------------------------------------------------------------
# Joint Gaussian NLPD on a small subgrid
# ---------------------------------------------------------------------------

def joint_gaussian_nlpd(
    y_true: ArrayLike,
    mean: ArrayLike,
    covariance: ArrayLike,
    jitter: float = 1e-6,
) -> dict:
    r"""Negative log predictive density under a joint multivariate Gaussian.

    .. math::

        \mathrm{NLPD}_\mathrm{joint} =
        \frac{1}{2m}\left[
            (y - \mu)^\top K^{-1} (y - \mu)
            + \log\det K
            + m \log(2\pi)
        \right]

    Uses a Cholesky factorisation with a configurable jitter on the diagonal
    for numerical stability.  No matrix inverse is formed explicitly.

    Args:
        y_true: Observed values, shape ``(m,)``.
        mean: Predictive means, shape ``(m,)``.
        covariance: Joint posterior covariance matrix, shape ``(m, m)``.
        jitter: Additive diagonal jitter for stability; e.g. ``1e-6``.

    Returns:
        Dictionary with keys (all JSON-serialisable):

        - ``m_test``: int — number of test points.
        - ``nlpd_joint``: float — averaged joint NLPD.
        - ``mahalanobis_sq``: float — :math:`(y - \mu)^\top K^{-1} (y - \mu)`.
        - ``logdet``: float — :math:`\log\det K`.
        - ``jitter``: float — jitter actually used.

        On numerical failure, returns ``{"error": <str>, "jitter": jitter,
        "m_test": m, ...}`` instead of raising.
    """
    y = _to_numpy_1d(y_true, "y_true")
    mu = _to_numpy_1d(mean, "mean")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return {
            "m_test": int(len(y)),
            "jitter": float(jitter),
            "nlpd_joint": float("nan"),
            "mahalanobis_sq": float("nan"),
            "logdet": float("nan"),
            "error": f"covariance must be square 2-D, got shape {cov.shape}",
        }
    m = len(y)
    if cov.shape[0] != m or len(mu) != m:
        return {
            "m_test": m,
            "jitter": float(jitter),
            "nlpd_joint": float("nan"),
            "mahalanobis_sq": float("nan"),
            "logdet": float("nan"),
            "error": (
                f"shape mismatch: y_true ({m}), mean ({len(mu)}), "
                f"covariance ({cov.shape})"
            ),
        }

    K = 0.5 * (cov + cov.T)  # enforce symmetry
    K = K + float(jitter) * np.eye(m)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError as e:
        return {
            "m_test": m,
            "jitter": float(jitter),
            "nlpd_joint": float("nan"),
            "mahalanobis_sq": float("nan"),
            "logdet": float("nan"),
            "error": f"Cholesky failed: {e}",
        }

    diff = y - mu
    alpha = np.linalg.solve(L, diff)
    mahalanobis_sq = float(np.dot(alpha, alpha))
    logdet = float(2.0 * np.sum(np.log(np.diag(L))))
    nlpd_joint = (mahalanobis_sq + logdet + m * math.log(2.0 * math.pi)) / (2.0 * m)
    return {
        "m_test": m,
        "jitter": float(jitter),
        "nlpd_joint": float(nlpd_joint),
        "mahalanobis_sq": mahalanobis_sq,
        "logdet": logdet,
    }


# ---------------------------------------------------------------------------
# Near-target diagnostics
# ---------------------------------------------------------------------------

def near_target_mask(
    f_true: ArrayLike,
    target: float,
    epsilon: float,
) -> np.ndarray:
    r"""Boolean mask of test points close to the target.

    Returns ``True`` for indices where :math:`|f_{\mathrm{true},i} - T|
    \le \epsilon`.

    Args:
        f_true: True function values at test points, shape ``(n,)``.
        target: Target value ``T``.
        epsilon: Neighbourhood radius around the target.

    Returns:
        Boolean array of shape ``(n,)``.
    """
    f = _to_numpy_1d(f_true, "f_true")
    return np.abs(f - target) <= epsilon


def near_target_calibration_summary(
    f_true: ArrayLike,
    mean: ArrayLike,
    std: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    target: Optional[float] = None,
    epsilon: Optional[float] = None,
    level: float = 0.95,
    beta: float = 2.0,
) -> dict:
    r"""Calibration summary restricted to near-target test points.

    Selects test points where :math:`|f_{\mathrm{true},i} - T| \le \epsilon`
    and computes :func:`calibration_summary` on that subset.

    If no near-target points exist, all calibration values are ``NaN`` and
    the function returns without raising an exception.

    Args:
        f_true: True function values at test points, shape ``(n,)``.
        mean: Predictive means at those test points, shape ``(n,)``.
        std: Predictive standard deviations, shape ``(n,)``.
            Provide either ``std`` or ``var``.
        var: Predictive variances, shape ``(n,)``.
        target: Target value ``T``.  Required.
        epsilon: Neighbourhood radius.  Required.
        level: Nominal coverage level.
        beta: UCB confidence scaling parameter.

    Returns:
        Dictionary with all keys from :func:`calibration_summary` plus:

        - ``near_target_count``: ``int`` — number of near-target test points
        - ``near_target_fraction``: ``float`` — fraction of all test points
          that are near the target

        All calibration keys are ``NaN`` when ``near_target_count == 0``.
    """
    if target is None or epsilon is None:
        raise ValueError("Both 'target' and 'epsilon' are required.")

    f = _to_numpy_1d(f_true, "f_true")
    mu = _to_numpy_1d(mean, "mean")
    mask = near_target_mask(f, target, epsilon)
    n_total = len(f)
    n_near = int(np.sum(mask))

    empty: dict = {
        "nlpd": float("nan"),
        "coverage": float("nan"),
        "mean_standardized_residual": float("nan"),
        "std_standardized_residual": float("nan"),
        "mean_abs_standardized_residual": float("nan"),
        "pit_mean": float("nan"),
        "pit_std": float("nan"),
        "confidence_violation_rate": float("nan"),
        "near_target_count": 0,
        "near_target_fraction": 0.0,
    }
    if n_near == 0:
        return empty

    f_sub = f[mask]
    mu_sub = mu[mask]
    std_sub: Optional[np.ndarray] = None
    var_sub: Optional[np.ndarray] = None
    if std is not None:
        std_sub = _to_numpy_1d(std, "std")[mask]
    if var is not None:
        var_sub = _to_numpy_1d(var, "var")[mask]

    summary = calibration_summary(
        f_sub, mu_sub, std=std_sub, var=var_sub, level=level, beta=beta
    )
    summary["near_target_count"] = n_near
    summary["near_target_fraction"] = float(n_near / n_total)
    return summary


# ---------------------------------------------------------------------------
# Optional KS tests (require scipy)
# ---------------------------------------------------------------------------

def ks_test_standard_normal(z: ArrayLike) -> dict:
    """Kolmogorov–Smirnov test of standardized residuals against ``N(0, 1)``.

    Requires scipy.  Returns ``NaN`` statistics and a flag when scipy is
    unavailable rather than raising.

    Args:
        z: Standardized residuals, shape ``(n,)``.

    Returns:
        Dictionary with keys:

        - ``statistic``: KS statistic (``float``)
        - ``p_value``: two-sided p-value (``float``)
        - ``scipy_available``: ``bool``
    """
    z_arr = _to_numpy_1d(z, "z")
    if not _SCIPY_AVAILABLE:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "scipy_available": False}
    import scipy.stats
    res = scipy.stats.kstest(z_arr, "norm")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "scipy_available": True}


def ks_test_uniform(u: ArrayLike) -> dict:
    """Kolmogorov–Smirnov test of PIT values against ``Uniform(0, 1)``.

    Requires scipy.  Returns ``NaN`` statistics and a flag when scipy is
    unavailable rather than raising.

    Args:
        u: PIT values, shape ``(n,)``.

    Returns:
        Dictionary with keys:

        - ``statistic``: KS statistic (``float``)
        - ``p_value``: two-sided p-value (``float``)
        - ``scipy_available``: ``bool``
    """
    u_arr = _to_numpy_1d(u, "u")
    if not _SCIPY_AVAILABLE:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "scipy_available": False}
    import scipy.stats
    res = scipy.stats.kstest(u_arr, "uniform")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "scipy_available": True}

