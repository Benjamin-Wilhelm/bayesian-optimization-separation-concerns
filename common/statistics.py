"""Multi-seed aggregation and pairwise comparison utilities.

All functions are **pure and stateless**.  They accept tensors, numpy arrays,
or plain Python lists, and normalise inputs to numpy arrays internally.

Sign convention
---------------
Smaller metric values are better (minimisation convention for regret /
target distance).

- :func:`mann_whitney_comparison` with ``alternative="less"`` tests that
  RIGHT values are stochastically smaller than WRONG values.
- :func:`rank_biserial_correlation` returns a **positive** value when RIGHT
  tends to have smaller values than WRONG.
- :func:`cohens_d_right_better` returns a **positive** value when WRONG has
  a larger mean, i.e. RIGHT is better.

Curve convention
----------------
Convergence curves have shape ``(n_runs, n_steps)``.
:func:`bootstrap_ci_curves` resamples the *run* dimension (rows) and returns
pointwise confidence intervals across the step dimension (columns).
The returned arrays are ``numpy.ndarray`` and are **not** JSON-serialisable;
call ``.tolist()`` before writing them to disk.

Inputs
------
All 1-D vector arguments accept:

- 1-D ``numpy.ndarray`` of shape ``(n,)``
- 2-D array of shape ``(n, 1)`` — flattened automatically
- PyTorch tensors of the above shapes
- Python lists of scalars
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy
# ---------------------------------------------------------------------------

_SCIPY_AVAILABLE = False
try:
    import scipy.stats as _scipy_stats  # noqa: F401
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


def _to_numpy_2d(values: ArrayLike, name: str = "values") -> np.ndarray:
    """Coerce any array-like input to a 2-D float64 numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_runs, n_steps), got shape {arr.shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# Per-sample descriptive statistics
# ---------------------------------------------------------------------------

def summarize_sample(values: ArrayLike) -> dict:
    """Descriptive statistics for a 1-D sample.

    Args:
        values: Sample values, shape ``(n,)``.

    Returns:
        Dictionary with keys (all JSON-serialisable):

        - ``n``: sample size (``int``)
        - ``mean``, ``std``, ``stderr``: location and spread (``float``)
        - ``median``, ``q25``, ``q75``: quartiles (``float``)
        - ``min``, ``max``: extremes (``float``)
        - ``q90``, ``q95``: upper quantiles (``float``)

        ``std`` and ``stderr`` are ``NaN`` for samples of size 1.
    """
    v = _to_numpy_1d(values, "values")
    n = len(v)
    std = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    stderr = std / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": float(np.mean(v)),
        "std": std,
        "stderr": stderr,
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "q90": float(np.percentile(v, 90)),
        "q95": float(np.percentile(v, 95)),
    }


def failure_rate(values: ArrayLike, threshold: float) -> float:
    """Empirical probability that ``values > threshold``.

    Args:
        values: Sample values, shape ``(n,)``.
        threshold: Failure criterion.  A run is counted as failed when its
            value is **strictly greater** than ``threshold``.

    Returns:
        Float in ``[0, 1]``.
    """
    v = _to_numpy_1d(values, "values")
    return float(np.mean(v > threshold))


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def mann_whitney_comparison(
    right_values: ArrayLike,
    wrong_values: ArrayLike,
    alternative: str = "less",
) -> dict:
    """Mann–Whitney U test comparing RIGHT vs WRONG final performance.

    Since smaller regret is better, ``alternative="less"`` tests the
    one-sided hypothesis that RIGHT values are stochastically smaller
    than WRONG values.

    Requires scipy.  Returns ``NaN`` statistics when scipy is unavailable.

    Args:
        right_values: Per-seed final metrics for the RIGHT method, shape
            ``(n,)``.
        wrong_values: Per-seed final metrics for the WRONG method, shape
            ``(m,)``.
        alternative: One of ``"less"`` (RIGHT < WRONG), ``"greater"``,
            or ``"two-sided"``.

    Returns:
        Dictionary with keys:

        - ``statistic``: U statistic (``float``)
        - ``p_value``: (``float``)
        - ``alternative``: (``str``)
        - ``n_right``, ``n_wrong``: sample sizes (``int``)
        - ``scipy_available``: (``bool``)
    """
    r = _to_numpy_1d(right_values, "right_values")
    w = _to_numpy_1d(wrong_values, "wrong_values")
    result: dict = {
        "alternative": alternative,
        "n_right": len(r),
        "n_wrong": len(w),
        "scipy_available": _SCIPY_AVAILABLE,
    }
    if not _SCIPY_AVAILABLE:
        result["statistic"] = float("nan")
        result["p_value"] = float("nan")
        return result
    import scipy.stats
    mw = scipy.stats.mannwhitneyu(r, w, alternative=alternative)
    result["statistic"] = float(mw.statistic)
    result["p_value"] = float(mw.pvalue)
    return result


def wilcoxon_paired_comparison(
    right_values: ArrayLike,
    wrong_values: ArrayLike,
    alternative: str = "greater",
) -> dict:
    r"""Paired Wilcoxon signed-rank test on differences ``wrong - right``.

    For loss-type metrics (smaller is better), RIGHT being preferable means
    ``Delta_i = wrong_i - right_i > 0`` on average, which is the default
    ``alternative="greater"``.  The test assumes RIGHT and WRONG share the
    *same seed and initial design* (paired by index).

    Requires scipy.  Returns ``NaN`` statistics with ``scipy_available=False``
    when scipy is missing.

    Args:
        right_values: Per-seed metric for RIGHT, shape ``(n,)``.
        wrong_values: Per-seed metric for WRONG, shape ``(n,)``.  Must align
            seed-for-seed with ``right_values``.
        alternative: ``"greater"`` (default, ``H_1: median(diff) > 0``,
            i.e. RIGHT better), ``"less"``, or ``"two-sided"``.

    Returns:
        Dictionary with keys:

        - ``test``: ``"wilcoxon_signed_rank"``
        - ``alternative``: forwarded string
        - ``n``: paired sample size (``int``)
        - ``statistic``: signed-rank statistic (``float``; ``NaN`` if scipy
          missing or all differences are zero)
        - ``p_value``: ``float``
        - ``median_difference_wrong_minus_right``: ``float``
        - ``mean_difference_wrong_minus_right``: ``float``
        - ``scipy_available``: ``bool``
    """
    r = _to_numpy_1d(right_values, "right_values")
    w = _to_numpy_1d(wrong_values, "wrong_values")
    if r.shape != w.shape:
        raise ValueError(
            f"right_values and wrong_values must have the same shape for a "
            f"paired test; got {r.shape} vs {w.shape}."
        )
    diff = w - r
    n = int(len(diff))
    result: dict = {
        "test": "wilcoxon_signed_rank",
        "alternative": alternative,
        "n": n,
        "median_difference_wrong_minus_right": (
            float(np.median(diff)) if n > 0 else float("nan")
        ),
        "mean_difference_wrong_minus_right": (
            float(np.mean(diff)) if n > 0 else float("nan")
        ),
        "scipy_available": _SCIPY_AVAILABLE,
    }
    if not _SCIPY_AVAILABLE or n == 0:
        result["statistic"] = float("nan")
        result["p_value"] = float("nan")
        return result
    import scipy.stats
    try:
        res = scipy.stats.wilcoxon(diff, alternative=alternative)
        result["statistic"] = float(res.statistic)
        result["p_value"] = float(res.pvalue)
    except ValueError as exc:
        # scipy raises when all diffs are zero — treat as no evidence.
        result["statistic"] = float("nan")
        result["p_value"] = float("nan")
        result["error"] = str(exc)
    return result


def rank_biserial_correlation(
    right_values: ArrayLike,
    wrong_values: ArrayLike,
) -> float:
    r"""Rank-biserial correlation between RIGHT and WRONG performance.

    Computed from all :math:`n_R \times n_W` pairwise comparisons:

    .. math::

        \rho = P(\text{wrong} > \text{right}) - P(\text{wrong} < \text{right})

    Ties are counted as neither a win nor a loss (neutral treatment).

    **Sign convention**: positive :math:`\rho` means RIGHT tends to have
    smaller values than WRONG — i.e. RIGHT is better for minimisation.

    Args:
        right_values: Per-seed final metrics for RIGHT, shape ``(n,)``.
        wrong_values: Per-seed final metrics for WRONG, shape ``(m,)``.

    Returns:
        Float in ``[-1, 1]``.  Returns ``NaN`` if either sample is empty.
    """
    r = _to_numpy_1d(right_values, "right_values")
    w = _to_numpy_1d(wrong_values, "wrong_values")
    n_pairs = len(r) * len(w)
    if n_pairs == 0:
        return float("nan")
    # Vectorised pairwise comparison: shape (n_right, n_wrong)
    r_grid, w_grid = np.meshgrid(r, w, indexing="ij")
    wins = float(np.sum(w_grid > r_grid))    # wrong > right: RIGHT wins
    losses = float(np.sum(w_grid < r_grid))  # wrong < right: RIGHT loses
    return (wins - losses) / n_pairs


def cohens_d_right_better(
    right_values: ArrayLike,
    wrong_values: ArrayLike,
) -> float:
    r"""Cohen's d with sign convention: positive means RIGHT is better.

    .. math::

        d = \frac{\bar{x}_{\text{wrong}} - \bar{x}_{\text{right}}}
                 {s_{\text{pooled}}}

    where :math:`s_{\text{pooled}}` is the pooled standard deviation.
    A positive value means WRONG has a larger mean regret, so RIGHT is better.

    Args:
        right_values: Per-seed final metrics for RIGHT, shape ``(n,)``.
        wrong_values: Per-seed final metrics for WRONG, shape ``(m,)``.

    Returns:
        Float — Cohen's d.  Returns ``NaN`` when either sample has fewer
        than 2 points or the pooled std is effectively zero.
    """
    r = _to_numpy_1d(right_values, "right_values")
    w = _to_numpy_1d(wrong_values, "wrong_values")
    nr, nw = len(r), len(w)
    if nr < 2 or nw < 2:
        return float("nan")
    var_r = np.var(r, ddof=1)
    var_w = np.var(w, ddof=1)
    pooled_var = ((nr - 1) * var_r + (nw - 1) * var_w) / (nr + nw - 2)
    pooled_std = math.sqrt(max(pooled_var, 0.0))
    if pooled_std < 1e-15:
        return float("nan")
    return float((np.mean(w) - np.mean(r)) / pooled_std)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for convergence curves
# ---------------------------------------------------------------------------

def bootstrap_ci_curves(
    curves: ArrayLike,
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    random_state: Optional[int] = None,
) -> dict:
    """Pointwise bootstrap confidence interval for a set of convergence curves.

    Resamples the *run* dimension (rows) with replacement and computes
    the chosen statistic at each step.  The CI is **pointwise in time**,
    not a simultaneous band.

    Args:
        curves: Array of shape ``(n_runs, n_steps)``.  Each row is one run's
            convergence trajectory (e.g. running target simple regret).
        statistic: ``"mean"`` or ``"median"``.
        confidence: Nominal CI level, e.g. ``0.95``.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Optional integer seed for reproducibility.

    Returns:
        Dictionary with keys (all ``numpy.ndarray`` of shape ``(n_steps,)``):

        - ``center``: point estimate (mean or median across runs)
        - ``lower``: lower CI bound
        - ``upper``: upper CI bound

        **Note**: these arrays are not JSON-serialisable; call ``.tolist()``
        before writing to disk.
    """
    arr = _to_numpy_2d(curves, "curves")
    n_runs, n_steps = arr.shape
    rng = np.random.default_rng(random_state)

    if statistic == "mean":
        def _stat(x: np.ndarray) -> np.ndarray:
            return np.mean(x, axis=1)
    elif statistic == "median":
        def _stat(x: np.ndarray) -> np.ndarray:
            return np.median(x, axis=1)
    else:
        raise ValueError(f"statistic must be 'mean' or 'median', got '{statistic}'")

    center = _stat(arr[np.newaxis, :, :])[0]   # shape (n_steps,)

    # Vectorised resampling: shape (n_bootstrap, n_runs, n_steps)
    idx = rng.integers(0, n_runs, size=(n_bootstrap, n_runs))
    boot_samples = arr[idx]                      # (n_bootstrap, n_runs, n_steps)

    # Compute statistic along the run axis for each bootstrap replicate
    if statistic == "mean":
        boot_stats = np.mean(boot_samples, axis=1)    # (n_bootstrap, n_steps)
    else:
        boot_stats = np.median(boot_samples, axis=1)  # (n_bootstrap, n_steps)

    alpha = 1.0 - confidence
    lower = np.percentile(boot_stats, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0), axis=0)

    return {"center": center, "lower": lower, "upper": upper}


def bootstrap_uniform_band(
    curves: ArrayLike,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
    statistic: Literal["mean", "median"] = "mean",
) -> dict:
    r"""Simultaneous (uniform) bootstrap confidence band over a curve.

    Given an array of shape ``(n_seeds, n_steps)``, the band is centred on a
    chosen central ``statistic`` (mean or median) and constructed by
    resampling rows (seeds) with replacement, tracking the running maximum
    standardised deviation:

    .. math::

        Q^{(b)} = \max_t \;
          \left| \frac{\hat{c}^{(b)}_t - \hat{c}_t}{\widehat{\mathrm{SE}}_t}
                 \right|

    where :math:`\hat{c}_t` is the central curve under the chosen
    ``statistic`` (the **same** statistic is recomputed on each resample) and
    :math:`\widehat{\mathrm{SE}}_t` is the bootstrap-estimated pointwise
    standard error of that statistic.  The ``confidence``-quantile of
    :math:`Q^{(b)}` gives the band half-width.  The band is therefore **not**
    necessarily centred on the mean — it follows ``statistic``.

    Args:
        curves: Array of shape ``(n_seeds, n_steps)``.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Nominal simultaneous coverage, e.g. ``0.95``.
        seed: Optional integer seed for reproducibility.
        statistic: ``"mean"`` or ``"median"`` — the central statistic the
            band is built around.

    Returns:
        Dictionary with keys (numpy arrays converted to lists for callers
        that need JSON output):

        - ``center``: central curve under ``statistic``, shape ``(n_steps,)``
        - ``lower``: ``center - q * se``, shape ``(n_steps,)``
        - ``upper``: ``center + q * se``, shape ``(n_steps,)``
        - ``statistic``: ``"mean"`` or ``"median"``
        - ``confidence``: ``float``
        - ``q``: scalar quantile multiplier
        - ``n_bootstrap``: ``int``
        - ``seed``: ``Optional[int]``
        - ``band_type``: ``"uniform"``

        For backward compatibility the central curve is also exposed under a
        key named after ``statistic`` (``"mean"`` for a mean band,
        ``"median"`` for a median band).  This named key only ever holds the
        matching statistic, so a median band can never be mislabelled as a
        mean band.

        The numerical fields ``center``, ``lower``, ``upper`` are
        ``numpy.ndarray``; call ``.tolist()`` before JSON serialisation.
    """
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got '{statistic}'")
    arr = _to_numpy_2d(curves, "curves")
    n_runs, n_steps = arr.shape
    rng = np.random.default_rng(seed)

    reducer = np.mean if statistic == "mean" else np.median

    center_curve = reducer(arr, axis=0)   # (n_steps,)
    idx = rng.integers(0, n_runs, size=(n_bootstrap, n_runs))
    boot = reducer(arr[idx], axis=1)      # (n_bootstrap, n_steps)
    se = boot.std(axis=0, ddof=1)
    se = np.where(se > 0, se, 1.0)  # avoid 0/0 for degenerate columns

    max_stat = np.max(np.abs((boot - center_curve[None, :]) / se[None, :]), axis=1)
    q = float(np.quantile(max_stat, confidence))

    band = {
        "center": center_curve,
        "lower":  center_curve - q * se,
        "upper":  center_curve + q * se,
        "statistic":   statistic,
        "confidence":  float(confidence),
        "q":           q,
        "n_bootstrap": int(n_bootstrap),
        "seed":        None if seed is None else int(seed),
        "band_type":   "uniform",
    }
    # Backward-compatible alias: also expose the central curve under its
    # statistic name. Never mislabels — the key matches what it contains.
    band[statistic] = center_curve
    return band


# ---------------------------------------------------------------------------
# Aggregate comparison utilities
# ---------------------------------------------------------------------------

def compare_final_performance(
    right_values: ArrayLike,
    wrong_values: ArrayLike,
    failure_threshold: Optional[float] = None,
) -> dict:
    """Aggregate comparison dictionary for final-iteration performance.

    Combines descriptive summaries, Mann–Whitney U test, rank-biserial
    correlation, and Cohen's d into a single dictionary.

    Args:
        right_values: Per-seed final metrics for RIGHT, shape ``(n,)``.
        wrong_values: Per-seed final metrics for WRONG, shape ``(m,)``.
        failure_threshold: If provided, compute empirical failure rates for
            both methods (fraction of seeds where metric > threshold).

    Returns:
        Dictionary with keys:

        - ``right_summary``: from :func:`summarize_sample`
        - ``wrong_summary``: from :func:`summarize_sample`
        - ``mann_whitney``: from :func:`mann_whitney_comparison`
        - ``rank_biserial``: ``float``
        - ``cohens_d``: ``float``
        - ``right_failure_rate``: ``float`` *(only when failure_threshold provided)*
        - ``wrong_failure_rate``: ``float`` *(only when failure_threshold provided)*
    """
    out: dict = {
        "right_summary": summarize_sample(right_values),
        "wrong_summary": summarize_sample(wrong_values),
        "mann_whitney": mann_whitney_comparison(right_values, wrong_values),
        "rank_biserial": rank_biserial_correlation(right_values, wrong_values),
        "cohens_d": cohens_d_right_better(right_values, wrong_values),
    }
    if failure_threshold is not None:
        out["right_failure_rate"] = failure_rate(right_values, failure_threshold)
        out["wrong_failure_rate"] = failure_rate(wrong_values, failure_threshold)
    return out


def aggregate_convergence_curves(
    method_to_curves: dict[str, ArrayLike],
    statistic: str = "median",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    band_type: str = "pointwise",
    random_state: Optional[int] = 0,
) -> dict[str, dict]:
    """Compute bootstrap CI curves for multiple methods simultaneously.

    ``band_type`` controls which kind of band is returned:

    - ``"pointwise"``: per-time-step quantile band from
      :func:`bootstrap_ci_curves`.  Each method returns
      ``{"center", "lower", "upper", "band_type": "pointwise"}``.
    - ``"uniform"``: simultaneous (Kolmogorov-style) band around the chosen
      ``statistic`` from :func:`bootstrap_uniform_band`.  Each method returns
      ``{"center", "lower", "upper", "q", "band_type": "uniform"}``.
    - ``"both"``: emits both bands.  Each method returns
      ``{"pointwise": {...}, "uniform": {...}}``.

    Args:
        method_to_curves: Mapping from method name to an array of shape
            ``(n_runs, n_steps)``.  All arrays must have the same ``n_steps``.
        statistic: ``"mean"`` or ``"median"``.  Used as the central statistic
            for **both** the pointwise and the uniform band.
        confidence: CI level.
        n_bootstrap: Number of bootstrap resamples.
        band_type: ``"pointwise"`` | ``"uniform"`` | ``"both"``.
        random_state: Optional integer seed for both bootstraps.

    Returns:
        Mapping from method name to either a single band dict or to
        ``{"pointwise", "uniform"}`` depending on ``band_type``.  Arrays
        are ``numpy.ndarray`` (call ``.tolist()`` before JSON serialisation).
    """
    if band_type not in ("pointwise", "uniform", "both"):
        raise ValueError(
            f"band_type must be 'pointwise', 'uniform' or 'both', got '{band_type}'"
        )

    out: dict[str, dict] = {}
    for name, curves in method_to_curves.items():
        pw = None
        uf = None
        if band_type in ("pointwise", "both"):
            pw_raw = bootstrap_ci_curves(
                curves,
                statistic=statistic,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            pw = {
                "center":     pw_raw["center"],
                "lower":      pw_raw["lower"],
                "upper":      pw_raw["upper"],
                "band_type":  "pointwise",
                "statistic":  statistic,
                "confidence": float(confidence),
            }
        if band_type in ("uniform", "both"):
            uf_raw = bootstrap_uniform_band(
                curves,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=random_state,
                statistic=statistic,
            )
            uf = {
                "center":     uf_raw["center"],
                "lower":      uf_raw["lower"],
                "upper":      uf_raw["upper"],
                "band_type":  "uniform",
                "statistic":  statistic,
                "confidence": float(confidence),
                "q":          uf_raw["q"],
            }
        if band_type == "pointwise":
            out[name] = pw  # type: ignore[assignment]
        elif band_type == "uniform":
            out[name] = uf  # type: ignore[assignment]
        else:
            out[name] = {"pointwise": pw, "uniform": uf}
    return out



def paired_bootstrap_difference(right, wrong, *, n_bootstrap=5000, seed=0):
    """95% percentile CI for mean(RIGHT - WRONG), matched by seed row.

    Arrays may be (seed, angle). Complete seed blocks are resampled; angles
    are averaged within each block and never treated as independent samples.
    Negative differences favor RIGHT for regret metrics. Not an IQR band.
    """
    r, w = np.asarray(right, dtype=float), np.asarray(wrong, dtype=float)
    if r.shape != w.shape or r.ndim not in (1, 2) or not r.size:
        raise ValueError("Matched nonempty seed arrays required")
    if not np.isfinite(r).all() or not np.isfinite(w).all() or n_bootstrap < 1:
        raise ValueError("Finite paired samples and positive replication count required")
    differences = r-w
    if r.ndim == 2:
        differences = differences.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(r), (n_bootstrap, len(r)))
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [.025, .975])
    return dict(estimate=float(differences.mean()), lower=float(lower), upper=float(upper),
                confidence=.95, n_bootstrap=n_bootstrap, n_seeds=len(r), seed=seed,
                statistic="mean paired difference RIGHT - WRONG",
                resampling_unit="complete seed block", identifiable=len(r) > 1)
