"""Noiseless query diagnostics in MINIMIZATION coordinates.

Normalization is fixed by the caller, never estimated from query trajectories.
IGD uses a nondominated archive; unlike hypervolume, it is not universally
monotone when previously archived points become dominated and are removed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def normalize_objectives(values, scales):
    values = np.asarray(values, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if values.ndim != 2 or scales.shape != (values.shape[1],):
        raise ValueError("Expected (n, m) values and (m,) fixed scales")
    if not np.isfinite(values).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Finite values and positive finite scales required")
    return values / scales


def arc_length_reference(front, scales, resolution=1001):
    """Resample an ordered dense analytic front uniformly in normalized arc length."""
    front = normalize_objectives(front, scales)
    if resolution < 2 or len(front) < 2:
        raise ValueError("At least two reference points required")
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(front, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 0]
    if distance[-1] == 0:
        raise ValueError("Reference front has zero arc length")
    grid = np.linspace(0, distance[-1], resolution)
    return np.column_stack([np.interp(grid, distance[keep], front[keep, j])
                            for j in range(front.shape[1])])


def dominance_regret(raw_y, scales, *, reference_front=None, analytical_front=None):
    """min_p max_j ((y_j-p_j)/scale_j)_+ for every query.

    ZDT1/ZDT2 use bisection on their continuous decreasing analytical front.
    Other fronts require a shared dense reference in raw minimization units.
    """
    y = normalize_objectives(raw_y, scales)
    if analytical_front in ("zdt1", "zdt2"):
        scales = np.asarray(scales)
        lo, hi = np.zeros(len(y)), np.ones(len(y))
        def h(x):
            return 1 - (np.sqrt(x) if analytical_front == "zdt1" else x*x)
        for _ in range(60):
            x = (lo + hi) / 2
            left = y[:, 0] - x / scales[0]
            right = y[:, 1] - h(x) / scales[1]
            lo = np.where(left > right, x, lo)
            hi = np.where(left > right, hi, x)
        x = (lo + hi) / 2
        interior = np.maximum(y[:, 0]-x/scales[0], y[:, 1]-h(x)/scales[1])
        endpoints = np.minimum(np.maximum(y[:, 0], y[:, 1]-1/scales[1]),
                               np.maximum(y[:, 0]-1/scales[0], y[:, 1]))
        return np.maximum(np.minimum(interior, endpoints), 0)
    if reference_front is None:
        raise ValueError("An analytical or shared reference front is required")
    front = normalize_objectives(reference_front, scales)
    if not len(front):
        raise ValueError("Empty reference front")
    return np.array([np.maximum(point-front, 0).max(axis=1).min() for point in y])


def query_dominance_curves(regret, tolerance=1e-8):
    """Include initial queries; prefix sums follow the stored global budget order.

    Dominated fraction means positive additive dominance distance > tolerance
    in normalized units (weak domination with zero distance is not counted).
    """
    r = np.asarray(regret, dtype=float)
    if r.ndim != 1 or (r < 0).any() or not np.isfinite(r).all() or tolerance < 0:
        raise ValueError("Nonnegative finite regret and tolerance required")
    t = np.arange(1, len(r)+1)
    cumulative = np.cumsum(r)
    return dict(dominance_regret=r, cumulative_dominance_regret=cumulative,
                average_dominance_regret=cumulative/t,
                dominated_query_fraction=np.cumsum(r > tolerance)/t)


def running_igd(raw_y, scales, normalized_reference):
    """IGD of each accumulated nondominated archive, initial queries included.

    The reference is already normalized and shared. Archive updates preserve
    the caller's budget order; no chronological ordering of subruns is inferred.
    """
    y = normalize_objectives(raw_y, scales)
    reference = np.asarray(normalized_reference, dtype=float)
    if reference.ndim != 2 or not len(reference) or reference.shape[1] != y.shape[1] or not np.isfinite(reference).all():
        raise ValueError("Finite nonempty reference of matching dimension required")
    archive = np.empty((0, y.shape[1]))
    out = []
    for point in y:
        dominated = len(archive) and np.any(np.all(archive <= point, axis=1))
        if not dominated:
            archive = archive[~np.all(point <= archive, axis=1)]
            archive = np.vstack([archive, point])
        out.append(cdist(reference, archive).min(axis=1).mean())
    return np.asarray(out)


def incumbent_indices(values):
    """Indices of strict scalarized MAXIMIZATION improvements in query order."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Nonempty finite scalarized trajectory required")
    previous = np.r_[-np.inf, np.maximum.accumulate(values)[:-1]]
    return np.flatnonzero(values > previous)
