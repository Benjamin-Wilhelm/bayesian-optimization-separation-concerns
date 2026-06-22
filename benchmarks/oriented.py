"""Orientation-only maximization view of synthetic benchmarks (Experiment B).

Wraps an existing :class:`~benchmarks.synthetic.SyntheticBenchmark` so that
the problem is always a **maximization** problem while keeping the objective
on the benchmark's **natural (raw) scale**.

This deliberately follows the Experiment A philosophy: the benchmark is
evaluated and the regret is measured on the raw function — the gaps between
the global optimum and the local optima are preserved, never squashed into
``[0, 1]``.  The only transform applied here is a sign flip that turns a
minimization benchmark into a maximization one::

    minimization benchmark (e.g. Branin):   g(x) = - f_raw(x)
    maximization benchmark:                  g(x) = + f_raw(x)

``g`` is the objective the BO loop sees (UCB maximizes).  Its optimum is::

    f_opt = max_x g(x)
          = - min_x f_raw(x)   (minimization benchmark)
          = + max_x f_raw(x)   (maximization benchmark)

and the simple regret used by Experiment B is the standard maximization
regret ``f_opt - g(x_t)`` on this raw scale.

.. note::

    Do **not** confuse this with input normalization.  The GP / BoTorch may
    still apply ``Normalize(X)`` to the inputs (see
    :class:`common.priors.PriorGPModelBuilder`); what this wrapper guarantees
    is that the *objective* is never replaced by a ``[0, 1]`` rescaled
    version.

Three distinct objects in Experiment B (kept explicitly separate):

- ``pi(x) in [0, 1]``  — the spatial Gaussian prior, used as an acquisition
  weight (:class:`common.priors.GaussianPrior`).  Dimensionless.
- ``f_prior(x)``       — a prior **mean** function living on the ``y`` scale,
  ``f_prior(x) = baseline + amplitude * pi(x)`` (see
  :meth:`OrientedBenchmark.prior_mean_scale`).  Used only by the
  prior-to-the-mean methods.
- ``f(x) = g(x)``      — this raw oriented objective.  Used to observe values
  and to measure regret.

Noise convention: ``noise_std`` passed to the wrapper is on the raw oriented
scale (the inner benchmark is constructed noiseless and the wrapper's
inherited ``_observe_prepared`` adds Gaussian noise to ``g``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from benchmarks import BENCHMARK_REGISTRY, _benchmark_section
from benchmarks.synthetic import SyntheticBenchmark


# ---------------------------------------------------------------------------
# Known raw output ranges
# ---------------------------------------------------------------------------
#
# (f_min, f_max, direction) of the **raw** inner benchmark over its *default*
# bounds.
#
# - ``f_min`` is the exact analytic global minimum where known.
# - ``f_max`` is a numerical estimate computed on dense grids / 4e6 Sobol
#   points (2026-06-10, torch.double):
#       forrester_1d : max at x = 1,  f(1) = 16 sin(8)  ~= 15.8297319
#       branin_2d    : max at corner (-5, 0)            ~= 308.1290960
#       hartmann_3d  : f < 0 on [0,1]^3; supremum bounded above by 0.0
#       hartmann_6d  : f < 0 on [0,1]^6; supremum bounded above by 0.0
# - ``direction`` is the goal of the inner benchmark ("min" or "max"); it
#   selects the sign flip applied to obtain the maximization objective.
KNOWN_OUTPUT_RANGES: dict[str, tuple[float, float, str]] = {
    "forrester_1d": (-6.020740055735769, 15.829731945974109, "min"),
    "branin_2d":    (0.39788735772973816, 308.12909601160663, "min"),
    "hartmann_3d":  (-3.86278214782076,   0.0,                "min"),
    "hartmann_6d":  (-3.32236801141551,   0.0,                "min"),
}


class OrientedBenchmark(SyntheticBenchmark):
    """Maximization view of a synthetic benchmark, on its raw scale.

    The wrapper owns the observation noise (``noise_std`` is on the raw
    oriented scale) and delegates noiseless evaluation to the inner
    benchmark, applying only the orientation sign flip.  Bounds, dimension,
    and test-set generation are inherited from the wrapped benchmark.

    Args:
        inner: Wrapped benchmark.  Should be constructed **noiseless**; any
            ``default_noise_std`` on the inner benchmark is ignored because
            this wrapper never calls ``inner.observe``.
        f_min: Minimum of the inner **raw** noiseless function over the
            domain.
        f_max: Maximum of the inner **raw** noiseless function over the
            domain.
        direction: ``"min"`` if the wrapped benchmark is a minimization
            problem (so ``g = -f_raw``), ``"max"`` otherwise (``g = f_raw``).
        noise_std: Observation noise standard deviation on the raw oriented
            scale.
    """

    def __init__(
        self,
        inner: SyntheticBenchmark,
        f_min: float,
        f_max: float,
        direction: str = "min",
        noise_std: float = 0.0,
        ) -> None:
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got '{direction}'.")
        if not float(f_max) > float(f_min):
            raise ValueError(
                f"f_max must be strictly greater than f_min, got "
                f"f_min={f_min}, f_max={f_max}."
            )
        self.inner = inner
        # Raw inner extrema over the domain.
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.direction = direction
        self._sign = -1.0 if direction == "min" else 1.0

        # Oriented (maximization) range of g = sign * f_raw.  The objective is
        # NOT rescaled; this is purely the sign-flipped raw range, used to
        # build the prior-mean function f_prior on the y scale.
        if direction == "min":
            self.f_lower = -self.f_max   # g_min
            self.f_upper = -self.f_min   # g_max = f_opt
        else:
            self.f_lower = self.f_min
            self.f_upper = self.f_max

        super().__init__(
            name=inner.name,
            dim=inner.dim,
            bounds=inner.bounds,
            default_noise_std=noise_std,
        )

    # ------------------------------------------------------------------
    # Noiseless oriented function (raw scale)
    # ------------------------------------------------------------------

    def f(self, X: torch.Tensor) -> torch.Tensor:
        """Oriented noiseless objective ``g(X) = sign * f_raw(X)``, shape ``(n, 1)``."""
        return self._sign * self.inner.f(X)

    # ------------------------------------------------------------------
    # Oracle for standard BO metrics (maximization convention)
    # ------------------------------------------------------------------

    @property
    def f_opt(self) -> float:
        """Maximum of the oriented objective ``g`` over the domain (raw scale).

        Equals ``-f_min`` for a minimization benchmark and ``f_max`` for a
        maximization one, provided the extremal value used is the exact
        analytic optimum of the wrapped benchmark.
        """
        return self.f_upper

    def prior_mean_scale(self) -> tuple[float, float]:
        """``(baseline, amplitude)`` mapping ``pi in [0, 1]`` to the ``y`` scale.

        Default prior-mean function on the objective scale::

            f_prior(x) = baseline + amplitude * pi(x)
                       = f_lower + (f_upper - f_lower) * pi(x)

        so that ``f_prior`` spans the oriented range of the objective and
        peaks at the optimum value ``f_opt`` where ``pi(x) = 1``.  The
        prior-to-the-mean methods use this; the spatial weight ``pi`` itself
        stays in ``[0, 1]``.
        """
        return self.f_lower, self.f_upper - self.f_lower

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta["orientation"] = {
            "base_benchmark": self.inner.name,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "direction": self.direction,
            "f_lower": self.f_lower,
            "f_upper": self.f_upper,
            "f_opt": self.f_opt,
        }
        return meta


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def build_oriented_benchmark(config: Mapping[str, Any]) -> OrientedBenchmark:
    """Construct an :class:`OrientedBenchmark` from a top-level YAML config.

    Layout (both ``benchmark`` layouts of :func:`benchmarks.build_benchmark`
    are supported; the nested form is shown)::

        benchmark:
          name: forrester_1d
          # optional intrinsic keys / bounds of the inner benchmark
        noise_std: 0.5             # raw oriented scale (top level)
        orientation:               # optional; defaults from KNOWN_OUTPUT_RANGES
          f_min: -6.020740055735769   # raw inner minimum over the domain
          f_max: 15.829731945974109   # raw inner maximum over the domain
          direction: min

    The inner benchmark is always constructed with ``noise_std = 0``; the
    observation noise lives on the wrapper, on the raw oriented scale.

    ``orientation.f_min`` / ``orientation.f_max`` are the **raw** inner
    extrema; the wrapper derives the oriented range (and ``f_opt``) from them
    via the sign flip.
    """
    name, sub = _benchmark_section(config)
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'.  Known: {sorted(BENCHMARK_REGISTRY)}"
        )

    # The wrapper owns the noise; the inner benchmark is noiseless.  A
    # noise_std accidentally placed inside the benchmark sub-config would be
    # silently re-interpreted, so reject it explicitly.
    if sub.get("noise_std") not in (None, 0, 0.0):
        raise ValueError(
            "For oriented benchmarks, put 'noise_std' at the top level of "
            "the config (raw oriented scale), not inside the 'benchmark' "
            "section."
        )
    sub = dict(sub)
    sub["noise_std"] = 0.0
    inner = BENCHMARK_REGISTRY[name].from_config(sub)

    # Raw-range resolution order (an ``orientation`` section is preferred; a
    # legacy ``normalization`` section is still accepted since it carried the
    # same {f_min, f_max, direction} keys):
    #   1. explicit ``orientation:`` config keys;
    #   2. a range provided by the benchmark itself via ``output_range()``
    #      (parameterizable benchmarks, e.g. DoubleGaussianWell1D, compute
    #      their raw range from their actual geometry);
    #   3. the hard-coded ``KNOWN_OUTPUT_RANGES`` constants.
    orient_cfg: Mapping[str, Any] = (
        config.get("orientation") or config.get("normalization") or {}
    )
    f_min = orient_cfg.get("f_min")
    f_max = orient_cfg.get("f_max")
    direction = orient_cfg.get("direction")
    if f_min is None or f_max is None or direction is None:
        own_range = getattr(inner, "output_range", None)
        if callable(own_range):
            fallback = own_range()
        else:
            # KNOWN_OUTPUT_RANGES constants are computed on each benchmark's
            # DEFAULT domain; with custom bounds they no longer hold, so the
            # caller must supply an explicit range.
            if sub.get("bounds") is not None:
                raise ValueError(
                    f"Benchmark '{name}' uses custom bounds, so the default "
                    "KNOWN_OUTPUT_RANGES constants do not apply.  Provide an "
                    "explicit 'orientation: {f_min, f_max, direction}' "
                    "section (raw inner extrema) computed on those bounds."
                )
            fallback = KNOWN_OUTPUT_RANGES.get(name)
        if fallback is None:
            raise ValueError(
                f"No known output range for benchmark '{name}'.  Provide "
                "'orientation: {f_min, f_max, direction}' explicitly in "
                "the config."
            )
        f_min = fallback[0] if f_min is None else f_min
        f_max = fallback[1] if f_max is None else f_max
        direction = fallback[2] if direction is None else direction

    noise_std = float(config.get("noise_std") or 0.0)

    return OrientedBenchmark(
        inner=inner,
        f_min=float(f_min),
        f_max=float(f_max),
        direction=str(direction),
        noise_std=noise_std,
    )


__all__ = [
    "KNOWN_OUTPUT_RANGES",
    "OrientedBenchmark",
    "build_oriented_benchmark",
]
