"""Output-normalized synthetic benchmarks (Experiment B).

Wraps an existing :class:`~benchmarks.synthetic.SyntheticBenchmark` so that
its noiseless output lives on a common ``[0, 1]`` scale and the problem is
always a **maximization** problem:

For a minimization benchmark (e.g. Branin)::

    f_norm(x) = 1 - (f(x) - f_min) / (f_max - f_min)

For a maximization benchmark::

    f_norm(x) = (f(x) - f_min) / (f_max - f_min)

In both cases the global optimum of the wrapped problem maps to
``f_norm = 1.0`` exactly, provided the corresponding extremal value
(``f_min`` for minimization, ``f_max`` for maximization) is the exact
analytic optimum.  The normalized oracle value is therefore::

    f_opt_normalized = 1.0

Backward compatibility: this module is purely additive.  Experiment A keeps
constructing raw benchmarks through :func:`benchmarks.build_benchmark`; the
wrapper is only used by Experiment B via :func:`build_normalized_benchmark`.

Noise convention: ``noise_std`` passed to the wrapper is interpreted on the
**normalized** scale (the inner benchmark is constructed noiseless and the
wrapper's inherited ``_observe_prepared`` adds Gaussian noise to ``f_norm``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from benchmarks import BENCHMARK_REGISTRY, _benchmark_section
from benchmarks.synthetic import SyntheticBenchmark


# ---------------------------------------------------------------------------
# Known output ranges
# ---------------------------------------------------------------------------
#
# (f_min, f_max, direction) over each benchmark's *default* bounds.
#
# - ``f_min`` is the exact analytic global minimum where known, so that the
#   optimum of the normalized problem is exactly 1.0.
# - ``f_max`` is a numerical estimate computed on dense grids / 4e6 Sobol
#   points (2026-06-10, torch.double):
#       forrester_1d : max at x = 1,  f(1) = 16 sin(8)  ≈ 15.8297319
#       branin_2d    : max at corner (-5, 0)            ≈ 308.1290960
#       hartmann_3d  : f < 0 on [0,1]^3; supremum bounded above by 0.0
#       hartmann_6d  : f < 0 on [0,1]^6; supremum bounded above by 0.0
# - ``direction`` is the goal of the *wrapped* benchmark ("min" or "max").
KNOWN_OUTPUT_RANGES: dict[str, tuple[float, float, str]] = {
    "forrester_1d": (-6.020740055735769, 15.829731945974109, "min"),
    "branin_2d":    (0.39788735772973816, 308.12909601160663, "min"),
    "hartmann_3d":  (-3.86278214782076,   0.0,                "min"),
    "hartmann_6d":  (-3.32236801141551,   0.0,                "min"),
}


class NormalizedBenchmark(SyntheticBenchmark):
    """Maximization view of a synthetic benchmark with outputs in ``[0, 1]``.

    The wrapper owns the observation noise (``noise_std`` is on the
    normalized scale) and delegates noiseless evaluation to the inner
    benchmark.  Bounds, dimension, and test-set generation are inherited
    from the wrapped benchmark's domain.

    Args:
        inner: Wrapped benchmark.  Should be constructed **noiseless**; any
            ``default_noise_std`` on the inner benchmark is ignored because
            this wrapper never calls ``inner.observe``.
        f_min: Minimum of the inner noiseless function over the domain.
        f_max: Maximum of the inner noiseless function over the domain.
        direction: ``"min"`` if the wrapped benchmark is a minimization
            problem (its minimizer maps to ``f_norm = 1``), ``"max"``
            otherwise.
        noise_std: Observation noise standard deviation on the normalized
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
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.direction = direction

        super().__init__(
            name=f"{inner.name}_normalized",
            dim=inner.dim,
            bounds=inner.bounds,
            default_noise_std=noise_std,
        )

    # ------------------------------------------------------------------
    # Noiseless function
    # ------------------------------------------------------------------

    def f(self, X: torch.Tensor) -> torch.Tensor:
        """Normalized noiseless value ``f_norm(X)``, shape ``(n, 1)``."""
        f_raw = self.inner.f(X)
        u = (f_raw - self.f_min) / (self.f_max - self.f_min)
        if self.direction == "min":
            return 1.0 - u
        return u

    # ------------------------------------------------------------------
    # Oracle for standard BO metrics (maximization convention)
    # ------------------------------------------------------------------

    @property
    def f_opt_normalized(self) -> float:
        """Maximum of ``f_norm`` over the domain.

        Exactly ``1.0`` when the extremal value used for normalization
        (``f_min`` for ``direction='min'``, ``f_max`` for ``'max'``) is the
        exact optimum of the wrapped benchmark.
        """
        return 1.0

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta["normalization"] = {
            "base_benchmark": self.inner.name,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "direction": self.direction,
        }
        return meta


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def build_normalized_benchmark(config: Mapping[str, Any]) -> NormalizedBenchmark:
    """Construct a :class:`NormalizedBenchmark` from a top-level YAML config.

    Layout (both ``benchmark`` layouts of :func:`benchmarks.build_benchmark`
    are supported; the nested form is shown)::

        benchmark:
          name: forrester_1d
          # optional intrinsic keys / bounds of the inner benchmark
        noise_std: 0.01            # normalized scale (top level)
        normalization:             # optional; defaults from KNOWN_OUTPUT_RANGES
          f_min: -6.020740055735769
          f_max: 15.829731945974109
          direction: min

    The inner benchmark is always constructed with ``noise_std = 0``; the
    observation noise lives on the wrapper, on the normalized scale.
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
            "For normalized benchmarks, put 'noise_std' at the top level of "
            "the config (normalized scale), not inside the 'benchmark' "
            "section."
        )
    sub = dict(sub)
    sub["noise_std"] = 0.0
    inner = BENCHMARK_REGISTRY[name].from_config(sub)

    # Output-range resolution order:
    #   1. explicit ``normalization:`` config keys;
    #   2. a range provided by the benchmark itself via ``output_range()``
    #      (parameterizable benchmarks, e.g. DoubleGaussianWell1D, compute
    #      their range from their actual geometry);
    #   3. the hard-coded ``KNOWN_OUTPUT_RANGES`` constants.
    norm_cfg: Mapping[str, Any] = config.get("normalization") or {}
    f_min = norm_cfg.get("f_min")
    f_max = norm_cfg.get("f_max")
    direction = norm_cfg.get("direction")
    if f_min is None or f_max is None or direction is None:
        own_range = getattr(inner, "output_range", None)
        fallback = own_range() if callable(own_range) else KNOWN_OUTPUT_RANGES.get(name)
        if fallback is None:
            raise ValueError(
                f"No known output range for benchmark '{name}'.  Provide "
                "'normalization: {f_min, f_max, direction}' explicitly in "
                "the config."
            )
        f_min = fallback[0] if f_min is None else f_min
        f_max = fallback[1] if f_max is None else f_max
        direction = fallback[2] if direction is None else direction

    noise_std = float(config.get("noise_std") or 0.0)

    return NormalizedBenchmark(
        inner=inner,
        f_min=float(f_min),
        f_max=float(f_max),
        direction=str(direction),
        noise_std=noise_std,
    )


__all__ = [
    "KNOWN_OUTPUT_RANGES",
    "NormalizedBenchmark",
    "build_normalized_benchmark",
]
