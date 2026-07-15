"""Multi-objective synthetic benchmarks for Experiment C (ZDT family).

These benchmarks are **multi-output** and therefore do not fit the
single-output :class:`common.problems.Observation` / ProblemSpec abstraction
used by Experiments A and B.  A small self-contained interface is defined here
instead, kept intentionally close to the shared conventions (``torch.double``
everywhere, ``bounds`` of shape ``(2, dim)``, a benchmark registry, a
``build_benchmark`` YAML entry point).

Orientation and representation conventions (centralised here, never re-coded
in the methods or metrics):

``evaluate_raw``
    ZDT standard **minimization** convention, shape ``(n, m)``.  This is the
    convention in which the analytic Pareto front is documented, and the one
    used for the *raw* objective-space figures (``lower is better``).

``evaluate_oriented``
    **Maximization** convention ``F = -f_raw``.  The whole optimization
    pipeline (GPs, acquisitions, scalarizations, hypervolume, dominance)
    operates in this convention.

``evaluate_model``
    Values actually fed to the GPs.  In this first version this equals
    ``evaluate_oriented`` (no extra fixed transform besides orientation),
    but the separate name keeps room for a future fixed normalization.

The hypervolume reference point is a **fixed** point in the oriented
convention (``z = (-1.1, -1.1)`` for the ZDT problems), dominated by the whole
true Pareto front (which lies in ``[-1, 0]^2`` after orientation) but not
necessarily by every dominated point of the domain.  It never depends on the
observed trajectory.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MultiObjectiveBenchmark:
    """Abstract base for synthetic multi-objective benchmarks.

    Subclasses implement :meth:`evaluate_raw` (minimization convention) and
    the analytic ``true_pareto_*`` accessors.  Everything else — orientation,
    the model view, metadata — is shared.

    Attributes:
        name: Benchmark identifier (matches the registry key).
        input_dim: Decision-space dimension ``d``.
        output_dim: Number of objectives ``m``.
        bounds: Box bounds, shape ``(2, input_dim)``.
        reference_point: Fixed hypervolume reference in the **oriented**
            (maximization) convention, shape ``(output_dim,)``.
    """

    name: str = "abstract_mo"
    output_dim: int = 2

    def __init__(
        self,
        input_dim: int,
        bounds: Optional[Tensor] = None,
        reference_point: Optional[Tensor] = None,
    ) -> None:
        if input_dim < 2:
            raise ValueError(f"input_dim must be >= 2, got {input_dim}.")
        self.input_dim = int(input_dim)
        if bounds is None:
            bounds = torch.stack(
                [torch.zeros(self.input_dim), torch.ones(self.input_dim)]
            )
        bounds = torch.as_tensor(bounds, dtype=torch.double)
        if bounds.shape != (2, self.input_dim):
            raise ValueError(
                f"bounds must have shape (2, input_dim)={(2, self.input_dim)}, "
                f"got {tuple(bounds.shape)}."
            )
        if not torch.all(bounds[0] < bounds[1]):
            raise ValueError("bounds must satisfy lower < upper componentwise.")
        self.bounds = bounds
        if reference_point is None:
            reference_point = self.default_reference_point()
        self.reference_point = torch.as_tensor(
            reference_point, dtype=torch.double
        ).reshape(-1)
        if self.reference_point.numel() != self.output_dim:
            raise ValueError(
                f"reference_point must have {self.output_dim} entries, "
                f"got {self.reference_point.numel()}."
            )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def evaluate_raw(self, X: Tensor) -> Tensor:
        """Noiseless objectives, ZDT minimization convention, shape ``(n, m)``."""
        raise NotImplementedError

    def true_pareto_set(self, n_points: int) -> Tensor:
        """Analytic Pareto set in decision space, shape ``(n_points, d)``."""
        raise NotImplementedError

    def default_reference_point(self) -> Tensor:
        """Fixed oriented reference point; overridable per benchmark."""
        return torch.full((self.output_dim,), -1.1, dtype=torch.double)

    # ------------------------------------------------------------------
    # Shared orientation logic
    # ------------------------------------------------------------------

    def prepare_X(self, X: Tensor) -> Tensor:
        """Validate/normalise decision points to ``(n, d)`` double."""
        X = torch.as_tensor(X, dtype=torch.double)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"X last dimension must be {self.input_dim}, got {X.shape[-1]}."
            )
        return X

    def evaluate_oriented(self, X: Tensor) -> Tensor:
        """Maximization objectives ``F = -f_raw``, shape ``(n, m)``."""
        return -self.evaluate_raw(X)

    def evaluate_model(self, X: Tensor) -> Tensor:
        """Values fed to the GPs.  Equals :meth:`evaluate_oriented` (v1)."""
        return self.evaluate_oriented(X)

    def observe(self, X: Tensor, noise_std: float = 0.0) -> dict[str, Tensor]:
        """Return the three objective views at ``X`` (noiseless + GP targets).

        - ``raw_y``: noiseless ZDT **minimization** values (for raw figures).
        - ``oriented_y``: noiseless **maximization** values ``-raw_y`` (used by
          the evaluation metrics — synthetic ground truth).
        - ``model_y``: values actually fed to the GPs, i.e. ``oriented_y`` plus
          objective-wise Gaussian noise of std ``noise_std``.  With
          ``noise_std == 0`` this equals ``oriented_y``.

        Only ``model_y`` is ever given to the BO algorithm; ``oriented_y`` is
        the noiseless synthetic quantity reserved for metrics/diagnostics.

        Args:
            X: Decision points, shape ``(n, d)`` or ``(d,)``.
            noise_std: Per-objective i.i.d. Gaussian noise std (oriented scale).
        """
        X = self.prepare_X(X)
        raw = self.evaluate_raw(X)
        oriented_y = -raw
        if noise_std and noise_std > 0.0:
            model_y = oriented_y + torch.randn_like(oriented_y) * float(noise_std)
        else:
            model_y = oriented_y.clone()
        return {
            "raw_y": raw,
            "oriented_y": oriented_y,
            "model_y": model_y,
        }

    # ------------------------------------------------------------------
    # True front (analytic, dense, deterministic)
    # ------------------------------------------------------------------

    def true_pareto_front_raw(self, n_points: int) -> Tensor:
        """Dense analytic Pareto front, ZDT minimization convention ``(n, m)``."""
        return self.evaluate_raw(self.true_pareto_set(n_points))

    def true_pareto_front_oriented(self, n_points: int) -> Tensor:
        """Dense analytic Pareto front, oriented (maximization) ``(n, m)``."""
        return -self.true_pareto_front_raw(n_points)

    # backward-friendly aliases matching the task's suggested interface
    def true_pareto_front(self, n_points: int) -> Tensor:
        """Alias for :meth:`true_pareto_front_oriented`."""
        return self.true_pareto_front_oriented(n_points)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def normalization_metadata(self) -> dict[str, Any]:
        """Fixed, trajectory-independent orientation / normalization record."""
        return {
            "orientation": "maximize",
            "transform": "F = -f_raw",
            "model_transform": "model_y = oriented_y (no extra fixed transform)",
            "normalization": "none (ZDT objectives already O(1) on the front)",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "bounds": self.bounds.tolist(),
            "reference_point": self.reference_point.tolist(),
            "normalization_metadata": self.normalization_metadata(),
        }


# ---------------------------------------------------------------------------
# ZDT helpers
# ---------------------------------------------------------------------------

def _zdt_g(X: Tensor) -> Tensor:
    """ZDT ``g(x) = 1 + 9/(d-1) * sum_{i>=2} x_i``, shape ``(n, 1)``."""
    d = X.shape[-1]
    tail = X[..., 1:]
    return 1.0 + 9.0 / (d - 1) * tail.sum(dim=-1, keepdim=True)


class ZDT1(MultiObjectiveBenchmark):
    r"""ZDT1: convex Pareto front, fully reachable by linear scalarization.

    Minimization convention::

        f1(x) = x_1
        g(x)  = 1 + 9/(d-1) * sum_{i>=2} x_i
        f2(x) = g(x) * (1 - sqrt(f1 / g))

    Pareto set: ``x_2 = ... = x_d = 0``; Pareto front: ``f2 = 1 - sqrt(f1)``,
    ``f1 in [0, 1]``.  Because the front is convex, every point is a
    *supported* solution — reachable by maximizing some positive weighted sum.
    """

    name = "zdt1"
    output_dim = 2

    def evaluate_raw(self, X: Tensor) -> Tensor:
        X = self.prepare_X(X)
        f1 = X[..., :1]
        g = _zdt_g(X)
        f2 = g * (1.0 - torch.sqrt(f1 / g))
        return torch.cat([f1, f2], dim=-1)

    def true_pareto_set(self, n_points: int) -> Tensor:
        x1 = torch.linspace(0.0, 1.0, n_points, dtype=torch.double).unsqueeze(-1)
        rest = torch.zeros((n_points, self.input_dim - 1), dtype=torch.double)
        return torch.cat([x1, rest], dim=-1)


class ZDT2(MultiObjectiveBenchmark):
    r"""ZDT2: concave Pareto front with **non-supported** points.

    Minimization convention (same ``f1`` and ``g`` as ZDT1)::

        f2(x) = g(x) * (1 - (f1 / g) ** 2)

    Pareto front: ``f2 = 1 - f1^2``.  The front is concave, so an interior
    region of it is *non-supported*: no strictly-positive weighted sum has its
    maximum there.  A failure of the linear scalarization to reach that region
    is a limitation of the scalarization family, **not** of RIGHT vs WRONG.
    """

    name = "zdt2"
    output_dim = 2

    def evaluate_raw(self, X: Tensor) -> Tensor:
        X = self.prepare_X(X)
        f1 = X[..., :1]
        g = _zdt_g(X)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return torch.cat([f1, f2], dim=-1)

    def true_pareto_set(self, n_points: int) -> Tensor:
        x1 = torch.linspace(0.0, 1.0, n_points, dtype=torch.double).unsqueeze(-1)
        rest = torch.zeros((n_points, self.input_dim - 1), dtype=torch.double)
        return torch.cat([x1, rest], dim=-1)


# ---------------------------------------------------------------------------
# Registry / YAML construction
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, type[MultiObjectiveBenchmark]] = {
    "zdt1": ZDT1,
    "zdt2": ZDT2,
}


def build_benchmark(config: Mapping[str, Any]) -> MultiObjectiveBenchmark:
    """Construct a benchmark from a ``benchmark:`` sub-config.

    Expected layout::

        benchmark:
          name: zdt1
          input_dim: 2
          orientation: maximize          # informational; enforced here
          normalization: fixed           # informational
          reference_point: [-1.1, -1.1]  # optional override (oriented)
    """
    name = str(config["name"])
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown multi-objective benchmark '{name}'. "
            f"Known: {sorted(BENCHMARK_REGISTRY)}."
        )
    orientation = str(config.get("orientation", "maximize"))
    if orientation != "maximize":
        raise ValueError(
            "Experiment C runs in maximization; set benchmark.orientation to "
            f"'maximize' (got '{orientation}'). The minimization->maximization "
            "flip is applied centrally by the benchmark."
        )
    input_dim = int(config.get("input_dim", 2))
    ref = config.get("reference_point")
    reference_point = (
        torch.as_tensor(ref, dtype=torch.double) if ref is not None else None
    )
    return BENCHMARK_REGISTRY[name](
        input_dim=input_dim, reference_point=reference_point
    )


__all__ = [
    "MultiObjectiveBenchmark",
    "ZDT1",
    "ZDT2",
    "BENCHMARK_REGISTRY",
    "build_benchmark",
]
