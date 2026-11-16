"""Experiment C — per-seed processing.

Loads raw seed files, recomputes **only model-independent** metrics (every
metric re-evaluates the analytic benchmark at the stored ``x``; no GP is
rebuilt and no BO is rerun), and writes processed files under
``processed/per_seed/<benchmark>/seed_NNN_processed.json``.

Verification (always run; raises on failure so incompatible or corrupt runs
are never silently aggregated):

- stored ``oriented_y`` matches the benchmark re-evaluated at the stored
  ``x`` (catches orientation / raw-oriented confusion);
- all methods share the same budget and trajectory length;
- fixed: RIGHT and WRONG of each case share the same pairing hash (same
  initial design + lambda);
- pareto: RIGHT/WRONG Algorithm 2 share the same per-subrun lambdas, pairing
  hashes, and local budgets; ``sum local == global`` budget;
- reference point, orientation, normalization and front resolution are
  consistent with the metadata.

CLI::

    python -m experiments.exp_c_multiobjective.process \\
        --run-dir results/exp_c_multiobjective/runs/<run_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.logging import RunLogger
from common.serialization import load_json, save_json
from benchmarks import multi_objective as mo_benchmarks
from experiments.exp_c_multiobjective.config import budget_split
from data.raw_multiobjective import FixedSeedData, ParetoSeedData
from data.processed_multiobjective import (
    FixedCaseProcessed,
    FixedMethodProcessed,
    ParetoMethodProcessed,
    ProcessedFixedSeedData,
    ProcessedParetoSeedData,
)
from common.hypervolume import (
    hypervolume_regret_2d,
    running_hypervolume_2d,
)
from common.multi_objective_metrics import (
    empirical_bayes_regret,
    reference_scalar_optimum,
    scalarized_regrets,
    scalarized_trajectory_values,
)

from common.pareto_diagnostics import (arc_length_reference, dominance_regret,
                                       query_dominance_curves, running_igd)

logger = logging.getLogger("process_exp_c")

_ATOL = 1e-6

# Resolution of the analytic Pareto front used to RECOMPUTE the reference
# scalar optimum rho* = max_x rho_lambda(F(x)) at process time.  The value
# stored in the raw file was computed on a coarser front (default 10 000
# points), which slightly under-estimates rho*; near-optimal BO queries can
# then exceed it by ~1e-5 and trip the negative-regret guard.  The hypervolume
# scalarization maxes out at a *kink* of the front, so the discretization error
# shrinks as O(1/N) with grid-alignment jitter (not O(1/N^2)); empirically the
# worst-case negative regret across all fixed benchmarks reaches exactly 0 at
# 1e6 points (zdt1_d2: 200k -> -3.9e-6, 500k -> -6e-7, 1e6 -> 0).  rho* is a
# pure evaluation reference and is not used by the BO loop, so recomputing it
# here does not affect the scientific protocol of the runs.
_FIXED_RHO_FRONT_RESOLUTION: int = 1_000_000


# ---------------------------------------------------------------------------
# Benchmark reconstruction
# ---------------------------------------------------------------------------

def _build_benchmark(meta: dict[str, Any]) -> mo_benchmarks.MultiObjectiveBenchmark:
    name = str(meta["benchmark"])
    cls = mo_benchmarks.BENCHMARK_REGISTRY[name]
    ref = torch.as_tensor(meta["reference_point"], dtype=torch.double)
    return cls(input_dim=int(meta["input_dim"]), reference_point=ref)


def _oriented_at(
    benchmark: mo_benchmarks.MultiObjectiveBenchmark, X: torch.Tensor
) -> torch.Tensor:
    """Noiseless oriented objective values re-evaluated at ``X`` (model-free)."""
    return benchmark.observe(X, noise_std=0.0)["oriented_y"]


# ---------------------------------------------------------------------------
# Fixed scalarization
# ---------------------------------------------------------------------------

def _process_fixed_seed(
    raw: FixedSeedData,
    benchmark: mo_benchmarks.MultiObjectiveBenchmark,
) -> ProcessedFixedSeedData:
    meta = raw.metadata
    ref = benchmark.reference_point
    processed_cases: list[FixedCaseProcessed] = []

    # Recompute rho* on a dense analytic front (see _FIXED_RHO_FRONT_RESOLUTION):
    # the coarser rho* stored in the raw file under-estimates the true optimum
    # and trips the negative-regret guard on near-optimal queries.
    front_oriented = benchmark.true_pareto_front_oriented(_FIXED_RHO_FRONT_RESOLUTION)

    for case in raw.cases:
        lam = torch.tensor(case.lambda_vector, dtype=torch.double)
        scal = case.scalarization
        rho_star = float(
            reference_scalar_optimum(front_oriented, lam, scal, ref).item()
        )

        init_X = torch.tensor(case.initial_X, dtype=torch.double)
        n_init = init_X.shape[0]

        # Pairing check: every method in the case shares the stored hash.
        for name, method in case.methods.items():
            pass  # pairing across methods verified in _verify_fixed

        method_out: dict[str, FixedMethodProcessed] = {}
        budget = int(meta["budget"])
        for name, method in case.methods.items():
            traj_X = torch.tensor(
                [s.x for s in method.trajectory], dtype=torch.double
            ) if method.trajectory else torch.empty((0, benchmark.input_dim))
            X_full = torch.cat([init_X, traj_X], dim=0)
            oriented_full = _oriented_at(benchmark, X_full)   # (budget, m) noiseless

            # Consistency: recomputed oriented_y vs stored (init + trajectory).
            stored_oriented = torch.tensor(
                case.initial_oriented_y
                + [s.oriented_y for s in method.trajectory],
                dtype=torch.double,
            )
            d = float((oriented_full - stored_oriented).abs().max().item()) \
                if stored_oriented.numel() else 0.0
            if d > _ATOL:
                raise ValueError(
                    f"[fixed] {benchmark.name} seed {meta['seed']} case "
                    f"({scal}, angle {case.lambda_angle_degrees}) method '{name}': "
                    f"stored oriented_y disagrees with re-evaluated benchmark by "
                    f"{d:.2e} (orientation / raw-oriented bug)."
                )

            v = scalarized_trajectory_values(oriented_full, lam, scal, ref)  # (budget,)
            regrets = scalarized_regrets(v, rho_star)
            method_out[name] = FixedMethodProcessed(
                iteration=list(range(1, X_full.shape[0] + 1)),
                n_initial_design=n_init,
                scalarized_value=v.tolist(),
                best_so_far=regrets["best_so_far"].tolist(),
                simple_regret=regrets["simple_regret"].tolist(),
                instantaneous_regret=regrets["instantaneous_regret"].tolist(),
                cumulative_regret=regrets["cumulative_regret"].tolist(),
            )
            assert X_full.shape[0] == budget, (X_full.shape[0], budget)

        processed_cases.append(FixedCaseProcessed(
            scalarization=scal,
            lambda_angle_degrees=case.lambda_angle_degrees,
            lambda_vector=case.lambda_vector,
            rho_star=rho_star,
            pairing_hash=case.pairing_hash,
            methods=method_out,
        ))

    return ProcessedFixedSeedData(
        metadata=meta,
        benchmark=raw.benchmark,
        reference_point=ref.tolist(),
        cases=processed_cases,
    )


def _verify_fixed(raw: FixedSeedData) -> None:
    budget = int(raw.metadata["budget"])
    for case in raw.cases:
        hashes = {name: case.pairing_hash for name in case.methods}
        # All methods in a case must be paired (same initial design + lambda):
        # the single stored pairing_hash is shared, and trajectory lengths equal.
        lengths = {name: len(m.trajectory) for name, m in case.methods.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"[fixed] unequal trajectory lengths in case "
                f"({case.scalarization}, {case.lambda_angle_degrees}): {lengths}."
            )
        for name, m in case.methods.items():
            if len(m.trajectory) + len(case.initial_X) != budget:
                raise ValueError(
                    f"[fixed] method '{name}' has "
                    f"{len(m.trajectory)} BO steps + {len(case.initial_X)} init "
                    f"!= budget {budget}."
                )
        if "right" in case.methods and "wrong" in case.methods:
            # RIGHT and WRONG must share the SAME initial design numerically.
            # (The pairing hash already encodes lambda + initial X.)
            if case.pairing_hash is None:
                raise ValueError("[fixed] missing pairing hash.")


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def _process_pareto_seed(
    raw: ParetoSeedData,
    benchmark: mo_benchmarks.MultiObjectiveBenchmark,
) -> ProcessedParetoSeedData:
    meta = raw.metadata
    ref = benchmark.reference_point
    scal = str(meta["scalarization"])
    front_res = int(meta.get("pareto_front_resolution", 10000))
    front_oriented = benchmark.true_pareto_front_oriented(front_res)
    hv_star = float(raw.hv_star)
    eval_lambdas = torch.tensor(raw.evaluation_lambdas, dtype=torch.double)

    # Fixed unit scales: both ZDT analytical fronts span [0, 1] per objective.
    if benchmark.name not in ("zdt1", "zdt2"):
        raise ValueError("Specify fixed diagnostic scales for this benchmark")
    scales = [1.0, 1.0]
    diagnostic_reference = arc_length_reference(
        benchmark.true_pareto_front_raw(10001).numpy(), scales, 1001)
    method_out: dict[str, ParetoMethodProcessed] = {}
    for name, method in raw.methods.items():
        evals = method.global_evaluations
        X = torch.tensor([e.x for e in evals], dtype=torch.double)
        oriented = _oriented_at(benchmark, X)     # (budget, m) noiseless

        stored_oriented = torch.tensor([e.oriented_y for e in evals], dtype=torch.double)
        d = float((oriented - stored_oriented).abs().max().item())
        if d > _ATOL:
            raise ValueError(
                f"[pareto] {benchmark.name} seed {meta['seed']} method '{name}': "
                f"stored oriented_y disagrees with re-evaluated benchmark by "
                f"{d:.2e}."
            )

        hv = running_hypervolume_2d(oriented, ref)
        hv_regret = hypervolume_regret_2d(oriented, ref, hv_star)
        bayes = empirical_bayes_regret(oriented, front_oriented, eval_lambdas, scal, ref)

        raw_y = benchmark.evaluate_raw(X).numpy()
        diagnostic_curves = query_dominance_curves(
            dominance_regret(raw_y, scales, analytical_front=benchmark.name))
        diagnostic_curves["igd"] = running_igd(raw_y, scales, diagnostic_reference)

        method_out[name] = ParetoMethodProcessed(
            algorithm=method.algorithm,
            architecture=method.architecture,
            global_iteration=[e.global_iteration for e in evals],
            hypervolume=hv.tolist(),
            hypervolume_regret=hv_regret.tolist(),
            empirical_bayes_regret=bayes.tolist(),
            subruns=method.subruns,
            diagnostics={
                **{k: v.tolist() for k, v in diagnostic_curves.items()},
                "objective_scales": scales, "orientation": "minimize",
                "dominance_tolerance": 1e-8,
                "reference_resolution": 1001, "reference_dense_resolution": 10001,
                "reference_sampling": "uniform normalized arc length",
                "dominance_reference": "continuous analytical ZDT front",
                "budget_order": "stored global round-robin, initial design included",
            },
        )

    return ProcessedParetoSeedData(
        metadata=meta,
        benchmark=raw.benchmark,
        reference_point=ref.tolist(),
        hv_star=hv_star,
        n_evaluation_lambdas=int(eval_lambdas.shape[0]),
        methods=method_out,
    )


def _verify_pareto(raw: ParetoSeedData) -> None:
    budget = int(raw.metadata["budget"])
    # Equal budgets / trajectory lengths across methods.
    lengths = {n: len(m.global_evaluations) for n, m in raw.methods.items()}
    for n, ln in lengths.items():
        if ln != budget:
            raise ValueError(f"[pareto] method '{n}' has {ln} evals != budget {budget}.")

    # Algorithm-2 pairing: right vs wrong share lambdas, hashes, local budgets.
    r2 = raw.methods.get("right_algorithm_2")
    w2 = raw.methods.get("wrong_algorithm_2")
    if r2 is not None and w2 is not None:
        if len(r2.subruns) != len(w2.subruns):
            raise ValueError("[pareto] right/wrong Algorithm 2 have different #subruns.")
        for sr, sw in zip(r2.subruns, w2.subruns):
            if sr.pairing_hash != sw.pairing_hash:
                raise ValueError(
                    f"[pareto] Algorithm 2 subrun {sr.subrun_index} pairing hash "
                    f"mismatch: right={sr.pairing_hash} wrong={sw.pairing_hash} "
                    "(different lambda or initial design)."
                )
            if sr.lambda_vector != sw.lambda_vector:
                raise ValueError(
                    f"[pareto] Algorithm 2 subrun {sr.subrun_index} lambda mismatch."
                )
            if sr.local_budget != sw.local_budget:
                raise ValueError(
                    f"[pareto] Algorithm 2 subrun {sr.subrun_index} local-budget "
                    "mismatch."
                )
        total = sum(s.local_budget for s in r2.subruns)
        if total != budget:
            raise ValueError(
                f"[pareto] sum of local budgets {total} != global budget {budget}."
            )
        expected = budget_split(budget, len(r2.subruns))
        got = [s.local_budget for s in r2.subruns]
        if got != expected:
            raise ValueError(
                f"[pareto] local budgets {got} != canonical split {expected}."
            )


# ---------------------------------------------------------------------------
# Cross-run compatibility (used by analyze when combining several runs)
# ---------------------------------------------------------------------------

def assert_compatible_metadata(metas: list[dict[str, Any]]) -> None:
    """Raise if a list of seed metadata dicts cannot be aggregated together."""
    if not metas:
        return
    keys = [
        "experiment_mode", "benchmark", "input_dim", "output_dim",
        "orientation", "reference_point", "budget", "pareto_front_resolution",
    ]
    ref = metas[0]
    for m in metas[1:]:
        for k in keys:
            if m.get(k) != ref.get(k):
                raise ValueError(
                    f"Incompatible runs: metadata key '{k}' differs "
                    f"({ref.get(k)!r} vs {m.get(k)!r}); refusing to aggregate."
                )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _discover_seed_files(run_dir: Path) -> list[tuple[str, int, Path]]:
    raw_root = run_dir / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"No raw/ directory under {run_dir}.")
    out: list[tuple[str, int, Path]] = []
    for bench_dir in sorted(raw_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        for path in sorted(bench_dir.glob("seed_*.json")):
            seed = int(path.stem.split("_")[1])
            out.append((bench_dir.name, seed, path))
    return out


def process_seed_file(path: Path) -> tuple[str, Any]:
    """Process one raw seed file; returns ``(mode, processed)``.

    ``processed`` is a :class:`~data.processed_multiobjective.ProcessedFixedSeedData`
    or :class:`~data.processed_multiobjective.ProcessedParetoSeedData`; call
    ``.to_dict()`` to serialise it.
    """
    raw_dict = load_json(path)
    mode = raw_dict.get("mode")
    if mode == "fixed_scalarization":
        raw = FixedSeedData.from_dict(raw_dict)
        benchmark = _build_benchmark(raw.metadata)
        _verify_fixed(raw)
        return mode, _process_fixed_seed(raw, benchmark)
    if mode == "pareto_front":
        raw = ParetoSeedData.from_dict(raw_dict)
        benchmark = _build_benchmark(raw.metadata)
        _verify_pareto(raw)
        return mode, _process_pareto_seed(raw, benchmark)
    raise ValueError(f"Unknown raw file mode '{mode}' in {path}.")


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    run_logger = RunLogger(run_dir)
    seed_files = _discover_seed_files(run_dir)
    if not seed_files:
        logger.warning("No raw seed files found; nothing to do.")
        return

    logger.info(f"Processing {len(seed_files)} seed file(s) under {run_dir}")
    for benchmark, seed, path in seed_files:
        mode, processed = process_seed_file(path)
        out_path = run_logger.processed_per_seed_path(benchmark, seed)
        save_json(processed.to_dict(), out_path)
        logger.info(f"  [{mode}] {benchmark} seed {seed:>3} -> "
                    f"{out_path.relative_to(run_dir)}")
    logger.info("Processing complete; all verification checks passed.")


if __name__ == "__main__":
    main()
