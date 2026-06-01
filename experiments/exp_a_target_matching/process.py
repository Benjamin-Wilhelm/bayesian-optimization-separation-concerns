"""Experiment A — per-seed processing.

Loads one raw seed JSON, reconstructs derived quantities, and writes a
``ProcessedSeedData`` file.  No BO is rerun; no GP is refit.  GP posteriors
needed for calibration are obtained by:

1. rebuilding ``train_X`` / ``train_Y`` at the chosen BO iteration from the
   initial design and the trajectory prefix,
2. calling :meth:`common.models.GPModelBuilder.construct` (no MLL fit),
3. calling :func:`common.models.load_state_dict_into` with the snapshot's
   serialised ``state_dict``.

Across-seed statistics live in ``analyze.py``; this script touches one seed
at a time.

CLI::

    python -m experiments.exp_a_target_matching.process \\
        --run-dir results/exp_a_target_matching/runs/<run_id>

Filters ``--benchmark`` / ``--seeds`` are optional; default is "everything in
the run dir".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

# Make project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks import (
    BENCHMARK_REGISTRY,
    SyntheticBenchmark,
    build_benchmark,
    build_benchmark_from_metadata,
)
from common.diagnostics import (
    calibration_summary,
    joint_gaussian_nlpd,
    near_target_calibration_summary,
)
from common.logging import RunLogger
from common.metrics import (
    area_under_curve,
    final_run_summary,
    target_best_so_far,
    target_cumulative_regret,
    target_distance,
)
from common.models import (
    GPModelBuilder,
    GPModelConfig,
    load_state_dict_into,
)
from common.serialization import load_json, save_json
from data.processed import (
    CalibrationReport,
    ConvergenceCurves,
    FinalScalars,
    CalibrationPredictionGrid,
    MethodProcessed,
    ProcessedSeedData,
    TestGrid,
)
from data.raw import (
    InitialDesign,
    MethodTrajectory,
    ModelSnapshot,
    RawSeedData,
)

logger = logging.getLogger("process")


# ---------------------------------------------------------------------------
# Train-data reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_X_Y(
    initial_design: InitialDesign,
    trajectory_prefix: list,
    train_label: str,
    target: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild ``(train_X, train_Y)`` as seen at the start of one iteration.

    Args:
        initial_design: Shared Sobol initial design.
        trajectory_prefix: Trajectory entries BEFORE the iteration of
            interest (so iteration ``t`` uses ``trajectory[:t-1]``).
        train_label: ``"y_obs"`` (RIGHT) or ``"abs(y_obs - target)"`` (WRONG).
        target: Target value ``T`` — used by the WRONG transform.
        device: Torch device for the returned tensors.

    Returns:
        ``(train_X, train_Y)`` as ``torch.double`` tensors with shapes
        ``(n, dim)`` and ``(n, 1)``.
    """
    X = torch.tensor(initial_design.X, dtype=torch.double, device=device)
    y_obs = torch.tensor(initial_design.y_obs, dtype=torch.double, device=device).unsqueeze(-1)

    if trajectory_prefix:
        X_cand = torch.tensor(
            [step.x for step in trajectory_prefix], dtype=torch.double, device=device
        )
        y_cand = torch.tensor(
            [step.y_obs for step in trajectory_prefix], dtype=torch.double, device=device
        ).unsqueeze(-1)
        X = torch.cat([X, X_cand], dim=0)
        y_obs = torch.cat([y_obs, y_cand], dim=0)

    if train_label == "y_obs":
        train_Y = y_obs
    elif train_label == "abs(y_obs - target)":
        train_Y = (y_obs - target).abs()
    else:
        raise ValueError(f"Unknown train_label '{train_label}'.")
    return X, train_Y


def _rebuild_gp_at_snapshot(
    snapshot: ModelSnapshot,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
) -> Any:
    """Construct a GP of the same shape as the snapshot's, then load its
    serialised state dict.  No MLL fitting.
    """
    config = GPModelConfig(noise_std=snapshot.noise_std)
    builder = GPModelBuilder(config)
    model = builder.construct(train_X, train_Y)
    load_state_dict_into(model, snapshot.state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-method derived quantities
# ---------------------------------------------------------------------------

def _full_f_history(
    initial_design: InitialDesign,
    trajectory: list,
) -> torch.Tensor:
    """Concatenate ``f_noiseless`` over the initial design and BO trajectory."""
    init_f = initial_design.f_noiseless
    if init_f is None:
        raise ValueError(
            "process.py currently requires synthetic benchmarks with "
            "f_noiseless available."
        )
    bo_f = [step.f_noiseless for step in trajectory]
    return torch.tensor(list(init_f) + bo_f, dtype=torch.double)


def _full_x_history(
    initial_design: InitialDesign,
    trajectory: list,
) -> torch.Tensor:
    init_X = torch.tensor(initial_design.X, dtype=torch.double)
    if not trajectory:
        return init_X
    bo_X = torch.tensor([step.x for step in trajectory], dtype=torch.double)
    return torch.cat([init_X, bo_X], dim=0)


def _final_scalars(
    f_history: torch.Tensor,
    X_history: torch.Tensor,
    target: float,
    oracle_best: float,
    epsilon: Optional[float],
    failure_threshold: Optional[float],
) -> FinalScalars:
    """Build :class:`FinalScalars` from the full evaluation history."""
    summary = final_run_summary(
        f_values=f_history,
        target=target,
        oracle_best=oracle_best,
        epsilon=epsilon,
        failure_threshold=failure_threshold,
    )
    distances = target_distance(f_history, target)
    bsf = target_best_so_far(f_history, target)
    auc = area_under_curve((bsf - oracle_best).clamp_min(0.0))
    best_idx = int(distances.argmin().item())
    return FinalScalars(
        n_evaluations=int(f_history.numel()),
        final_best_target_distance=summary["final_best_target_distance"],
        final_target_simple_regret=summary["final_target_simple_regret"],
        mean_inst_target_distance=summary["mean_instantaneous_target_distance"],
        median_inst_target_distance=summary["median_instantaneous_target_distance"],
        worst_inst_target_distance=summary["worst_instantaneous_target_distance"],
        cumulative_inst_target_regret=summary["cumulative_instantaneous_target_regret"],
        cumulative_bsf_target_regret=summary["cumulative_best_so_far_target_regret"],
        auc_running_target_regret=auc,
        hitting_time=summary.get("hitting_time"),
        failure_indicator=summary.get("failure_indicator"),
        best_x=X_history[best_idx].tolist(),
    )


def _convergence_curves(
    f_history: torch.Tensor,
    target: float,
    oracle_best: float,
) -> ConvergenceCurves:
    """Build :class:`ConvergenceCurves` from the full evaluation history."""
    n = int(f_history.numel())
    inst = target_distance(f_history, target)
    bsf = target_best_so_far(f_history, target)
    cum_inst = target_cumulative_regret(
        f_history, target, best_achievable=oracle_best, cumulative_best=False
    )
    cum_bsf = target_cumulative_regret(
        f_history, target, best_achievable=oracle_best, cumulative_best=True
    )
    return ConvergenceCurves(
        iteration=list(range(1, n + 1)),
        instantaneous_target_distance=inst.tolist(),
        running_best_target_distance=bsf.tolist(),
        cumulative_inst_target_regret=cum_inst.tolist(),
        cumulative_bsf_target_regret=cum_bsf.tolist(),
    )


def _y_true_for_method(
    train_label: str,
    f_test: torch.Tensor,
    target: float,
) -> torch.Tensor:
    """Pick the appropriate ground-truth quantity for held-out calibration.

    - ``"y_obs"`` → the GP models ``f``; compare against ``f_test``.
    - ``"abs(y_obs - target)"`` → the GP models ``|f - T|``;
      compare against ``|f_test - T|``.
    """
    if train_label == "y_obs":
        return f_test
    if train_label == "abs(y_obs - target)":
        return (f_test - target).abs()
    raise ValueError(f"Unknown train_label '{train_label}'.")


def _compute_near_target_joint_nlpd(
    *,
    model: Any,
    X_near: torch.Tensor,
    f_near: torch.Tensor,
    target: float,
    train_label: str,
    jitter: float,
    near_total: int,
    snapshot_iter: int,
    method_name: str,
) -> dict[str, Any]:
    """Joint Gaussian NLPD on a pre-selected near-target subset.

    Mirrors the global joint-NLPD calculation in :func:`_calibration_reports`
    but on the near-target slice.  Returns a dict whose keys are merged into
    the ``near_target`` calibration block via ``dict.update``::

        joint_n_test, joint_total_count, joint_nlpd,
        joint_mahalanobis_sq, joint_logdet, joint_jitter

    Numerical failures are absorbed: ``joint_nlpd`` is set to ``nan`` and an
    ``joint_error`` key carries the exception message, matching the convention
    used by the global joint path.
    """
    out: dict[str, Any] = {
        "joint_n_test": int(X_near.shape[0]),
        "joint_total_count": int(near_total),
        "joint_jitter": float(jitter),
    }
    try:
        with torch.no_grad():
            posterior = model.posterior(X_near)
            joint_mean = posterior.mean.squeeze(-1)
            joint_cov = posterior.mvn.covariance_matrix
        y_true = _y_true_for_method(train_label, f_near, target)
        joint = joint_gaussian_nlpd(
            y_true=y_true.cpu().numpy(),
            mean=joint_mean.cpu().numpy(),
            covariance=joint_cov.cpu().numpy(),
            jitter=float(jitter),
        )
        out["joint_nlpd"] = float(joint.get("nlpd_joint", float("nan")))
        out["joint_mahalanobis_sq"] = float(joint.get("mahalanobis_sq", float("nan")))
        out["joint_logdet"] = float(joint.get("logdet", float("nan")))
        if "error" in joint:
            out["joint_error"] = str(joint["error"])
    except Exception as exc:  # pragma: no cover — diagnostic resilience
        logger.warning(
            "Near-target joint NLPD failed at iter %d for method '%s': %s",
            snapshot_iter, method_name, exc,
        )
        out["joint_nlpd"] = float("nan")
        out["joint_mahalanobis_sq"] = float("nan")
        out["joint_logdet"] = float("nan")
        out["joint_error"] = str(exc)
    return out


def _calibration_reports(
    method: MethodTrajectory,
    initial_design: InitialDesign,
    problem: SyntheticBenchmark,
    X_test: torch.Tensor,
    f_test: torch.Tensor,
    target: float,
    near_target_epsilon: Optional[float],
    device: torch.device,
    every: int = 5,
    compute_global: bool = True,
    compute_near_target: bool = True,
    compute_joint: bool = False,
    store_predictions: bool = True,
    m_test_joint_distrib: Optional[int] = None,
    m_test_joint_near_target: Optional[int] = None,
    joint_nlpd_jitter: float = 1e-6,
) -> list[CalibrationReport]:
    """Held-out calibration at selected GP snapshots.

    Rebuilds the trained GP via load_state_dict, evaluates the posterior on
    the test grid, and computes calibration diagnostics against the appropriate
    ground truth: f for RIGHT, |f - T| for WRONG.

    When ``compute_joint`` is True and ``m_test_joint_distrib`` is a positive
    integer, a global joint Gaussian NLPD is computed on the first
    ``m_test_joint_distrib`` points of ``X_test``.  Only scalars are stored
    (never the covariance itself); see :func:`joint_gaussian_nlpd`.

    When ``m_test_joint_near_target`` is also a positive integer, an
    additional joint Gaussian NLPD restricted to the near-target slice
    (``|f_test - target| <= near_target_epsilon``) is computed on the first
    ``m_test_joint_near_target`` such points (deterministic, shared across
    methods and iterations).  Results are stored under the ``joint_*`` keys
    of the ``near_target`` calibration dict.
    """
    reports = []
    if not method.model_snapshots:
        raise ValueError(f"No model snapshots for method '{method.method_name}'.")

    do_joint = (
        compute_joint
        and m_test_joint_distrib is not None
        and int(m_test_joint_distrib) > 0
    )
    if do_joint:
        m_joint = min(int(m_test_joint_distrib), int(X_test.shape[0]))
        X_joint = X_test[:m_joint]
        f_joint = f_test.squeeze(-1)[:m_joint]

    # Pre-compute the near-target joint subset once: same indices, same order,
    # for every method and every snapshot iteration.
    near_joint_X: Optional[torch.Tensor] = None
    near_joint_f: Optional[torch.Tensor] = None
    near_joint_total: int = 0
    if (
        compute_near_target
        and near_target_epsilon is not None
        and m_test_joint_near_target is not None
        and int(m_test_joint_near_target) > 0
    ):
        flat_f = f_test.squeeze(-1)
        mask = (flat_f - target).abs() <= float(near_target_epsilon)
        all_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        near_joint_total = int(all_idx.numel())
        k = min(int(m_test_joint_near_target), near_joint_total)
        # A joint distribution requires at least 2 points.
        if k >= 2:
            sel = all_idx[:k]
            near_joint_X = X_test[sel]
            near_joint_f = flat_f[sel]

    for snapshot in method.model_snapshots:
        if snapshot.iteration % every != 0 and snapshot != method.model_snapshots[-1]:
            continue

        prefix = method.trajectory[: snapshot.iteration - 1]
        train_X, train_Y = _reconstruct_X_Y(
            initial_design=initial_design,
            trajectory_prefix=prefix,
            train_label=method.train_label,
            target=target,
            device=device,
        )

        model = _rebuild_gp_at_snapshot(snapshot, train_X, train_Y)

        with torch.no_grad():
            posterior = model.posterior(X_test)
            mean = posterior.mean.squeeze(-1)
            var = posterior.variance.clamp_min(1e-12).squeeze(-1)
            std = var.sqrt()

        y_true_full = _y_true_for_method(method.train_label, f_test.squeeze(-1), target)

        grid_summary: dict[str, Any] = {}
        if compute_global:
            grid_summary = calibration_summary(
                y_true=y_true_full.cpu().numpy(),
                mean=mean.cpu().numpy(),
                std=std.cpu().numpy(),
            )

        near_target_summary: Optional[dict[str, Any]] = None
        if compute_near_target and near_target_epsilon is not None:
            near_target_summary = near_target_calibration_summary(
                f_true=f_test.squeeze(-1).cpu().numpy(),
                mean=mean.cpu().numpy(),
                std=std.cpu().numpy(),
                target=target,
                epsilon=near_target_epsilon,
            )
            if near_joint_X is not None:
                near_target_summary.update(
                    _compute_near_target_joint_nlpd(
                        model=model,
                        X_near=near_joint_X,
                        f_near=near_joint_f,
                        target=target,
                        train_label=method.train_label,
                        jitter=float(joint_nlpd_jitter),
                        near_total=near_joint_total,
                        snapshot_iter=snapshot.iteration,
                        method_name=method.method_name,
                    )
                )

        joint_post: Optional[dict[str, Any]] = None
        if do_joint:
            try:
                with torch.no_grad():
                    joint_posterior = model.posterior(X_joint)
                    joint_mean = joint_posterior.mean.squeeze(-1)
                    joint_cov = joint_posterior.mvn.covariance_matrix
                y_joint = _y_true_for_method(method.train_label, f_joint, target)
                joint_post = joint_gaussian_nlpd(
                    y_true=y_joint.cpu().numpy(),
                    mean=joint_mean.cpu().numpy(),
                    covariance=joint_cov.cpu().numpy(),
                    jitter=float(joint_nlpd_jitter),
                )
            except Exception as exc:  # pragma: no cover — diagnostic resilience
                logger.warning(
                    "Joint NLPD failed at iter %d for method '%s': %s",
                    snapshot.iteration, method.method_name, exc,
                )
                joint_post = {
                    "m_test": int(X_joint.shape[0]),
                    "jitter": float(joint_nlpd_jitter),
                    "nlpd_joint": float("nan"),
                    "mahalanobis_sq": float("nan"),
                    "logdet": float("nan"),
                    "error": str(exc),
                }

        predictions: Optional[CalibrationPredictionGrid] = None
        if store_predictions:
            predictions = CalibrationPredictionGrid(
                posterior_mean=mean.detach().cpu().tolist(),
                posterior_std=std.detach().cpu().tolist(),
            )

        reports.append(
            CalibrationReport(
                iteration_index=snapshot.iteration,
                global_calibration=grid_summary,
                near_target=near_target_summary,
                joint_posterior=joint_post,
                predictions=predictions,
            )
        )

    return reports


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def _reconstruction_config(
    config: dict[str, Any],
    raw: RawSeedData,
) -> dict[str, Any]:
    """Build a top-level config snapshot suitable for :func:`build_benchmark`.

    Overrides the persisted YAML config with the resolved bounds and
    noise_std actually recorded at run time, so reconstruction is independent
    of any later edit to defaults.  Supports both the legacy flat layout
    (``benchmark`` is a string) and the nested layout (``benchmark`` is a
    mapping with a ``name`` key).
    """
    reco = dict(config)
    bench_cfg = reco.get("benchmark")
    if isinstance(bench_cfg, dict):
        bench_cfg = dict(bench_cfg)
        if raw.metadata.bounds is not None:
            bench_cfg["bounds"] = raw.metadata.bounds
        bench_cfg["noise_std"] = raw.metadata.noise_std
        reco["benchmark"] = bench_cfg
    else:
        # Flat layout: noise_std and bounds live at the top level.
        if raw.metadata.bounds is not None:
            reco["bounds"] = raw.metadata.bounds
        reco["noise_std"] = raw.metadata.noise_std
    return reco


def _process_one_seed(
    raw: RawSeedData,
    config: dict[str, Any],
    device: torch.device,
) -> ProcessedSeedData:
    benchmark_name = raw.metadata.benchmark
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Benchmark '{benchmark_name}' not in BENCHMARK_REGISTRY.")
    # Use the persisted run config as the source of truth, but force the
    # bounds and noise_std actually used at run time (recorded per seed) to
    # guarantee deterministic reconstruction even if the YAML defaults later
    # drift.
    problem = build_benchmark(_reconstruction_config(config, raw))

    target = raw.metadata.target
    epsilon = config.get("epsilon")
    failure_threshold = config.get("failure_threshold")
    near_target_epsilon = config.get("near_target_epsilon")
    test_grid_cfg = config.get("test_grid", {})
    calibration_cfg = config.get("calibration", {})
    n_test = int(test_grid_cfg.get("n_test", 256))
    test_grid_seed = int(test_grid_cfg.get("seed", 0))

    m_joint_raw = test_grid_cfg.get("m_test_joint_distrib")
    m_test_joint_distrib = (
        int(m_joint_raw) if m_joint_raw is not None else None
    )
    m_joint_near_raw = test_grid_cfg.get("m_test_joint_near_target")
    m_test_joint_near_target = (
        int(m_joint_near_raw) if m_joint_near_raw is not None else None
    )
    joint_nlpd_jitter = float(test_grid_cfg.get("joint_nlpd_jitter", 1e-6))

    compute_global = bool(calibration_cfg.get("compute_global", True))
    compute_near_target = bool(calibration_cfg.get("compute_near_target", True))
    compute_joint = bool(calibration_cfg.get("compute_joint", False))
    store_predictions = bool(calibration_cfg.get("store_predictions", True))

    test_grid = TestGrid(
        n_test=n_test,
        seed=test_grid_seed,
        near_target_epsilon=(
            float(near_target_epsilon)
            if near_target_epsilon is not None
            else None
        ),
        near_target_mode=(
            "mask_on_global_grid"
            if near_target_epsilon is not None
            else None
        ),
        n_test_near_target=None,
        near_target_seed=None,
        near_target_pool_size=None,
        m_test_joint_distrib=m_test_joint_distrib,
        m_test_joint_near_target=m_test_joint_near_target,
    )
    # Held-out grid is deterministic from (benchmark, n_test, seed).
    X_test = problem.test_set(n=n_test, seed=test_grid_seed).to(device)
    f_test = problem.f(X_test)

    # Oracle.  Synthetic benchmarks can override; default returns None,
    # in which case we treat the simple regret as min_t |f - T| - 0.
    oracle = problem.oracle_target_distance(target)
    oracle_best = float(oracle) if oracle is not None else 0.0

    methods: dict[str, MethodProcessed] = {}
    for name, method in raw.methods.items():
        f_history = _full_f_history(raw.initial_design, method.trajectory)
        X_history = _full_x_history(raw.initial_design, method.trajectory)
        final = _final_scalars(
            f_history, X_history, target, oracle_best,
            epsilon=float(epsilon) if epsilon is not None else None,
            failure_threshold=(
                float(failure_threshold) if failure_threshold is not None else None
            ),
        )
        curves = _convergence_curves(f_history, target, oracle_best)
        calibration = _calibration_reports(
            method=method,
            initial_design=raw.initial_design,
            problem=problem,
            X_test=X_test,
            f_test=f_test,
            target=target,
            near_target_epsilon=(
                float(near_target_epsilon) if near_target_epsilon is not None
                else None
            ),
            device=device,
            every=int(test_grid_cfg.get("calibration_every", 5)),
            compute_global=compute_global,
            compute_near_target=compute_near_target,
            compute_joint=compute_joint,
            store_predictions=store_predictions,
            m_test_joint_distrib=m_test_joint_distrib,
            m_test_joint_near_target=m_test_joint_near_target,
            joint_nlpd_jitter=joint_nlpd_jitter,
        )
        methods[name] = MethodProcessed(
            method_name=name,
            train_label=method.train_label,
            final=final,
            convergence=curves,
            calibrations=calibration,
        )

    return ProcessedSeedData(
        metadata=raw.metadata.to_dict(),
        target=target,
        epsilon=float(epsilon) if epsilon is not None else None,
        failure_threshold=(
            float(failure_threshold) if failure_threshold is not None else None
        ),
        oracle_best=oracle_best,
        test_grid=test_grid,
        methods=methods,
    )


# ---------------------------------------------------------------------------
# Run-dir orchestration
# ---------------------------------------------------------------------------

def _discover_seed_files(
    run_dir: Path,
    benchmarks: Optional[list[str]],
    seeds: Optional[list[int]],
) -> list[tuple[str, int, Path]]:
    raw_root = run_dir / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"No raw/ directory under {run_dir}.")
    out: list[tuple[str, int, Path]] = []
    for bench_dir in sorted(raw_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        if benchmarks is not None and bench_dir.name not in benchmarks:
            continue
        for path in sorted(bench_dir.glob("seed_*.json")):
            seed = int(path.stem.split("_")[1])
            if seeds is not None and seed not in seeds:
                continue
            out.append((bench_dir.name, seed, path))
    return out


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", default=None,
                        help="Filter (repeatable).  Defaults to all benchmarks.")
    parser.add_argument("--seed", action="append", type=int, default=None,
                        help="Filter (repeatable).  Defaults to all seeds.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device for GP reconstruction, e.g. 'cpu', 'cuda'. "
                             "Falls back to config['device'] or 'cpu'.")
    return parser.parse_args(argv)

def resolve_device(name: str) -> torch.device:
    "manages automatically device attribution"
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected {config_path}; was the run produced by the new run.py?"
        )
    config = load_json(config_path)

    device_name = args.device or config.get("device", "cpu")
    device = resolve_device(device_name)

    run_logger = RunLogger(run_dir)

    seed_files = _discover_seed_files(run_dir, args.benchmark, args.seed)
    if not seed_files:
        logger.warning("No raw seed files matched the filters; nothing to do.")
        return

    logger.info(f"Processing {len(seed_files)} raw seed file(s) under {run_dir}")
    for benchmark, seed, raw_path in seed_files:
        raw = RawSeedData.from_dict(load_json(raw_path))
        processed = _process_one_seed(raw, config, device)
        out_path = run_logger.processed_per_seed_path(benchmark, seed)
        save_json(processed.to_dict(), out_path)
        logger.info(f"  [{benchmark}] seed {seed:>3} -> {out_path.relative_to(run_dir)}")


if __name__ == "__main__":
    main()
