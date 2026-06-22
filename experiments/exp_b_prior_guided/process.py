"""Experiment B — per-seed processing.

Loads one raw seed JSON (:mod:`data.raw_prior`), computes standard BO
metrics under the maximization convention on the raw oriented objective
(:class:`benchmarks.oriented.OrientedBenchmark`, no [0, 1] rescaling), and writes
a :class:`~data.processed_prior.ProcessedPriorSeedData` file.  No BO is
rerun; no GP is refit.  GP posteriors needed for calibration are obtained by

1. rebuilding ``train_X`` / ``train_Y`` at the chosen BO iteration from the
   initial design and the trajectory prefix (always raw ``y_obs``),
2. constructing the same model shape via the shared
   :func:`experiments.exp_b_prior_guided.run.make_model_builder`,
3. loading the snapshot's serialized ``state_dict``.

Calibration is always evaluated on the objective ``f`` (raw oriented
noiseless values), for every method.  In addition to the global held-out
grid, a **prior-region** summary is computed on
``{x : pi(x) >= prior_region_threshold}`` to study overconfidence near the
region favored by the prior.

CLI::

    python -m experiments.exp_b_prior_guided.process \\
        --run-dir results/exp_b_prior_guided/runs/<run_id>
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

from benchmarks.oriented import OrientedBenchmark, build_oriented_benchmark
from common.diagnostics import build_calibration_report_dict
from common.logging import RunLogger
from common.metrics import (
    area_under_curve,
    best_so_far,
    hitting_time,
    instantaneous_regret,
)
from common.models import load_state_dict_into
from common.priors import GaussianPrior
from common.serialization import load_json, save_json
from data.processed import CalibrationPredictionGrid
from data.processed_prior import (
    PriorCalibrationReport,
    PriorConvergenceCurves,
    PriorFinalScalars,
    PriorMethodProcessed,
    PriorTestGrid,
    ProcessedPriorSeedData,
)
from data.raw import InitialDesign, ModelSnapshot
from data.raw_prior import PriorMethodTrajectory, RawPriorSeedData
from experiments.exp_b_prior_guided.run import make_model_builder

logger = logging.getLogger("process_exp_b")


# ---------------------------------------------------------------------------
# Train-data reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_X_Y(
    initial_design: InitialDesign,
    trajectory_prefix: list,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild ``(train_X, y_obs)`` as seen at the start of one iteration.

    Every Experiment B method trains on raw ``y_obs`` — no transform.
    """
    X = torch.tensor(initial_design.X, dtype=torch.double, device=device)
    y_obs = torch.tensor(
        initial_design.y_obs, dtype=torch.double, device=device
    ).unsqueeze(-1)

    if trajectory_prefix:
        X_cand = torch.tensor(
            [step.x for step in trajectory_prefix], dtype=torch.double, device=device
        )
        y_cand = torch.tensor(
            [step.y_obs for step in trajectory_prefix],
            dtype=torch.double, device=device,
        ).unsqueeze(-1)
        X = torch.cat([X, X_cand], dim=0)
        y_obs = torch.cat([y_obs, y_cand], dim=0)
    return X, y_obs


def _rebuild_gp_at_snapshot(
    snapshot: ModelSnapshot,
    method: PriorMethodTrajectory,
    config: dict[str, Any],
    problem: OrientedBenchmark,
    prior: GaussianPrior,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
) -> Any:
    """Construct a GP of the same shape as the snapshot's, then load its
    serialized state dict.  No MLL fitting.
    """
    builder = make_model_builder(config, problem, prior, method.method_name)
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
    method: PriorMethodTrajectory,
    f_opt: float,
    epsilon: Optional[float],
) -> PriorFinalScalars:
    """Build :class:`PriorFinalScalars` from the full evaluation history.

    Maximization convention: standard minimization metrics are applied to
    the negated trajectory, as documented in :mod:`common.metrics`.
    """
    inst = instantaneous_regret(-f_history, -float(f_opt))   # f_opt - f(x_t)
    bsf = best_so_far(f_history, minimize=False)             # max_{s<=t} f(x_s)
    simple = float(f_opt) - bsf                              # running simple regret

    best_idx = int(f_history.argmax().item())
    lambda_values = [
        s.lambda_value for s in method.trajectory if s.lambda_value is not None
    ]
    prior_values = [s.prior_value_at_candidate for s in method.trajectory]

    return PriorFinalScalars(
        n_evaluations=int(f_history.numel()),
        final_best_value=float(bsf[-1].item()),
        final_simple_regret=float(simple[-1].item()),
        mean_inst_regret=float(inst.mean().item()),
        median_inst_regret=float(inst.median().item()),
        worst_inst_regret=float(inst.max().item()),
        cumulative_inst_regret=float(inst.sum().item()),
        auc_simple_regret=area_under_curve(simple.clamp_min(0.0)),
        hitting_time=(
            hitting_time(simple, float(epsilon)) if epsilon is not None else None
        ),
        best_x=X_history[best_idx].tolist(),
        lambda_final=(lambda_values[-1] if lambda_values else None),
        mean_prior_value=(
            float(torch.tensor(prior_values, dtype=torch.double).mean().item())
            if prior_values else None
        ),
    )


def _convergence_curves(
    f_history: torch.Tensor,
    method: PriorMethodTrajectory,
    f_opt: float,
) -> PriorConvergenceCurves:
    """Build :class:`PriorConvergenceCurves` from the full evaluation history."""
    n = int(f_history.numel())
    inst = instantaneous_regret(-f_history, -float(f_opt))
    bsf = best_so_far(f_history, minimize=False)
    simple = float(f_opt) - bsf
    cum_inst = inst.cumsum(dim=0)

    lambda_values = [s.lambda_value for s in method.trajectory]
    has_lambda = any(v is not None for v in lambda_values)

    return PriorConvergenceCurves(
        iteration=list(range(1, n + 1)),
        instantaneous_regret=inst.tolist(),
        running_best_value=bsf.tolist(),
        simple_regret=simple.tolist(),
        cumulative_inst_regret=cum_inst.tolist(),
        bo_iteration=[s.iteration for s in method.trajectory],
        prior_value_at_candidate=[
            s.prior_value_at_candidate for s in method.trajectory
        ],
        lambda_per_iteration=(
            [float(v) for v in lambda_values] if has_lambda else None
        ),
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _to_prior_region_dict(
    near_calib: Optional[dict[str, Any]],
    threshold: float,
    n_global: int,
) -> Optional[dict[str, Any]]:
    """Re-key the generic near-grid summary as a prior-region summary."""
    if near_calib is None:
        return None
    out = dict(near_calib)
    count = out.pop("near_target_count", out.get("n_test"))
    out.pop("epsilon", None)
    out["prior_region_count"] = int(count) if count is not None else None
    out["prior_region_fraction"] = (
        float(count) / float(n_global) if count is not None and n_global else None
    )
    out["pi_threshold"] = float(threshold)
    return out


def _calibration_reports(
    method: PriorMethodTrajectory,
    initial_design: InitialDesign,
    config: dict[str, Any],
    problem: OrientedBenchmark,
    prior: GaussianPrior,
    X_global: torch.Tensor,
    f_global: torch.Tensor,
    X_prior_region: Optional[torch.Tensor],
    f_prior_region: Optional[torch.Tensor],
    prior_region_threshold: Optional[float],
    device: torch.device,
    every: int = 5,
    compute_global: bool = True,
    compute_prior_region: bool = True,
    compute_joint: bool = False,
    store_predictions: bool = True,
    m_test_joint_distrib: Optional[int] = None,
    m_test_joint_prior_region: Optional[int] = None,
    joint_nlpd_jitter: float = 1e-6,
    beta: float = 2.0,
) -> list[PriorCalibrationReport]:
    """Held-out calibration at selected GP snapshots.

    At each captured iteration, rebuild the GP from its snapshot, then
    delegate all diagnostic computations to
    :func:`common.diagnostics.build_calibration_report_dict`.  The generic
    near-grid slot of that helper is fed with the **prior-region** subset
    ``{x : pi(x) >= threshold}`` of the global grid; ground truth is always
    the normalized noiseless objective ``f``.
    """
    if not method.model_snapshots:
        raise ValueError(f"No model snapshots for method '{method.method_name}'.")

    y_true_global = f_global.squeeze(-1)
    y_true_region = (
        f_prior_region.squeeze(-1) if f_prior_region is not None else None
    )

    reports: list[PriorCalibrationReport] = []
    for snapshot in method.model_snapshots:
        if snapshot.iteration % every != 0 and snapshot != method.model_snapshots[-1]:
            continue

        prefix = method.trajectory[: snapshot.iteration - 1]
        train_X, train_Y = _reconstruct_X_Y(initial_design, prefix, device)
        model = _rebuild_gp_at_snapshot(
            snapshot, method, config, problem, prior, train_X, train_Y
        )

        artifacts = build_calibration_report_dict(
            model=model,
            X_global=X_global,
            y_true_global=y_true_global,
            X_near=X_prior_region if compute_prior_region else None,
            y_true_near=y_true_region if compute_prior_region else None,
            near_target_epsilon=None,
            compute_global=compute_global,
            compute_near_target=compute_prior_region,
            compute_joint=compute_joint,
            m_test_joint_distrib=m_test_joint_distrib,
            m_test_joint_near_target=m_test_joint_prior_region,
            joint_nlpd_jitter=joint_nlpd_jitter,
            store_predictions=store_predictions,
            beta=beta,
        )

        predictions = (
            CalibrationPredictionGrid(**artifacts["predictions"])
            if artifacts["predictions"] is not None
            else None
        )
        reports.append(
            PriorCalibrationReport(
                iteration_index=snapshot.iteration,
                global_calibration=artifacts["global_calibration"],
                prior_region=_to_prior_region_dict(
                    artifacts["near_target"],
                    threshold=float(prior_region_threshold or 0.0),
                    n_global=int(X_global.shape[0]),
                ),
                joint_posterior=artifacts["joint_posterior"],
                predictions=predictions,
            )
        )

    return reports


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def _reconstruction_config(
    config: dict[str, Any],
    raw: RawPriorSeedData,
) -> dict[str, Any]:
    """Top-level config snapshot for :func:`build_oriented_benchmark`.

    Overrides the persisted YAML config with the bounds, noise_std, and
    orientation parameters actually recorded at run time, so the
    reconstruction is independent of any later edit to defaults.
    """
    reco = dict(config)
    bench_cfg = reco.get("benchmark")
    if isinstance(bench_cfg, dict):
        bench_cfg = dict(bench_cfg)
    else:
        bench_cfg = {"name": str(bench_cfg)}
    if raw.metadata.bounds is not None:
        bench_cfg["bounds"] = raw.metadata.bounds
    reco["benchmark"] = bench_cfg
    reco["noise_std"] = raw.metadata.noise_std
    if raw.metadata.orientation is not None:
        reco["orientation"] = dict(raw.metadata.orientation)
    return reco


def _process_one_seed(
    raw: RawPriorSeedData,
    config: dict[str, Any],
    device: torch.device,
) -> ProcessedPriorSeedData:
    problem = build_oriented_benchmark(_reconstruction_config(config, raw))
    if problem.name != raw.metadata.benchmark:
        raise ValueError(
            f"Reconstructed benchmark '{problem.name}' does not match raw "
            f"metadata benchmark '{raw.metadata.benchmark}'."
        )
    if raw.metadata.prior is None:
        raise ValueError("Raw metadata lacks the prior configuration.")
    prior = GaussianPrior.from_config(raw.metadata.prior, dim=problem.dim)

    f_opt = float(raw.metadata.f_opt)
    epsilon = config.get("epsilon")
    beta = float(raw.metadata.beta if raw.metadata.beta is not None
                 else config.get("beta", 2.0))

    test_grid_cfg = config.get("test_grid", {})
    calibration_cfg = config.get("calibration", {})
    n_test = int(test_grid_cfg.get("n_test", 256))
    test_grid_seed = int(test_grid_cfg.get("seed", 0))

    m_joint_raw = test_grid_cfg.get("m_test_joint_distrib")
    m_test_joint_distrib = int(m_joint_raw) if m_joint_raw is not None else None
    m_joint_region_raw = test_grid_cfg.get("m_test_joint_prior_region")
    m_test_joint_prior_region = (
        int(m_joint_region_raw) if m_joint_region_raw is not None else None
    )
    joint_nlpd_jitter = float(test_grid_cfg.get("joint_nlpd_jitter", 1e-6))

    compute_global = bool(calibration_cfg.get("compute_global", True))
    compute_prior_region = bool(calibration_cfg.get("compute_prior_region", True))
    compute_joint = bool(calibration_cfg.get("compute_joint", False))
    store_predictions = bool(calibration_cfg.get("store_predictions", True))

    prior_region_threshold = calibration_cfg.get("prior_region_threshold", 0.5)
    prior_region_threshold = (
        float(prior_region_threshold) if prior_region_threshold is not None else None
    )
    do_prior_region = compute_prior_region and prior_region_threshold is not None

    # Held-out grid is deterministic from (benchmark, n_test, seed) and is
    # shared by every method and snapshot.
    X_test = problem.test_set(n=n_test, seed=test_grid_seed).to(device)
    f_test = problem.f(X_test)

    # Prior-region subset: mask on the global grid by pi(x) >= threshold.
    X_region: Optional[torch.Tensor] = None
    f_region: Optional[torch.Tensor] = None
    n_region = 0
    if do_prior_region:
        mask = prior.evaluate(X_test) >= prior_region_threshold
        n_region = int(mask.sum().item())
        if n_region > 0:
            X_region = X_test[mask]
            f_region = f_test[mask]
        else:
            logger.warning(
                "Prior region {pi >= %.3f} contains no point of the %d-point "
                "global test grid; prior-region calibration will be null.",
                prior_region_threshold, n_test,
            )

    test_grid = PriorTestGrid(
        n_test=n_test,
        seed=test_grid_seed,
        prior_region_threshold=(
            prior_region_threshold if do_prior_region else None
        ),
        n_prior_region=(n_region if do_prior_region else None),
        m_test_joint_distrib=m_test_joint_distrib,
    )

    methods: dict[str, PriorMethodProcessed] = {}
    for name, method in raw.methods.items():
        f_history = _full_f_history(raw.initial_design, method.trajectory)
        X_history = _full_x_history(raw.initial_design, method.trajectory)
        final = _final_scalars(
            f_history, X_history, method, f_opt,
            epsilon=float(epsilon) if epsilon is not None else None,
        )
        curves = _convergence_curves(f_history, method, f_opt)
        calibrations = _calibration_reports(
            method=method,
            initial_design=raw.initial_design,
            config=config,
            problem=problem,
            prior=prior,
            X_global=X_test,
            f_global=f_test,
            X_prior_region=X_region,
            f_prior_region=f_region,
            prior_region_threshold=prior_region_threshold,
            device=device,
            every=int(test_grid_cfg.get("calibration_every", 5)),
            compute_global=compute_global,
            compute_prior_region=do_prior_region and X_region is not None,
            compute_joint=compute_joint,
            store_predictions=store_predictions,
            m_test_joint_distrib=m_test_joint_distrib,
            m_test_joint_prior_region=m_test_joint_prior_region,
            joint_nlpd_jitter=joint_nlpd_jitter,
            beta=beta,
        )
        methods[name] = PriorMethodProcessed(
            method_name=name,
            train_label=method.train_label,
            model_kind=method.model_kind,
            final=final,
            convergence=curves,
            calibrations=calibrations,
        )

    return ProcessedPriorSeedData(
        metadata=raw.metadata.to_dict(),
        f_opt=f_opt,
        epsilon=float(epsilon) if epsilon is not None else None,
        test_grid=test_grid,
        methods=methods,
    )


# ---------------------------------------------------------------------------
# Regret debug / verification
# ---------------------------------------------------------------------------

def recompute_regret_from_raw(
    raw: RawPriorSeedData,
    problem: OrientedBenchmark,
    f_opt: float,
) -> dict[str, dict[str, torch.Tensor]]:
    """Recompute the regret directly from the raw queries.

    For each method, re-evaluate the **oriented** benchmark ``f`` on the
    initial design and on every BO query, then form the maximization regret
    from scratch::

        f(x_t)               = problem.f(x_t)            # raw oriented value
        instantaneous_regret = f_opt - f(x_t)
        simple_regret        = f_opt - max_{s <= t} f(x_s)

    This is intentionally independent of the stored ``f_noiseless`` and of the
    processed JSON, so a mismatch pinpoints a raw/oriented confusion, a
    sign/min-max confusion, or an initial-design/BO concatenation offset.

    Returns a dict ``{method: {"f", "instantaneous_regret", "simple_regret"}}``
    of 1-D ``torch.double`` tensors over the full evaluation history.
    """
    init_X = torch.tensor(raw.initial_design.X, dtype=torch.double)
    init_f = problem.f(init_X).squeeze(-1)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, method in raw.methods.items():
        if method.trajectory:
            Xq = torch.tensor([s.x for s in method.trajectory], dtype=torch.double)
            f_bo = problem.f(Xq).squeeze(-1)
        else:
            f_bo = torch.empty(0, dtype=torch.double)
        f_all = torch.cat([init_f, f_bo])
        out[name] = {
            "f": f_all,
            "instantaneous_regret": float(f_opt) - f_all,
            "simple_regret": float(f_opt) - torch.cummax(f_all, dim=0).values,
        }
    return out


def verify_seed_consistency(
    raw: RawPriorSeedData,
    processed: ProcessedPriorSeedData,
    problem: OrientedBenchmark,
    f_opt: float,
    atol: float = 1e-6,
) -> list[tuple[str, float, float, float]]:
    """Cross-check stored vs recomputed regret for one seed.

    Compares, per method, the from-scratch recomputation
    (:func:`recompute_regret_from_raw`) against (a) the stored
    ``f_noiseless`` and (b) the processed convergence curves.  Returns the
    list of ``(method, max|df|, max|d_inst|, max|d_simple|)`` tuples that
    exceed ``atol`` (empty when everything is consistent).
    """
    recompute = recompute_regret_from_raw(raw, problem, f_opt)
    problems: list[tuple[str, float, float, float]] = []
    for name, rec in recompute.items():
        method = raw.methods[name]
        if method.trajectory:
            Xq = torch.tensor([s.x for s in method.trajectory], dtype=torch.double)
            recomputed_bo_f = problem.f(Xq).squeeze(-1)
            stored_f = torch.tensor(
                [s.f_noiseless for s in method.trajectory], dtype=torch.double
            )
            d_f = float((recomputed_bo_f - stored_f).abs().max().item())
        else:
            d_f = 0.0

        conv = processed.methods[name].convergence
        proc_inst = torch.tensor(conv.instantaneous_regret, dtype=torch.double)
        proc_sr = torch.tensor(conv.simple_regret, dtype=torch.double)
        d_inst = float((rec["instantaneous_regret"] - proc_inst).abs().max().item())
        d_sr = float((rec["simple_regret"] - proc_sr).abs().max().item())

        if max(d_f, d_inst, d_sr) > atol:
            problems.append((name, d_f, d_inst, d_sr))
    return problems


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
                        help="Torch device for GP reconstruction.  Falls back "
                             "to config['device'] or 'cpu'.")
    parser.add_argument("--verify", action="store_true",
                        help="After processing each seed, recompute the regret "
                             "directly from the raw queries (re-evaluating the "
                             "oriented benchmark) and assert it matches the "
                             "stored f_noiseless and the processed curves.")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    """Manage automatic device attribution."""
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
            f"Expected {config_path}; was the run produced by run.py?"
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
    n_verify_fail = 0
    for benchmark, seed, raw_path in seed_files:
        raw = RawPriorSeedData.from_dict(load_json(raw_path))
        processed = _process_one_seed(raw, config, device)
        out_path = run_logger.processed_per_seed_path(benchmark, seed)
        save_json(processed.to_dict(), out_path)
        logger.info(f"  [{benchmark}] seed {seed:>3} -> {out_path.relative_to(run_dir)}")

        if args.verify:
            problem = build_oriented_benchmark(_reconstruction_config(config, raw))
            problems = verify_seed_consistency(
                raw, processed, problem, float(raw.metadata.f_opt)
            )
            if problems:
                n_verify_fail += 1
                for name, d_f, d_inst, d_sr in problems:
                    logger.error(
                        "  [verify] %s seed %s method '%s': max|df|=%.2e "
                        "max|d_inst|=%.2e max|d_simple|=%.2e",
                        benchmark, seed, name, d_f, d_inst, d_sr,
                    )
            else:
                logger.info(f"  [verify] {benchmark} seed {seed:>3}: regret consistent")

    if args.verify:
        if n_verify_fail:
            raise SystemExit(
                f"Regret verification FAILED for {n_verify_fail} seed file(s); "
                "see the [verify] errors above."
            )
        logger.info("Regret verification passed for all processed seeds.")


if __name__ == "__main__":
    main()
