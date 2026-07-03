"""Experiment A — RIGHT vs WRONG target matching.  Raw data generation.

This script runs the BO loops described in CLAUDE.md and writes ONLY raw
experimental data.  It does NOT compute per-seed metrics, calibration, or
across-seed statistics — those live in ``process.py`` and ``analyze.py``.

Output layout (see :mod:`common.logging`)::

    results/<experiment>/runs/<run_id>/
        config.json
        metadata.json
        raw/<benchmark>/seed_NNN.json
        logs/run.log

Usage::

    python -m experiments.exp_a_target_matching.run \\
        --config experiments/exp_a_target_matching/configs/smoke_1d.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import yaml
from botorch.optim import optimize_acqf
import time

# Make project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from acquisitions.target_matching import TargetMatchingUCB, WrongTargetMatchingUCB
from benchmarks import BENCHMARK_REGISTRY, SyntheticBenchmark, build_benchmark
from common.acquisition_utils import BetaSchedule
from common.initial_design import sobol_design
from common.logging import RunLogger
from common.models import GPModelBuilder, GPModelConfig, serialize_state_dict
from data.raw import (
    InitialDesign,
    MethodTrajectory,
    ModelSnapshot,
    RawSeedData,
    SeedMetadata,
    StepRecord,
)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
#
# Each entry encodes the experiment-A-specific behaviour of a method.  Keep
# this dict small: it is the only place where RIGHT / WRONG differ from a
# common scaffold.

_TransformFn = Callable[[torch.Tensor, float], torch.Tensor]
_AcqBuilderFn = Callable[[Any, float, float], Any]


def _identity_y(y_raw: torch.Tensor, target: float) -> torch.Tensor:
    """RIGHT: train on raw noisy observations y_obs."""
    return y_raw


def _abs_minus_target(y_raw: torch.Tensor, target: float) -> torch.Tensor:
    """WRONG: train on the SoC-violating transform |y_obs - T|."""
    return (y_raw - target).abs()


def _build_right_acq(model: Any, target: float, beta: float) -> Any:
    return TargetMatchingUCB(model=model, target=target, beta=beta)


def _build_wrong_acq(model: Any, target: float, beta: float) -> Any:
    return WrongTargetMatchingUCB(model=model, beta=beta)


_METHOD_SPECS: dict[str, dict[str, Any]] = {
    "right_target_matching_ucb": {
        "train_label": "y_obs",
        "transform":   _identity_y,
        "acq_builder": _build_right_acq,
        # Noise on y_obs is the known benchmark noise; use FixedNoiseGP.
        "use_fixed_noise": True,
    },
    "wrong_gp_on_abs_lcb": {
        "train_label": "abs(y_obs - target)",
        "transform":   _abs_minus_target,
        "acq_builder": _build_wrong_acq,
        # Noise on |y_obs - T| is folded near the target and is not simply
        # Gaussian; let the GP infer it.
        "use_fixed_noise": True,
    },
}


# ---------------------------------------------------------------------------
# BO loop per (seed, method)
# ---------------------------------------------------------------------------

def _capture_snapshot(
    builder: GPModelBuilder,
    model: Any,
    iteration: int,
) -> ModelSnapshot:
    """Build a :class:`ModelSnapshot` from a fitted GP and its builder."""
    return ModelSnapshot(
        iteration=iteration,
        model_type=type(model).__name__,
        kernel_name=builder.config.kernel,
        noise_std=builder.config.noise_std,
        state_dict=serialize_state_dict(model),
        hyperparameters=builder.extract_hyperparameters(model),
    )


def _run_one_method(
    method_name: str,
    config: dict[str, Any],
    problem: SyntheticBenchmark,
    X_init: torch.Tensor,
    Y_obs_init: torch.Tensor,
    log,
    device: torch.device,
) -> MethodTrajectory:
    """Run the BO loop for one method under the current seed."""
    if method_name not in _METHOD_SPECS:
        raise ValueError(f"Unknown method '{method_name}'.  "
                         f"Known: {sorted(_METHOD_SPECS)}")
    spec = _METHOD_SPECS[method_name]

    target = float(config["target"])
    beta_schedule = BetaSchedule.from_config(config)
    budget = int(config["budget"])
    num_restarts = int(config["num_restarts"])
    raw_samples = int(config["raw_samples"])

    # Builder: fixed noise for RIGHT, inferred for WRONG.
    builder_noise = problem.default_noise_std if spec["use_fixed_noise"] else None
    builder = GPModelBuilder(GPModelConfig(noise_std=builder_noise))

    # Running state.  We always keep the raw y_obs and apply the method's
    # transform inside the loop so the recipe is fully reproducible by
    # process.py.
    X = X_init.clone()
    Y_raw = Y_obs_init.clone()

    trajectory:  list[StepRecord]     = []
    snapshots:   list[ModelSnapshot]  = []

    for iteration in range(1, budget + 1):
        Y_train = spec["transform"](Y_raw, target)

        model = builder.fit(X, Y_train)
        snapshots.append(_capture_snapshot(builder, model, iteration))

        beta_t = beta_schedule.beta_at(iteration, problem.dim)
        acq = spec["acq_builder"](model, target, beta_t)
        cand, acq_val = optimize_acqf(
            acq_function=acq,
            bounds=problem.bounds.to(device),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        # Posterior at the candidate BEFORE adding it to the training set.
        mean, var = builder.predict(model, cand)
        post_mean = float(mean.squeeze().item())
        post_std  = float(var.clamp_min(1e-12).sqrt().squeeze().item())

        # Observe the raw problem — NEVER the transformed quantity.
        obs = problem.observe(cand)
        trajectory.append(StepRecord.from_tensors(
            iteration=iteration,
            x_cand=cand,
            y_obs_val=obs.y_obs,
            f_val=obs.f_noiseless,
            acq_value=acq_val,
            train_size=X.shape[0],
            posterior_mean=post_mean,
            posterior_std=post_std,
            beta_value=beta_t,
        ))

        X = torch.cat([X, cand],            dim=0)
        Y_raw = torch.cat([Y_raw, obs.y_obs], dim=0)

        log.info(
            f"    [{method_name:<28}] iter={iteration:>3}  "
            f"acq={acq_val.item():>9.4f}  "
            f"f={obs.f_noiseless.squeeze().item():>8.4f}  "
            f"|f-T|={abs(obs.f_noiseless.squeeze().item() - target):.4f}  "
            f"beta={beta_t:.3f}"
        )

    return MethodTrajectory(
        method_name=method_name,
        train_label=spec["train_label"],
        trajectory=trajectory,
        model_snapshots=snapshots,
    )


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def _run_one_seed(
    config: dict[str, Any],
    problem: SyntheticBenchmark,
    seed: int,
    log,
    device: torch.device,
) -> RawSeedData:
    """Run all configured methods under one shared initial design."""
    torch.manual_seed(seed)

    n_initial = int(config["initial_design_size"])
    target    = float(config["target"])
    # noise_std lives on the benchmark — single source of truth.
    noise_std = float(problem.default_noise_std or 0.0)

    X_init = sobol_design(problem.bounds, n=n_initial, seed=seed).to(device)
    obs_init = problem.observe(X_init)

    initial_design = InitialDesign.from_tensors(
        X=X_init, y_obs=obs_init.y_obs, f_noiseless=obs_init.f_noiseless,
    )

    log.info(
        f"  seed={seed}: initial design X{tuple(X_init.shape)}  "
        f"y_obs in [{obs_init.y_obs.min():.3f}, {obs_init.y_obs.max():.3f}]"
    )

    methods: dict[str, MethodTrajectory] = {}
    for method_name in config["methods"]:
        log.info(f"  method: {method_name}")
        methods[method_name] = _run_one_method(
            method_name=method_name,
            config=config,
            problem=problem,
            X_init=X_init,
            Y_obs_init=obs_init.y_obs,
            log=log,
            device=device,
        )

    beta_schedule = BetaSchedule.from_config(config)
    metadata = SeedMetadata(
        experiment=config["experiment"],
        problem_type=problem.problem_type,
        benchmark=problem.name,
        seed=seed,
        budget=int(config["budget"]),
        initial_design_size=n_initial,
        target=target,
        noise_std=noise_std,
        dim=problem.dim,
        bounds=problem.bounds.tolist(),
        beta=float(beta_schedule.beta_at(1, problem.dim)),
        beta_schedule=beta_schedule.to_dict(),
    )
    return RawSeedData(
        metadata=metadata,
        initial_design=initial_design,
        methods=methods,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Override the run tag.  Defaults to config['tag'] or the "
             "benchmark name.",
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=_PROJECT_ROOT / "results",
        help="Root directory under which run dirs are created.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device for tensor computation, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    return parser.parse_args(argv)

def resolve_device(name: str) -> torch.device:
    """manages automatically device attribution"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: Optional[list[str]] = None) -> Path:
    start_time = time.perf_counter()
    args = _parse_args(argv)
    config = _load_config(args.config)
    device_name = args.device if args.device is not None else config.get("device", "cpu")
    device = resolve_device(device_name)
    problem = build_benchmark(config)

    tag = args.tag or config.get("tag") 
    logger = RunLogger.create(
        experiment=config["experiment"],
        tag=tag,
        results_root=args.results_root,
    )
    log = logger.setup_file_logger("run")

    beta_schedule = BetaSchedule.from_config(config)
    logger.save_config(config)
    logger.save_metadata(extra={
        "experiment": config["experiment"],
        "benchmark":  problem.name,
        "tag":        tag,
        "seeds":      list(config["seeds"]),
        "target":     float(config["target"]),
        "budget":     int(config["budget"]),
        "noise_std":  float(problem.default_noise_std or 0.0),
        "methods":    list(config["methods"]),
        "bounds":     problem.bounds.detach().cpu().tolist(),
        "beta":       float(beta_schedule.beta_at(1, problem.dim)),
        "beta_schedule": beta_schedule.to_dict(),
    })

    log.info("=" * 60)
    log.info(f"Run: {logger.run_id()}")
    log.info(f"  device    = {device}")
    log.info(f"  benchmark = {problem.name}  dim={problem.dim}")
    log.info(f"  bounds    = {problem.bounds.detach().cpu().tolist()}")
    log.info(f"  target    = {config['target']}  noise_std = {problem.default_noise_std}")
    log.info(f"  n_initial = {config['initial_design_size']}  budget = {config['budget']}")
    log.info(f"  methods   = {list(config['methods'])}")
    log.info(f"  seeds     = {list(config['seeds'])}")
    log.info("=" * 60)

    for seed in config["seeds"]:
        raw_seed = _run_one_seed(config, problem, int(seed), log, device)
        seed_path = logger.save_seed_result(
            benchmark=problem.name,
            seed=int(seed),
            result=raw_seed.to_dict(),
        )
        log.info(f"  -> saved {seed_path.relative_to(args.results_root)}")

    log.info("=" * 60)
    log.info("Done.")
    elapsed = time.perf_counter() - start_time
    log.info(f"Run completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    return logger.run_dir


if __name__ == "__main__":
    main()
