"""Smoke test: end-to-end Experiment A workflow on a 1D toy benchmark.

Exercises the full infrastructure on a minimal example via the proper
problem abstraction:

    problem = SineTrend1D(noise_std=NOISE_STD)
    obs = problem.observe(X)   # returns Observation(X, y_obs, f_noiseless)

Two methods are compared from the same initial design:

    RIGHT  — GP trained on obs.y_obs;       TargetMatchingUCB acquisition.
    WRONG  — GP trained on |obs.y_obs - T|; WrongTargetMatchingUCB acquisition.

Purpose:
    - Verify that benchmarks/, common/, and acquisitions/ integrate correctly.
    - Exercise the ProblemSpec / Observation abstraction.
    - Check tensor shapes at each step.
    - Exercise RunLogger (config, metadata, raw seed file, processed summary).

Usage:
    python scripts/smoke_test_exp_a_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from botorch.optim import optimize_acqf

# Make project root importable regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.toy import SineTrend1D
from common.initial_design import sobol_design
from common.models import GPModelBuilder, GPModelConfig, ModelState
from common.logging import RunLogger
from common.metrics import target_distance
from acquisitions.target_matching import TargetMatchingUCB, WrongTargetMatchingUCB
from data.results import (
    ExperimentSeedResult,
    MethodProcessedResult,
    MethodRawResult,
    SeedMetadata,
    StepResult,
)


# ---------------------------------------------------------------------------
# Config (hardcoded for the smoke test)
# ---------------------------------------------------------------------------

EXPERIMENT   = "exp_a_target_matching"
TAG          = "smoke_test_1d"
SEED         = 20
TARGET       = 0.5
BETA         = 2.0
N_INITIAL    = 20
BUDGET       = 20
NOISE_STD    = 0.00001   # overrides SineTrend1D default for a cleaner smoke run
NUM_RESTARTS = 10
RAW_SAMPLES  = 256


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)

    # ── Problem (exercises the SyntheticBenchmark / ProblemSpec abstraction) ──
    problem = SineTrend1D(noise_std=NOISE_STD)

    # ── RunLogger setup ──────────────────────────────────────────────────────
    logger = RunLogger.create(
        experiment=EXPERIMENT,
        tag=TAG,
        results_root=_PROJECT_ROOT / "results",
    )
    log = logger.setup_file_logger("smoke_test")

    config = {
        "experiment":   EXPERIMENT,
        "tag":          TAG,
        "benchmark":    problem.name,
        "seed":         SEED,
        "dim":          problem.dim,
        "bounds":       problem.bounds.tolist(),
        "target":       TARGET,
        "beta":         BETA,
        "n_initial":    N_INITIAL,
        "budget":       BUDGET,
        "noise_std":    NOISE_STD,
        "methods":      ["right_target_matching_ucb", "wrong_gp_on_abs_lcb"],
    }
    logger.save_config(config)
    logger.save_metadata(extra={
        "experiment": EXPERIMENT,
        "benchmark":  problem.name,
        "seed":       SEED,
        "budget":     BUDGET,
        "target":     TARGET,
        "noise_std":  NOISE_STD,
    })

    log.info("=" * 60)
    log.info(f"Smoke test  —  Experiment A  —  {problem.name}")
    log.info(f"  run_dir  : {logger.run_dir}")
    log.info(f"  f(x)     : sin(6x) + 0.3x,  x in [0, 1]")
    log.info(f"  target T : {TARGET}")
    log.info(f"  beta     : {BETA}")
    log.info(f"  n_init   : {N_INITIAL}    budget : {BUDGET}")
    log.info(f"  noise    : {NOISE_STD}")
    log.info("=" * 60)

    # ── Shared initial design ─────────────────────────────────────────────────
    # Both RIGHT and WRONG start from identical observations.
    X_init  = sobol_design(problem.bounds, n=N_INITIAL, seed=SEED)
    obs_init = problem.observe(X_init)
    # obs_init.y_obs      : (N_INITIAL, 1) — given to the BO algorithm
    # obs_init.f_noiseless: (N_INITIAL, 1) — true values, for metrics only

    log.info(
        f"Initial design: X {tuple(X_init.shape)}  "
        f"y_obs {tuple(obs_init.y_obs.shape)}  "
        f"f_noiseless {tuple(obs_init.f_noiseless.shape)}"
    )
    log.info(
        f"  y_obs range      : [{obs_init.y_obs.min():.3f}, {obs_init.y_obs.max():.3f}]"
    )
    log.info(
        f"  f_noiseless range: [{obs_init.f_noiseless.min():.3f}, {obs_init.f_noiseless.max():.3f}]"
    )

    # ── RIGHT running state ───────────────────────────────────────────────────
    # The RIGHT GP sees obs.y_obs (raw).
    X_right = X_init.clone()
    Y_right = obs_init.y_obs.clone()           # raw observations
    F_right = obs_init.f_noiseless.clone()     # noiseless — metrics only

    # ── WRONG running state ───────────────────────────────────────────────────
    # We keep the raw y_obs; the transformation to |y_obs - T| happens
    # just before each GP fit inside the loop.
    X_wrong = X_init.clone()
    Y_wrong = obs_init.y_obs.clone()           # raw observations (untransformed)
    F_wrong = obs_init.f_noiseless.clone()

    traj_right: list[StepResult] = []
    traj_wrong: list[StepResult] = []
    model_states_right: list[ModelState] = []
    model_states_wrong: list[ModelState] = []

    # Both methods use a fixed noise_std; create builders once before the loop.
    builder_right = GPModelBuilder(GPModelConfig(noise_std=problem.default_noise_std))
    builder_wrong = GPModelBuilder(GPModelConfig(noise_std=problem.default_noise_std))

    # ── BO loop ───────────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("BO loop")
    log.info(
        f"  {'iter':>4}  {'method':<8}  {'x':>8}  {'f(x)':>8}  "
        f"{'|f-T|':>8}  {'acq':>9}"
    )
    log.info(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")

    for iteration in range(1, BUDGET + 1):

        # ── RIGHT: GP on raw y_obs, TargetMatchingUCB ─────────────────────────
        # Noise variance is fixed and known from the benchmark, so we pass it
        # explicitly to avoid re-estimating it via MLL.
        model_right = builder_right.fit(X_right, Y_right)
        model_states_right.append(builder_right.capture_state(model_right, iteration))
        acq_right   = TargetMatchingUCB(model=model_right, target=TARGET, beta=BETA)

        cand_right, acq_val_right = optimize_acqf(
            acq_function=acq_right,
            bounds=problem.bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )   # cand_right: (1, dim)

        obs_right = problem.observe(cand_right)
        mean_r, var_r = builder_right.predict(model_right, cand_right)
        mu_r    = mean_r.squeeze().item()
        sigma_r = var_r.clamp_min(1e-10).sqrt().squeeze().item()

        traj_right.append(StepResult.from_tensors(
            iteration=iteration,
            x_cand=cand_right,
            y_obs_val=obs_right.y_obs,
            f_val=obs_right.f_noiseless,
            acq_value=acq_val_right,
            train_size=X_right.shape[0],
            posterior_mean=mu_r,
            posterior_std=sigma_r,
            target=TARGET,
        ))

        X_right = torch.cat([X_right, cand_right],        dim=0)
        Y_right = torch.cat([Y_right, obs_right.y_obs],   dim=0)
        F_right = torch.cat([F_right, obs_right.f_noiseless], dim=0)

        log.info(
            f"  {iteration:>4}  {'RIGHT':<8}  "
            f"{cand_right.item():>8.4f}  "
            f"{obs_right.f_noiseless.item():>8.4f}  "
            f"{abs(obs_right.f_noiseless.item() - TARGET):>8.4f}  "
            f"{acq_val_right.item():>9.4f}"
        )

        # ── WRONG: GP on |y_obs - T|, WrongTargetMatchingUCB ─────────────────
        # SoC violation: the target is baked into the training labels.
        # The noise on g = |y_obs - T| is not simply Gaussian (it is folded
        # near the target), so we let the model infer the noise level here.
        G_wrong     = (Y_wrong - TARGET).abs()
        model_wrong = builder_wrong.fit(X_wrong, G_wrong)
        model_states_wrong.append(builder_wrong.capture_state(model_wrong, iteration))
        acq_wrong   = WrongTargetMatchingUCB(model=model_wrong, beta=BETA)

        cand_wrong, acq_val_wrong = optimize_acqf(
            acq_function=acq_wrong,
            bounds=problem.bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        # Observe the raw function — NOT the transformed g.
        obs_wrong = problem.observe(cand_wrong)
        mean_w, var_w = builder_wrong.predict(model_wrong, cand_wrong)
        mu_w    = mean_w.squeeze().item()
        sigma_w = var_w.clamp_min(1e-10).sqrt().squeeze().item()

        traj_wrong.append(StepResult.from_tensors(
            iteration=iteration,
            x_cand=cand_wrong,
            y_obs_val=obs_wrong.y_obs,
            f_val=obs_wrong.f_noiseless,
            acq_value=acq_val_wrong,
            train_size=X_wrong.shape[0],
            posterior_mean=mu_w,
            posterior_std=sigma_w,
            target=TARGET,
        ))

        X_wrong = torch.cat([X_wrong, cand_wrong],         dim=0)
        Y_wrong = torch.cat([Y_wrong, obs_wrong.y_obs],    dim=0)
        F_wrong = torch.cat([F_wrong, obs_wrong.f_noiseless], dim=0)

        log.info(
            f"  {iteration:>4}  {'WRONG':<8}  "
            f"{cand_wrong.item():>8.4f}  "
            f"{obs_wrong.f_noiseless.item():>8.4f}  "
            f"{abs(obs_wrong.f_noiseless.item() - TARGET):>8.4f}  "
            f"{acq_val_wrong.item():>9.4f}"
        )

    # ── Final metrics (use f_noiseless — never y_obs) ─────────────────────────
    f_right_all = F_right.squeeze()
    f_wrong_all = F_wrong.squeeze()

    dist_right = target_distance(f_right_all, TARGET)
    dist_wrong = target_distance(f_wrong_all, TARGET)

    best_td_right = float(dist_right.min())
    best_td_wrong = float(dist_wrong.min())
    best_x_right  = X_right[int(dist_right.argmin())].squeeze(0).tolist()
    best_x_wrong  = X_wrong[int(dist_wrong.argmin())].squeeze(0).tolist()

    # ── Build and save result dataclass ──────────────────────────────────────
    metadata = SeedMetadata(
        experiment=EXPERIMENT,
        problem_type=problem.problem_type,
        benchmark=problem.name,
        seed=SEED,
        budget=BUDGET,
        initial_design_size=N_INITIAL,
        target=TARGET,
        noise_std=NOISE_STD,
    )
    result = ExperimentSeedResult(
        metadata=metadata,
        raw_results={
            "right_target_matching_ucb": MethodRawResult("right_target_matching_ucb", traj_right, model_states_right),
            "wrong_gp_on_abs_lcb":       MethodRawResult("wrong_gp_on_abs_lcb", traj_wrong, model_states_wrong),
        },
        processed_results={
            "right_target_matching_ucb": MethodProcessedResult(
                method_name="right_target_matching_ucb",
                best_target_distance=best_td_right,
                best_x=best_x_right,
                n_evaluations=N_INITIAL + BUDGET,
            ),
            "wrong_gp_on_abs_lcb": MethodProcessedResult(
                method_name="wrong_gp_on_abs_lcb",
                best_target_distance=best_td_wrong,
                best_x=best_x_wrong,
                n_evaluations=N_INITIAL + BUDGET,
            ),
        },
    )

    seed_path = logger.save_seed_result(problem.name, seed=SEED, result=result.to_dict())
    logger.save_processed("model_diagnostics.json", {
        "right_target_matching_ucb": [s.to_dict() for s in model_states_right],
        "wrong_gp_on_abs_lcb":       [s.to_dict() for s in model_states_wrong],
    })
    logger.save_processed("summary.json", {
        "benchmark": problem.name,
        "seed":      SEED,
        "target":    TARGET,
        "right": {"best_target_distance": best_td_right, "best_x": best_x_right},
        "wrong": {"best_target_distance": best_td_wrong, "best_x": best_x_wrong},
    })

    # ── Final report ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Run complete.")
    log.info(f"  run_dir   : {logger.run_dir}")
    log.info(f"  seed file : {seed_path}")
    log.info(f"  RIGHT  best |f - T| = {best_td_right:.6f}  at x = {best_x_right:.4f}")
    log.info(f"  WRONG  best |f - T| = {best_td_wrong:.6f}  at x = {best_x_wrong:.4f}")
    log.info("-" * 60)
    log.info("Last 3 RIGHT iterations:")
    for s in traj_right[-3:]:
        log.info(f"    iter={s.iteration:>2}  x={s.x[0]:.4f}  |f-T|={s.target_distance:.4f}")
    log.info("Last 3 WRONG iterations:")
    for s in traj_wrong[-3:]:
        log.info(f"    iter={s.iteration:>2}  x={s.x[0]:.4f}  |f-T|={s.target_distance:.4f}")
    log.info("-" * 60)
    log.info("Final model hyperparameters:")
    for label, state in [("RIGHT", model_states_right[-1]), ("WRONG", model_states_wrong[-1])]:
        hp = state.hyperparameters
        log.info(
            f"  {label}  type={state.model_type}  "
            f"ls={hp.get('lengthscale', 'n/a')}  "
            f"os={hp.get('outputscale', 'n/a'):.4f}  "
            f"noise={hp.get('noise_variance_standardized', 'n/a'):.6f}  "
            f"std={hp.get('outcome_std', 'n/a'):.6f}  "
            f"noise_raw={hp.get('noise_variance_raw', 'n/a'):.6f}  "
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
