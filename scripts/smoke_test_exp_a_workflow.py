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
from common.models import fit_single_task_gp
from common.logging import RunLogger
from common.metrics import target_distance
from acquisitions.target_matching import TargetMatchingUCB, WrongTargetMatchingUCB


# ---------------------------------------------------------------------------
# Config (hardcoded for the smoke test)
# ---------------------------------------------------------------------------

EXPERIMENT   = "exp_a_target_matching"
TAG          = "smoke_test_1d"
SEED         = 0
TARGET       = 0.5
BETA         = 2.0
N_INITIAL    = 5
BUDGET       = 8
NOISE_STD    = 0.01    # overrides SineTrend1D default for a cleaner smoke run
NUM_RESTARTS = 5
RAW_SAMPLES  = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def posterior_stats(model, X_cand: torch.Tensor) -> tuple[float, float]:
    """Return (mean, std) of the posterior at a single candidate point.

    Args:
        model: Fitted SingleTaskGP.
        X_cand: shape (1, dim).

    Returns:
        (mean_val, std_val) as Python floats.
    """
    with torch.no_grad():
        post = model.posterior(X_cand.unsqueeze(0))
        mean = post.mean.squeeze().item()
        std  = post.variance.clamp_min(1e-10).sqrt().squeeze().item()
    return mean, std


def record_step(
    iteration: int,
    x_cand: torch.Tensor,
    y_obs_val: torch.Tensor,
    f_val: torch.Tensor,
    acq_value: torch.Tensor,
    train_size: int,
    posterior_mean: float,
    posterior_std: float,
    target: float,
    ) -> dict:
    """Build one trajectory entry; all tensors converted to plain Python."""
    return {
        "iteration":                   iteration,
        "x":                           x_cand.squeeze().tolist(),
        "y_obs":                       y_obs_val.squeeze().item(),
        "f_noiseless":                 f_val.squeeze().item(),
        "target_distance":             abs(f_val.squeeze().item() - target),
        "acq_value":                   acq_value.item(),
        "train_size":                  train_size,
        "posterior_mean_at_candidate": posterior_mean,
        "posterior_std_at_candidate":  posterior_std,
    }


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

    traj_right: list[dict] = []
    traj_wrong: list[dict] = []

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
        model_right = fit_single_task_gp(X_right, Y_right, noise_std=problem.default_noise_std)
        acq_right   = TargetMatchingUCB(model=model_right, target=TARGET, beta=BETA)

        cand_right, acq_val_right = optimize_acqf(
            acq_function=acq_right,
            bounds=problem.bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )   # cand_right: (1, dim)

        obs_right = problem.observe(cand_right)
        mu_r, sigma_r = posterior_stats(model_right, cand_right)

        traj_right.append(record_step(
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
        model_wrong = fit_single_task_gp(X_wrong, G_wrong, noise_std=problem.default_noise_std)
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
        mu_w, sigma_w = posterior_stats(model_wrong, cand_wrong)

        traj_wrong.append(record_step(
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
    best_x_right  = float(X_right[int(dist_right.argmin())].item())
    best_x_wrong  = float(X_wrong[int(dist_wrong.argmin())].item())

    # ── Build and save result dict ────────────────────────────────────────────
    result = {
        "metadata": {
            "experiment":          EXPERIMENT,
            "problem_type":        problem.problem_type,
            "benchmark":           problem.name,
            "seed":                SEED,
            "budget":              BUDGET,
            "initial_design_size": N_INITIAL,
            "target":              TARGET,
            "noise_std":           NOISE_STD,
        },
        "methods": {
            "right_target_matching_ucb": {
                "trajectory":    traj_right,
                "final_metrics": {
                    "best_target_distance": best_td_right,
                    "best_x":               best_x_right,
                    "n_evaluations":        N_INITIAL + BUDGET,
                },
            },
            "wrong_gp_on_abs_lcb": {
                "trajectory":    traj_wrong,
                "final_metrics": {
                    "best_target_distance": best_td_wrong,
                    "best_x":               best_x_wrong,
                    "n_evaluations":        N_INITIAL + BUDGET,
                },
            },
        },
    }

    seed_path = logger.save_seed_result(problem.name, seed=SEED, result=result)
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
        log.info(f"    iter={s['iteration']:>2}  x={s['x']:.4f}  |f-T|={s['target_distance']:.4f}")
    log.info("Last 3 WRONG iterations:")
    for s in traj_wrong[-3:]:
        log.info(f"    iter={s['iteration']:>2}  x={s['x']:.4f}  |f-T|={s['target_distance']:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
