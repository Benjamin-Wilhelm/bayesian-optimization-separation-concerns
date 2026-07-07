"""Experiment B — prior-guided BO.  Raw data generation.

Compares four ways of incorporating a Gaussian domain prior into GP-BO on
benchmarks kept on their **raw (natural) scale**, only oriented to a
maximization problem (see :class:`benchmarks.oriented.OrientedBenchmark`):

- ``vanilla_ucb``            GP-UCB without prior (baseline).
- ``alpha_pi_ucb``           SoC-compliant: generic GP; the prior ``pi`` (in
                             ``[0, 1]``) reshapes the acquisition only.
- ``lambda_pi_fixed_ucb``    Prior in the GP mean, ``f ~ GP(1 * f_prior, k)``.
- ``lambda_pi_adaptive_ucb`` Prior in the GP mean,
                             ``f ~ GP(lambda * f_prior, k)`` with ``lambda``
                             learned jointly with the kernel hyperparameters
                             by MLL maximization.

Here ``f_prior(x) = baseline + amplitude * pi(x)`` lives on the observation
(``y``) scale (see :class:`common.priors.PriorMeanModule`).  The GP always
models the original objective ``f`` and trains on raw noisy ``y_obs``; no
method runs BO on residuals.

Standardization policy (per method): the standard-GP methods
(``vanilla_ucb`` / ``alpha_pi_ucb``) use ``Standardize(Y)`` since the raw
objective can have an arbitrary scale; the prior-to-the-mean methods disable
``Standardize(Y)`` so the mean function ``lambda * f_prior`` stays on the
same scale as the targets (otherwise ``lambda`` would absorb a scale
mismatch instead of learning a genuine prior/objective alignment).

This script writes ONLY raw experimental data.  Per-seed metrics and
calibration live in ``process.py``; across-seed statistics in ``analyze.py``.

Output layout (see :mod:`common.logging`)::

    results/exp_b_prior_guided/runs/<run_id>/
        config.json
        metadata.json
        raw/<benchmark>/seed_NNN.json
        logs/run.log

Usage::

    python -m experiments.exp_b_prior_guided.run \\
        --config experiments/exp_b_prior_guided/configs/smoke_forrester.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from botorch.optim import optimize_acqf

# Make project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from acquisitions.prior_guided import AlphaPiUCB, LambdaPriorUCB, VanillaUCB
from benchmarks.oriented import OrientedBenchmark, build_oriented_benchmark
from common.acquisition_utils import BetaSchedule
from common.initial_design import sobol_design
from common.logging import RunLogger
from common.models import GPModelConfig, serialize_state_dict
from common.priors import GaussianPrior, PriorGPModelBuilder, PriorMeanModule
from data.raw import InitialDesign, ModelSnapshot
from data.raw_prior import (
    PriorMethodTrajectory,
    PriorSeedMetadata,
    PriorStepRecord,
    RawPriorSeedData,
)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
#
# Each entry encodes how a method differs from the shared scaffold: whether
# the GP carries the prior in its mean (and if ``lambda`` is trained), and
# how to build the acquisition.  Every method trains on raw ``y_obs``.

_METHOD_SPECS: dict[str, dict[str, Any]] = {
    "vanilla_ucb": {
        "model_kind":   "standard_gp",
        "learn_lambda": None,
        "acq_builder":  lambda model, prior, alpha, beta: VanillaUCB(
            model=model, beta=beta
        ),
    },
    "alpha_pi_ucb": {
        "model_kind":   "standard_gp",
        "learn_lambda": None,
        "acq_builder":  lambda model, prior, alpha, beta: AlphaPiUCB(
            model=model, prior=prior, alpha=alpha, beta=beta
        ),
    },
    "lambda_pi_fixed_ucb": {
        "model_kind":   "prior_mean_gp",
        "learn_lambda": False,
        "acq_builder":  lambda model, prior, alpha, beta: LambdaPriorUCB(
            model=model, beta=beta
        ),
    },
    "lambda_pi_adaptive_ucb": {
        "model_kind":   "prior_mean_gp",
        "learn_lambda": True,
        "acq_builder":  lambda model, prior, alpha, beta: LambdaPriorUCB(
            model=model, beta=beta
        ),
    },
    # P-BO-style standardized variants: the GP uses Standardize(Y) and the
    # prior weight pi(x) is itself z-scored (see common.priors), so lambda
    # couples two O(1) standardized quantities instead of absorbing the raw
    # objective scale.  Same acquisition (UCB on the de-standardized posterior
    # of f) as the raw lambda methods; only the model's internal scaling
    # differs.  Kept as separate methods so the raw versions are untouched and
    # the two can be compared side by side.
    "lambda_pi_fixed_standardized_ucb": {
        "model_kind":            "prior_mean_gp",
        "learn_lambda":          False,
        "standardize_prior_mean": True,
        "acq_builder":  lambda model, prior, alpha, beta: LambdaPriorUCB(
            model=model, beta=beta
        ),
    },
    "lambda_pi_adaptive_standardized_ucb": {
        "model_kind":            "prior_mean_gp",
        "learn_lambda":          True,
        "standardize_prior_mean": True,
        "acq_builder":  lambda model, prior, alpha, beta: LambdaPriorUCB(
            model=model, beta=beta
        ),
    },
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_model_builder(
    config: dict[str, Any],
    problem: OrientedBenchmark,
    prior: GaussianPrior,
    method_name: str,
) -> PriorGPModelBuilder:
    """Build the GP builder for one method from the run config.

    Shared between ``run.py`` (fitting) and ``process.py`` (snapshot
    reconstruction) so the model shape is identical in both stages.

    Model defaults for Experiment B: fixed known noise, fixed-bounds input
    normalization.  Outcome standardization is decided **per method**, not by
    a global flag:

    - standard-GP methods (``vanilla_ucb`` / ``alpha_pi_ucb``) use
      ``Standardize(Y)`` (the raw objective scale is arbitrary; this is a
      purely internal GP transform that does not touch the benchmark or the
      regret);
    - prior-to-the-mean methods (``lambda_pi_*``) disable ``Standardize(Y)``
      so the mean function ``lambda * f_prior`` stays on the target scale.

    ``model.standardize_output`` in the config (default ``True``) only affects
    the standard-GP methods; it is forced to ``False`` for prior-mean methods.
    For prior-mean methods the prior mean is built on the ``y`` scale via
    ``f_prior(x) = baseline + amplitude * pi(x)`` with ``(baseline,
    amplitude)`` from :meth:`OrientedBenchmark.prior_mean_scale`.
    """
    if method_name not in _METHOD_SPECS:
        raise ValueError(f"Unknown method '{method_name}'.  "
                         f"Known: {sorted(_METHOD_SPECS)}")
    spec = _METHOD_SPECS[method_name]
    model_cfg = config.get("model", {})

    use_fixed_noise = bool(model_cfg.get("use_fixed_noise", True))
    noise_std = float(problem.default_noise_std or 0.0)
    is_prior_mean = spec["model_kind"] == "prior_mean_gp"
    standardize_prior_mean = bool(spec.get("standardize_prior_mean", False))

    # Per-method standardization:
    #   - standardized prior-mean methods REQUIRE Standardize(Y) (the prior
    #     weight is z-scored, so lambda couples two standardized quantities);
    #   - raw prior-mean methods must NOT standardize (mean on the y scale);
    #   - standard-GP methods honor the config (default True).
    if is_prior_mean:
        standardize_output = standardize_prior_mean
    else:
        standardize_output = bool(model_cfg.get("standardize_output", True))
    gp_config = GPModelConfig(
        noise_std=noise_std if use_fixed_noise else None,
        normalize_input=bool(model_cfg.get("normalize_input", True)),
        standardize_output=standardize_output,
        kernel=str(model_cfg.get("kernel", "matern52")),
    )

    # Raw prior mean f_prior(x) = baseline + amplitude * pi(x) lives on the y
    # scale; the standardized variant z-scores pi(x) instead and ignores
    # (baseline, amplitude), so do not pull the oracle prior_mean_scale there.
    if is_prior_mean and not standardize_prior_mean:
        baseline, amplitude = problem.prior_mean_scale()
    else:
        baseline, amplitude = 0.0, 1.0
    return PriorGPModelBuilder(
        gp_config,
        bounds=problem.bounds,
        prior=prior if is_prior_mean else None,
        lambda_init=float(config.get("lambda_init", 1.0)),
        learn_lambda=bool(spec["learn_lambda"]) if is_prior_mean else True,
        prior_baseline=baseline,
        prior_amplitude=amplitude,
        standardize_prior_mean=standardize_prior_mean,
    )


# ---------------------------------------------------------------------------
# BO loop per (seed, method)
# ---------------------------------------------------------------------------

def _capture_snapshot(
    builder: PriorGPModelBuilder,
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
    problem: OrientedBenchmark,
    prior: GaussianPrior,
    X_init: torch.Tensor,
    Y_obs_init: torch.Tensor,
    log,
    device: torch.device,
) -> PriorMethodTrajectory:
    """Run the BO loop for one method under the current seed."""
    spec = _METHOD_SPECS[method_name]

    alpha = float(config.get("alpha", 0.5))
    beta_schedule = BetaSchedule.from_config(config)
    budget = int(config["budget"])
    num_restarts = int(config["num_restarts"])
    raw_samples = int(config["raw_samples"])

    builder = make_model_builder(config, problem, prior, method_name)

    # Running state.  All Experiment B methods train on raw y_obs.
    X = X_init.clone()
    Y_obs = Y_obs_init.clone()

    trajectory: list[PriorStepRecord]  = []
    snapshots:  list[ModelSnapshot]    = []

    for iteration in range(1, budget + 1):
        model = builder.fit(X, Y_obs)
        snapshots.append(_capture_snapshot(builder, model, iteration))

        lambda_value: Optional[float] = None
        if isinstance(model.mean_module, PriorMeanModule):
            lambda_value = model.mean_module.lambda_value

        beta_t = beta_schedule.beta_at(iteration, problem.dim)
        acq = spec["acq_builder"](model, prior, alpha, beta_t)
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

        prior_value = float(prior.evaluate(cand).squeeze().item())

        obs = problem.observe(cand)
        trajectory.append(PriorStepRecord.from_tensors(
            iteration=iteration,
            x_cand=cand,
            y_obs_val=obs.y_obs,
            f_val=obs.f_noiseless,
            acq_value=acq_val,
            train_size=X.shape[0],
            posterior_mean=post_mean,
            posterior_std=post_std,
            prior_value=prior_value,
            lambda_value=lambda_value,
            beta_value=beta_t,
        ))

        X = torch.cat([X, cand], dim=0)
        Y_obs = torch.cat([Y_obs, obs.y_obs], dim=0)

        lam_str = f"  lambda={lambda_value:>7.4f}" if lambda_value is not None else ""
        log.info(
            f"    [{method_name:<24}] iter={iteration:>3}  "
            f"acq={acq_val.item():>9.4f}  "
            f"f={obs.f_noiseless.squeeze().item():>7.4f}  "
            f"pi={prior_value:.4f}  beta={beta_t:.3f}{lam_str}"
        )

    return PriorMethodTrajectory(
        method_name=method_name,
        train_label="y_obs",
        model_kind=spec["model_kind"],
        learn_lambda=spec["learn_lambda"],
        trajectory=trajectory,
        model_snapshots=snapshots,
    )


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def _run_one_seed(
    config: dict[str, Any],
    problem: OrientedBenchmark,
    prior: GaussianPrior,
    seed: int,
    log,
    device: torch.device,
) -> RawPriorSeedData:
    """Run all configured methods under one shared initial design."""
    torch.manual_seed(seed)

    n_initial = int(config["initial_design_size"])
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

    methods: dict[str, PriorMethodTrajectory] = {}
    for method_name in config["methods"]:
        log.info(f"  method: {method_name}")
        methods[method_name] = _run_one_method(
            method_name=method_name,
            config=config,
            problem=problem,
            prior=prior,
            X_init=X_init,
            Y_obs_init=obs_init.y_obs,
            log=log,
            device=device,
        )

    prior_meta = dict(prior.to_dict())
    prior_cfg = config.get("prior", {})
    if isinstance(prior_cfg, dict) and prior_cfg.get("label") is not None:
        prior_meta["label"] = str(prior_cfg["label"])

    beta_schedule = BetaSchedule.from_config(config)

    metadata = PriorSeedMetadata(
        experiment=config["experiment"],
        problem_type=problem.problem_type,
        benchmark=problem.name,
        base_benchmark=problem.inner.name,
        seed=seed,
        budget=int(config["budget"]),
        initial_design_size=n_initial,
        noise_std=noise_std,
        dim=problem.dim,
        bounds=problem.bounds.tolist(),
        orientation={
            "f_min": problem.f_min,
            "f_max": problem.f_max,
            "direction": problem.direction,
            "f_lower": problem.f_lower,
            "f_upper": problem.f_upper,
            "f_opt": problem.f_opt,
        },
        prior=prior_meta,
        f_opt=float(problem.f_opt),
        alpha=float(config.get("alpha", 0.5)),
        beta=float(beta_schedule.beta_at(1, problem.dim)),
        beta_schedule=beta_schedule.to_dict(),
    )
    return RawPriorSeedData(
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
        help="Override the run tag.  Defaults to config['tag'], then to the "
             "config file name (stem).",
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=_PROJECT_ROOT / "results",
        help="Root directory under which run dirs are created.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device, e.g. 'cpu', 'cuda', 'cuda:0', 'auto'.",
    )
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    """Manage automatic device attribution."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: Optional[list[str]] = None) -> Path:
    start_time = time.perf_counter()
    args = _parse_args(argv)
    config = _load_config(args.config)
    device_name = args.device if args.device is not None else config.get("device", "cpu")
    device = resolve_device(device_name)

    problem = build_oriented_benchmark(config)
    prior = GaussianPrior.from_config(config["prior"], dim=problem.dim)
    beta_schedule = BetaSchedule.from_config(config)

    # Priority: CLI --tag > config 'tag' > config file stem.  The final
    # fallback keeps run-id generation safe when a config omits 'tag'
    # (make_run_id would otherwise crash on None) and stays discriminant
    # (the file name usually encodes benchmark + prior, e.g.
    # "branin_2d_bad_prior").
    tag = args.tag or config.get("tag") or args.config.stem
    logger = RunLogger.create(
        experiment=config["experiment"],
        tag=tag,
        results_root=args.results_root,
    )
    log = logger.setup_file_logger("run_exp_b")

    logger.save_config(config)
    logger.save_metadata(extra={
        "experiment":     config["experiment"],
        "benchmark":      problem.name,
        "base_benchmark": problem.inner.name,
        "tag":            tag,
        "seeds":          list(config["seeds"]),
        "budget":         int(config["budget"]),
        "noise_std":      float(problem.default_noise_std or 0.0),
        "methods":        list(config["methods"]),
        "bounds":         problem.bounds.detach().cpu().tolist(),
        "prior":          prior.to_dict(),
        "alpha":          float(config.get("alpha", 0.5)),
        "beta":           float(beta_schedule.beta_at(1, problem.dim)),
        "beta_schedule":  beta_schedule.to_dict(),
        "f_opt":          float(problem.f_opt),
        "orientation": {
            "f_min": problem.f_min,
            "f_max": problem.f_max,
            "direction": problem.direction,
            "f_lower": problem.f_lower,
            "f_upper": problem.f_upper,
            "f_opt": problem.f_opt,
        },
    })

    log.info("=" * 60)
    log.info(f"Run: {logger.run_id()}")
    log.info(f"  device    = {device}")
    log.info(f"  benchmark = {problem.name}  dim={problem.dim}")
    log.info(f"  bounds    = {problem.bounds.detach().cpu().tolist()}")
    log.info(f"  prior     = {prior}  (label={config.get('prior', {}).get('label')})")
    log.info(f"  alpha     = {config.get('alpha', 0.5)}  beta = {beta_schedule!r}")
    log.info(f"  f_opt     = {problem.f_opt:.4f}  (oriented, raw scale; direction={problem.direction})")
    log.info(f"  noise_std = {problem.default_noise_std}  (raw oriented scale)")
    log.info(f"  n_initial = {config['initial_design_size']}  budget = {config['budget']}")
    log.info(f"  methods   = {list(config['methods'])}")
    log.info(f"  seeds     = {list(config['seeds'])}")
    log.info("=" * 60)

    for seed in config["seeds"]:
        raw_seed = _run_one_seed(config, problem, prior, int(seed), log, device)
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
