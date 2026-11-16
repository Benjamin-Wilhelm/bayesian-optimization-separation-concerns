"""Experiment C — multi-objective BO, raw data generation.

One shared multi-objective pipeline with two explicitly separated runners:

- :func:`run_fixed_scalarization` — RIGHT vs WRONG at fixed preferences;
- :func:`run_pareto_front` — RIGHT Algorithm 1 (shared posterior) vs
  RIGHT/WRONG Algorithm 2 (independent scalarized subruns, equal global
  budget, round-robin schedule).

This script writes ONLY raw data (``raw/<benchmark>/seed_NNN.json``).  Metrics
live in ``process.py``; figures in ``analyze.py``.

Reproducibility: the optimization preferences and the Bayes-regret evaluation
preferences use **disjoint** RNG seeds derived from the run seed, so the
evaluation preferences are independent of anything the algorithms use.

Usage::

    python -m experiments.exp_c_multiobjective.run \\
        --config experiments/exp_c_multiobjective/configs/fixed_scalarization/zdt1_d2.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.acquisition_utils import BetaSchedule
from common.initial_design import sobol_design
from common.logging import RunLogger
from benchmarks import multi_objective as mo_benchmarks
from experiments.exp_c_multiobjective.config import budget_split, validate_config
from data.raw_multiobjective import (
    FixedCase,
    FixedSeedData,
    FixedStepRecord,
    MethodTrajectory,
    ParetoEvaluation,
    ParetoMethod,
    ParetoSeedData,
    ParetoSubrunMeta,
)
from common.lambda_sampling import (
    fixed_angle_grid_2d,
    sample_positive_sphere,
)
from common.hypervolume import exact_hypervolume_2d
from common.multi_objective_metrics import reference_scalar_optimum
from experiments.exp_c_multiobjective.methods import (
    fit_right_models,
    fit_wrong_model,
    make_gp_builder,
    pairing_hash,
    right_next_point,
    wrong_next_point,
    wrong_targets,
)
from common.scalarizations import scalarize

Tensor = torch.Tensor

# RNG seed offsets keeping optimization and evaluation preferences disjoint.
_EVAL_SEED_OFFSET = 10_000_000
_OPT_SEED_ALG1_OFFSET = 20_000_000
_OPT_SEED_ALG2_OFFSET = 30_000_000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _subrun_design_seed(seed: int, subrun_index: int) -> int:
    """Deterministic distinct Sobol seed per Algorithm-2 subrun."""
    return int(seed) * 97 + int(subrun_index) + 1


def _to_list(t: Tensor) -> list:
    return t.detach().cpu().tolist()


def _beta_schedule(config: dict[str, Any]) -> BetaSchedule:
    acq = config.get("acquisition", {}) or {}
    return BetaSchedule.from_config(acq)


def _acq_settings(config: dict[str, Any]) -> tuple[int, int]:
    acq = config.get("acquisition", {}) or {}
    return int(acq.get("num_restarts", 10)), int(acq.get("raw_samples", 256))


# ---------------------------------------------------------------------------
# Fixed scalarization
# ---------------------------------------------------------------------------

def _fixed_one_method(
    *,
    architecture: str,
    config: dict[str, Any],
    benchmark: mo_benchmarks.MultiObjectiveBenchmark,
    X_init: Tensor,
    model_y_init: Tensor,
    lambda_vec: Tensor,
    scalarization: str,
    reference_point: Tensor,
    beta_schedule: BetaSchedule,
    budget: int,
    n_init: int,
    noise_std: float,
    log,
) -> MethodTrajectory:
    """Run one method (right/wrong) of a fixed-scalarization case."""
    num_restarts, raw_samples = _acq_settings(config)
    builder = make_gp_builder(config, noise_std)
    bounds = benchmark.bounds
    dim = benchmark.input_dim

    X = X_init.clone()
    model_Y = model_y_init.clone()          # (n, m) observed (noisy) objectives

    trajectory: list[FixedStepRecord] = []
    hypers: list[dict[str, Any]] = []
    n_bo = budget - n_init

    for it in range(1, n_bo + 1):
        beta_t = beta_schedule.beta_at(it, dim)
        if architecture == "right":
            model = fit_right_models(builder, X, model_Y)
            cand, acq_val = right_next_point(
                model, lambda_vec, beta_t, scalarization, reference_point,
                bounds, num_restarts, raw_samples,
            )
        elif architecture == "wrong":
            g = wrong_targets(model_Y, lambda_vec, scalarization, reference_point)
            model = fit_wrong_model(builder, X, g)
            cand, acq_val = wrong_next_point(
                model, beta_t, bounds, num_restarts, raw_samples,
            )
        else:
            raise ValueError(f"Unknown architecture '{architecture}'.")
        hypers.append(builder.extract_hyperparameters(model))

        obs = benchmark.observe(cand, noise_std=noise_std)
        # Evaluation uses the noiseless oriented value; the GP saw model_y.
        scal_val = float(
            scalarize(obs["oriented_y"], lambda_vec, reference_point, scalarization)
            .reshape(-1)[0].item()
        )
        trajectory.append(FixedStepRecord(
            iteration=it,
            x=_to_list(cand.reshape(-1)),
            raw_y=_to_list(obs["raw_y"].reshape(-1)),
            oriented_y=_to_list(obs["oriented_y"].reshape(-1)),
            model_y=_to_list(obs["model_y"].reshape(-1)),
            scalarized_value=scal_val,
            acq_value=float(acq_val.item()),
            beta=float(beta_t),
        ))
        X = torch.cat([X, cand], dim=0)
        model_Y = torch.cat([model_Y, obs["model_y"]], dim=0)

    train_label = (
        "oriented model_y per objective (RIGHT)" if architecture == "right"
        else f"rho_lambda(model_y) [{scalarization}] (WRONG)"
    )
    return MethodTrajectory(
        method_name=architecture,
        train_label=train_label,
        trajectory=trajectory,
        gp_hyperparameters=hypers,
    )


def _fixed_one_seed(
    config: dict[str, Any],
    benchmark: mo_benchmarks.MultiObjectiveBenchmark,
    seed: int,
    log,
) -> FixedSeedData:
    torch.manual_seed(seed)
    budget = int(config["budget"])
    n_init = int(config["initial_design_size"])
    noise_std = float((config.get("noise", {}) or {}).get("std", 0.0))
    beta_schedule = _beta_schedule(config)
    ref = benchmark.reference_point
    front_res = int((config.get("reference", {}) or {}).get("pareto_front_resolution", 10000))
    front_oriented = benchmark.true_pareto_front_oriented(front_res)

    angles = list(config["lambda"]["angles_degrees"])
    lam_grid = fixed_angle_grid_2d(angles)   # (K, 2)

    # Shared initial design for the whole seed (deterministic from the seed);
    # RIGHT and WRONG in every case see identical initial observations.
    X_init = sobol_design(benchmark.bounds, n=n_init, seed=seed)
    obs_init = benchmark.observe(X_init, noise_std=noise_std)
    model_y_init = obs_init["model_y"]

    log.info(
        f"  seed={seed}: init X{tuple(X_init.shape)} "
        f"raw_y in [{obs_init['raw_y'].min():.3f}, {obs_init['raw_y'].max():.3f}]"
    )

    cases: list[FixedCase] = []
    for scal in config["scalarizations"]:
        for k, angle in enumerate(angles):
            lam = lam_grid[k]
            rho_star = float(
                reference_scalar_optimum(front_oriented, lam, scal, ref).item()
            )
            ph = pairing_hash(lam, X_init)
            methods: dict[str, MethodTrajectory] = {}
            for architecture in config["methods"]:
                methods[architecture] = _fixed_one_method(
                    architecture=architecture,
                    config=config,
                    benchmark=benchmark,
                    X_init=X_init,
                    model_y_init=model_y_init,
                    lambda_vec=lam,
                    scalarization=scal,
                    reference_point=ref,
                    beta_schedule=beta_schedule,
                    budget=budget,
                    n_init=n_init,
                    noise_std=noise_std,
                    log=log,
                )
            cases.append(FixedCase(
                scalarization=scal,
                lambda_angle_degrees=float(angle),
                lambda_vector=_to_list(lam),
                rho_star=rho_star,
                pairing_hash=ph,
                initial_X=_to_list(X_init),
                initial_raw_y=_to_list(obs_init["raw_y"]),
                initial_oriented_y=_to_list(obs_init["oriented_y"]),
                methods=methods,
            ))
            log.info(
                f"    [{scal:<11} angle={angle:>4}] rho*={rho_star:.4f} "
                f"methods={list(methods)}"
            )

    metadata = _fixed_metadata(config, benchmark, seed, beta_schedule)
    return FixedSeedData(
        metadata=metadata,
        benchmark=benchmark.metadata(),
        cases=cases,
    )


def _fixed_metadata(config, benchmark, seed, beta_schedule) -> dict[str, Any]:
    return {
        "experiment": config["experiment_mode"],
        "experiment_mode": "fixed_scalarization",
        "problem_type": "synthetic",
        "benchmark": benchmark.name,
        "input_dim": benchmark.input_dim,
        "output_dim": benchmark.output_dim,
        "orientation": "maximize",
        "normalization_metadata": benchmark.normalization_metadata(),
        "reference_point": _to_list(benchmark.reference_point),
        "seed": int(seed),
        "budget": int(config["budget"]),
        "initial_design_size": int(config["initial_design_size"]),
        "noise_std": float((config.get("noise", {}) or {}).get("std", 0.0)),
        "use_fixed_noise": bool((config.get("model", {}) or {}).get("use_fixed_noise", True)),
        "scalarizations": list(config["scalarizations"]),
        "methods": list(config["methods"]),
        "angles_degrees": list(config["lambda"]["angles_degrees"]),
        "beta_schedule": beta_schedule.to_dict(),
        "pareto_front_resolution": int(
            (config.get("reference", {}) or {}).get("pareto_front_resolution", 10000)
        ),
    }


def run_fixed_scalarization(
    config: dict[str, Any],
    logger: RunLogger,
    log,
    device: torch.device,
) -> None:
    """Run every seed of a ``fixed_scalarization`` config and save raw data."""
    benchmark = mo_benchmarks.build_benchmark(config["benchmark"])
    seeds = range(int(config["seeds"]["start"]), int(config["seeds"]["end"]) + 1)
    for seed in seeds:
        raw = _fixed_one_seed(config, benchmark, int(seed), log)
        path = logger.save_seed_result(benchmark.name, int(seed), raw.to_dict())
        log.info(f"  -> saved {path.relative_to(logger.run_dir.parents[2])}")


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def _mk_eval(
    *, global_iteration, subrun_index, local_iteration, is_init, lam, cand, obs,
    acq_val, beta,
) -> ParetoEvaluation:
    return ParetoEvaluation(
        global_iteration=int(global_iteration),
        subrun_index=int(subrun_index),
        local_iteration=int(local_iteration),
        is_initial_design=bool(is_init),
        lambda_vector=(_to_list(lam.reshape(-1)) if lam is not None else []),
        x=_to_list(cand.reshape(-1)),
        raw_y=_to_list(obs["raw_y"].reshape(-1)),
        oriented_y=_to_list(obs["oriented_y"].reshape(-1)),
        model_y=_to_list(obs["model_y"].reshape(-1)),
        acq_value=(float(acq_val) if acq_val is not None else None),
        beta=(float(beta) if beta is not None else None),
    )


def _run_algorithm_1(
    *, config, benchmark, seed, scalarization, reference_point, beta_schedule,
    budget, noise_std, log,
) -> ParetoMethod:
    """RIGHT Algorithm 1: one shared per-objective posterior; online lambdas."""
    num_restarts, raw_samples = _acq_settings(config)
    n_init = int(config["algorithm_1"]["initial_design_size"])
    builder = make_gp_builder(config, noise_std)
    bounds = benchmark.bounds
    dim = benchmark.input_dim
    opt_gen = _generator(_OPT_SEED_ALG1_OFFSET + seed)

    X_init = sobol_design(bounds, n=n_init, seed=seed)
    obs_init = benchmark.observe(X_init, noise_std=noise_std)
    X = X_init.clone()
    model_Y = obs_init["model_y"].clone()

    evals: list[ParetoEvaluation] = []
    for j in range(n_init):
        evals.append(_mk_eval(
            global_iteration=j + 1, subrun_index=0, local_iteration=j + 1,
            is_init=True, lam=None, cand=X_init[j : j + 1],
            obs={k: v[j : j + 1] for k, v in obs_init.items()},
            acq_val=None, beta=None,
        ))

    opt_lambdas: list[list[float]] = []
    n_bo = budget - n_init
    for it in range(1, n_bo + 1):
        lam = sample_positive_sphere(1, benchmark.output_dim, generator=opt_gen)[0]
        model = fit_right_models(builder, X, model_Y)
        beta_t = beta_schedule.beta_at(it, dim)
        cand, acq_val = right_next_point(
            model, lam, beta_t, scalarization, reference_point,
            bounds, num_restarts, raw_samples,
        )
        obs = benchmark.observe(cand, noise_std=noise_std)
        evals.append(_mk_eval(
            global_iteration=n_init + it, subrun_index=0, local_iteration=it,
            is_init=False, lam=lam, cand=cand, obs=obs,
            acq_val=float(acq_val.item()), beta=beta_t,
        ))
        opt_lambdas.append(_to_list(lam))
        X = torch.cat([X, cand], dim=0)
        model_Y = torch.cat([model_Y, obs["model_y"]], dim=0)

    assert len(evals) == budget, (len(evals), budget)
    log.info(f"    [right_algorithm_1] {len(evals)} evals (n_init={n_init})")
    return ParetoMethod(
        method_name="right_algorithm_1",
        algorithm="algorithm_1",
        architecture="right",
        global_evaluations=evals,
        subruns=[],
        optimization_lambdas=opt_lambdas,
        global_order=[],
    )


def _run_one_subrun(
    *, architecture, config, benchmark, subrun_index, lam, local_budget,
    n_init_sub, X_sub_init, obs_sub_init, scalarization, reference_point,
    beta_schedule, noise_std,
) -> tuple[list[ParetoEvaluation], list[dict[str, Any]]]:
    """Run one independent Algorithm-2 subrun; return (local evals, hypers).

    Local evaluations are in local order: initial design first, then BO steps.
    """
    num_restarts, raw_samples = _acq_settings(config)
    builder = make_gp_builder(config, noise_std)
    bounds = benchmark.bounds
    dim = benchmark.input_dim

    X = X_sub_init.clone()
    model_Y = obs_sub_init["model_y"].clone()

    local_evals: list[ParetoEvaluation] = []
    for j in range(n_init_sub):
        local_evals.append(_mk_eval(
            global_iteration=-1, subrun_index=subrun_index, local_iteration=j + 1,
            is_init=True, lam=lam, cand=X_sub_init[j : j + 1],
            obs={k: v[j : j + 1] for k, v in obs_sub_init.items()},
            acq_val=None, beta=None,
        ))

    hypers: list[dict[str, Any]] = []
    n_bo_sub = local_budget - n_init_sub
    for it in range(1, n_bo_sub + 1):
        beta_t = beta_schedule.beta_at(it, dim)
        if architecture == "right":
            model = fit_right_models(builder, X, model_Y)
            cand, acq_val = right_next_point(
                model, lam, beta_t, scalarization, reference_point,
                bounds, num_restarts, raw_samples,
            )
        else:  # wrong
            g = wrong_targets(model_Y, lam, scalarization, reference_point)
            model = fit_wrong_model(builder, X, g)
            cand, acq_val = wrong_next_point(
                model, beta_t, bounds, num_restarts, raw_samples,
            )
        hypers.append(builder.extract_hyperparameters(model))
        obs = benchmark.observe(cand, noise_std=noise_std)
        local_evals.append(_mk_eval(
            global_iteration=-1, subrun_index=subrun_index,
            local_iteration=n_init_sub + it, is_init=False, lam=lam,
            cand=cand, obs=obs, acq_val=float(acq_val.item()), beta=beta_t,
        ))
        X = torch.cat([X, cand], dim=0)
        model_Y = torch.cat([model_Y, obs["model_y"]], dim=0)

    return local_evals, hypers


def _round_robin(
    local_sequences: list[list[ParetoEvaluation]],
) -> tuple[list[ParetoEvaluation], list[list[int]]]:
    """Interleave per-subrun local sequences into a global order.

    Order: (subrun 0 step 1), (subrun 1 step 1), ..., (subrun L-1 step 1),
    (subrun 0 step 2), ...  Initial-design points are the first local steps,
    so they are interleaved too.  Subruns that have run out are skipped.
    Assigns ``global_iteration`` (1-based) in this order.
    """
    L = len(local_sequences)
    max_local = max((len(s) for s in local_sequences), default=0)
    ordered: list[ParetoEvaluation] = []
    order_index: list[list[int]] = []
    g = 0
    for k in range(max_local):
        for ell in range(L):
            if k < len(local_sequences[ell]):
                g += 1
                ev = local_sequences[ell][k]
                ev.global_iteration = g
                ordered.append(ev)
                order_index.append([ell, ev.local_iteration])
    return ordered, order_index


def _run_algorithm_2(
    *, architecture, config, benchmark, seed, subrun_lambdas, subrun_designs,
    local_budgets, n_init_sub, scalarization, reference_point, beta_schedule,
    noise_std, log,
) -> ParetoMethod:
    """RIGHT or WRONG Algorithm 2: L independent scalarized subruns."""
    L = len(subrun_lambdas)
    local_sequences: list[list[ParetoEvaluation]] = []
    subrun_meta: list[ParetoSubrunMeta] = []
    for ell in range(L):
        lam = subrun_lambdas[ell]
        X_sub_init, obs_sub_init = subrun_designs[ell]
        local_evals, hypers = _run_one_subrun(
            architecture=architecture, config=config, benchmark=benchmark,
            subrun_index=ell, lam=lam, local_budget=local_budgets[ell],
            n_init_sub=n_init_sub, X_sub_init=X_sub_init,
            obs_sub_init=obs_sub_init, scalarization=scalarization,
            reference_point=reference_point, beta_schedule=beta_schedule,
            noise_std=noise_std,
        )
        local_sequences.append(local_evals)
        subrun_meta.append(ParetoSubrunMeta(
            subrun_index=ell,
            lambda_vector=_to_list(lam),
            local_budget=int(local_budgets[ell]),
            initial_design_size=int(n_init_sub),
            pairing_hash=pairing_hash(lam, X_sub_init),
            gp_hyperparameters=hypers,
        ))
    ordered, order_index = _round_robin(local_sequences)
    method_name = f"{architecture}_algorithm_2"
    assert len(ordered) == int(config["budget"]), (len(ordered), config["budget"])
    log.info(
        f"    [{method_name}] {len(ordered)} evals over L={L} subruns "
        f"(local budgets {list(local_budgets)})"
    )
    return ParetoMethod(
        method_name=method_name,
        algorithm="algorithm_2",
        architecture=architecture,
        global_evaluations=ordered,
        subruns=subrun_meta,
        optimization_lambdas=[_to_list(l) for l in subrun_lambdas],
        global_order=order_index,
    )


def _pareto_one_seed(
    config: dict[str, Any],
    benchmark: mo_benchmarks.MultiObjectiveBenchmark,
    seed: int,
    log,
) -> ParetoSeedData:
    torch.manual_seed(seed)
    budget = int(config["budget"])
    noise_std = float((config.get("noise", {}) or {}).get("std", 0.0))
    scalarization = str(config["scalarization"])
    beta_schedule = _beta_schedule(config)
    ref = benchmark.reference_point
    m = benchmark.output_dim

    front_res = int((config.get("reference", {}) or {}).get("pareto_front_resolution", 10000))
    front_oriented = benchmark.true_pareto_front_oriented(front_res)
    hv_star = exact_hypervolume_2d(front_oriented, ref)

    lam_cfg = config["lambda"]
    n_scal = int(lam_cfg["n_scalarizations_algorithm_2"])
    n_eval = int(lam_cfg["evaluation_samples"])

    # Evaluation preferences: independent RNG, shared across all methods.
    eval_gen = _generator(_EVAL_SEED_OFFSET + seed)
    eval_lambdas = sample_positive_sphere(n_eval, m, generator=eval_gen)

    methods: dict[str, ParetoMethod] = {}
    requested = list(config["methods"])

    if "right_algorithm_1" in requested:
        methods["right_algorithm_1"] = _run_algorithm_1(
            config=config, benchmark=benchmark, seed=seed,
            scalarization=scalarization, reference_point=ref,
            beta_schedule=beta_schedule, budget=budget, noise_std=noise_std,
            log=log,
        )

    need_alg2 = any(mth in requested for mth in ("right_algorithm_2", "wrong_algorithm_2"))
    if need_alg2:
        n_init_sub = int(config["algorithm_2"]["initial_design_size_per_subrun"])
        share_init = bool(config["algorithm_2"].get("share_initial_design_across_runs", False))
        local_budgets = budget_split(budget, n_scal)
        # Subrun preferences: independent RNG, shared by right & wrong Alg 2.
        alg2_gen = _generator(_OPT_SEED_ALG2_OFFSET + seed)
        subrun_lambdas = [
            sample_positive_sphere(1, m, generator=alg2_gen)[0] for _ in range(n_scal)
        ]
        # Initial designs: shared numerically between right & wrong Alg 2 (same
        # points) but each subrun keeps an INDEPENDENT posterior.
        subrun_designs = []
        for ell in range(n_scal):
            d_seed = seed if share_init else _subrun_design_seed(seed, ell)
            X_sub = sobol_design(benchmark.bounds, n=n_init_sub, seed=d_seed)
            obs_sub = benchmark.observe(X_sub, noise_std=noise_std)
            subrun_designs.append((X_sub, obs_sub))

        for architecture in ("right", "wrong"):
            name = f"{architecture}_algorithm_2"
            if name not in requested:
                continue
            methods[name] = _run_algorithm_2(
                architecture=architecture, config=config, benchmark=benchmark,
                seed=seed, subrun_lambdas=subrun_lambdas,
                subrun_designs=subrun_designs, local_budgets=local_budgets,
                n_init_sub=n_init_sub, scalarization=scalarization,
                reference_point=ref, beta_schedule=beta_schedule,
                noise_std=noise_std, log=log,
            )

    metadata = _pareto_metadata(config, benchmark, seed, beta_schedule, hv_star)
    return ParetoSeedData(
        metadata=metadata,
        benchmark=benchmark.metadata(),
        reference_point=_to_list(ref),
        hv_star=float(hv_star),
        evaluation_lambdas=_to_list(eval_lambdas),
        methods=methods,
    )


def _pareto_metadata(config, benchmark, seed, beta_schedule, hv_star) -> dict[str, Any]:
    lam_cfg = config["lambda"]
    return {
        "experiment": config["experiment_mode"],
        "experiment_mode": "pareto_front",
        "problem_type": "synthetic",
        "benchmark": benchmark.name,
        "input_dim": benchmark.input_dim,
        "output_dim": benchmark.output_dim,
        "orientation": "maximize",
        "normalization_metadata": benchmark.normalization_metadata(),
        "reference_point": _to_list(benchmark.reference_point),
        "seed": int(seed),
        "budget": int(config["budget"]),
        "scalarization": str(config["scalarization"]),
        "n_scalarizations_algorithm_2": int(lam_cfg["n_scalarizations_algorithm_2"]),
        "evaluation_samples": int(lam_cfg["evaluation_samples"]),
        "algorithm_1": dict(config.get("algorithm_1", {})),
        "algorithm_2": dict(config.get("algorithm_2", {})),
        "noise_std": float((config.get("noise", {}) or {}).get("std", 0.0)),
        "use_fixed_noise": bool((config.get("model", {}) or {}).get("use_fixed_noise", True)),
        "methods": list(config["methods"]),
        "beta_schedule": beta_schedule.to_dict(),
        "hv_star": float(hv_star),
        "pareto_front_resolution": int(
            (config.get("reference", {}) or {}).get("pareto_front_resolution", 10000)
        ),
        "eval_seed_offset": _EVAL_SEED_OFFSET,
        "opt_seed_alg1_offset": _OPT_SEED_ALG1_OFFSET,
        "opt_seed_alg2_offset": _OPT_SEED_ALG2_OFFSET,
    }


def run_pareto_front(
    config: dict[str, Any],
    logger: RunLogger,
    log,
    device: torch.device,
) -> None:
    """Run every seed of a ``pareto_front`` config and save raw data."""
    benchmark = mo_benchmarks.build_benchmark(config["benchmark"])
    seeds = range(int(config["seeds"]["start"]), int(config["seeds"]["end"]) + 1)
    for seed in seeds:
        raw = _pareto_one_seed(config, benchmark, int(seed), log)
        path = logger.save_seed_result(benchmark.name, int(seed), raw.to_dict())
        log.info(f"  -> saved {path.relative_to(logger.run_dir.parents[2])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=_PROJECT_ROOT / "results")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Path:
    start = time.perf_counter()
    args = _parse_args(argv)
    config = _load_config(args.config)
    validate_config(config)

    experiment = "exp_c_multiobjective"
    mode = config["experiment_mode"]
    device = resolve_device(args.device or config.get("device", "cpu"))

    tag = args.tag or config.get("tag") or args.config.stem
    logger = RunLogger.create(
        experiment=experiment, tag=tag, results_root=args.results_root,
    )
    log = logger.setup_file_logger("run_exp_c")
    logger.save_config(config)
    logger.save_metadata(extra={
        "experiment": experiment,
        "experiment_mode": mode,
        "benchmark": config["benchmark"]["name"],
        "tag": tag,
        "seeds": [int(config["seeds"]["start"]), int(config["seeds"]["end"])],
        "budget": int(config["budget"]),
    })

    log.info("=" * 64)
    log.info(f"Run: {logger.run_id()}  mode={mode}  device={device}")
    log.info(f"  benchmark = {config['benchmark']}")
    log.info(f"  seeds     = {config['seeds']}  budget = {config['budget']}")
    log.info("=" * 64)

    if mode == "fixed_scalarization":
        run_fixed_scalarization(config, logger, log, device)
    elif mode == "pareto_front":
        run_pareto_front(config, logger, log, device)
    else:
        raise ValueError(f"Unknown experiment_mode '{mode}'.")

    elapsed = time.perf_counter() - start
    log.info(f"Done in {elapsed:.2f}s ({elapsed / 60:.2f} min).")
    return logger.run_dir


if __name__ == "__main__":
    main()
