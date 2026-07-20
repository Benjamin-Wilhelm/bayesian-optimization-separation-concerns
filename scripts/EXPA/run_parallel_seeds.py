#!/usr/bin/env python
"""Seed-parallel driver for the Experiment A configs.

Unlike ``runA_all_cpu.sh`` (one process per config, seeds run sequentially),
this schedules every ``(config, seed)`` pair as an independent job across a
process pool so all idle cores are used.

Every seed is self-seeded (``_run_one_seed`` calls ``torch.manual_seed(seed)``
and re-derives the Sobol initial design from ``seed``), so scheduling the seeds
across a pool yields byte-identical results to the sequential runner while
saturating the cores.

Each worker is pinned to a single BLAS/OMP thread; parallelism comes from
running many one-seed jobs at once, not from threading inside one seed.

Each launch creates a FRESH timestamped run dir per tag and keeps it, so
re-launching never overwrites a previous run (the date makes every launch
unique).  Pass ``--reuse-existing`` to instead fill the newest existing dir
for a tag, to resume a partial run.  Seed files are one-per-file, so
concurrent writes never collide.

Usage::

    .venv/bin/python -m scripts.EXPA.run_parallel_seeds --max-workers 26
"""
from __future__ import annotations

import os

# Pin math-library threads BEFORE torch is imported so every worker stays
# single-threaded (parallelism is across seeds, not within one).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from benchmarks import build_benchmark  # noqa: E402
from common.acquisition_utils import BetaSchedule  # noqa: E402
from common.logging import RunLogger  # noqa: E402
import experiments.exp_a_target_matching.run as R  # noqa: E402

EXPERIMENT = "exp_a_target_matching"

# Config sub-directory (matches runA_all_cpu.sh).  Change here to run a
# different beta/kernel family.
CONFIG_DIR = (_ROOT / "experiments" / "exp_a_target_matching"
              / "configs" / "ADAPTATIVE_MATERN_BETA")

# Config files to run, relative to CONFIG_DIR.  Comment out the ones you do
# not want (mirrors the CONFIGS array in runA_all_cpu.sh).
CONFIGS = [
    # "linear_1d_short.yaml",
    # "exponential_1d_short.yaml",
    # "branin_2d_short.yaml",
    "hartmann_6d_full.yaml",
    # "gaussian_trap_1d_short.yaml",
    # "gaussian_trap_1d_short_thin.yaml",
    # "gaussian_trap_1d_short_shifted.yaml",
    # "gaussian_trap_1d_short_deep.yaml",
    # "sine_trend_1d_full.yaml",
    # "forrester_1d_full.yaml",
    # "damped_oscillatory_trap_1d_short.yaml",
]


class _NullLog:
    """Silent logger stand-in (per-seed workers do not write the run log)."""

    def info(self, *a, **k):
        pass

    warning = error = debug = info


def _run_one_seed(config: dict, run_dir: str, seed: int):
    """Execute a single seed and persist its raw JSON.  Runs in a worker."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    problem = build_benchmark(config)
    log = _NullLog()
    try:
        raw = R._run_one_seed(config, problem, seed, log, device)
        RunLogger(Path(run_dir)).save_seed_result(
            problem.name, seed, raw.to_dict()
        )
        return (Path(run_dir).name, seed, True, "")
    except Exception as exc:  # keep one bad seed from killing the pool
        return (Path(run_dir).name, seed, False, repr(exc))


def _resolve_run_dir(tag: str, results_root: Path,
                     reuse: bool = False) -> RunLogger:
    """Return the run dir for ``tag``.

    Default (``reuse=False``): always create a FRESH timestamped dir
    ``YYYY-MM-DD_HH-MM-SS_<tag>`` and keep it, so re-launching a config never
    overwrites a previous run — the date makes every launch unique.

    ``reuse=True`` restores the resume behaviour: fill the newest existing dir
    matching the tag (one-per-seed files, so missing seeds are added) instead
    of creating a new one.  Use it only to complete a partial/crashed run.
    """
    if reuse:
        runs_root = results_root / EXPERIMENT / "runs"
        existing = sorted(runs_root.glob(f"*_{tag}")) if runs_root.exists() else []
        if existing:
            return RunLogger(existing[-1])
    return RunLogger.create(EXPERIMENT, tag, results_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--results-root", type=Path, default=_ROOT / "results")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="Resume: fill the newest existing run dir for each tag "
                         "instead of creating a fresh dated dir. Off by default "
                         "so re-launching never overwrites a previous run.")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    jobs = []
    for name in CONFIGS:
        config = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        tag = config.get("tag") or Path(name).stem
        problem = build_benchmark(config)
        beta_schedule = BetaSchedule.from_config(config)
        logger = _resolve_run_dir(tag, args.results_root,
                                  reuse=args.reuse_existing)
        logger.save_config(config)
        logger.save_metadata(extra={
            "experiment": config["experiment"],
            "benchmark": problem.name,
            "tag": tag,
            "seeds": list(config["seeds"]),
            "target": float(config["target"]),
            "budget": int(config["budget"]),
            "noise_std": float(problem.default_noise_std or 0.0),
            "methods": list(config["methods"]),
            "bounds": problem.bounds.detach().cpu().tolist(),
            "beta": float(beta_schedule.beta_at(1, problem.dim)),
            "beta_schedule": beta_schedule.to_dict(),
            "launcher": "run_parallel_seeds",
        }, repo_root=_ROOT)
        seeds = [int(s) for s in config["seeds"]]
        for seed in seeds:
            jobs.append((config, str(logger.run_dir), seed))
        print(f"queued {tag}: {len(seeds)} seeds -> {logger.run_dir.name}",
              flush=True)

    total = len(jobs)
    print(f"\n{total} seed-jobs over {args.max_workers} workers\n", flush=True)

    t0 = time.perf_counter()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run_one_seed, *j) for j in jobs]
        for fut in as_completed(futs):
            run_name, seed, ok, err = fut.result()
            done += 1
            if not ok:
                failed += 1
            elapsed = time.perf_counter() - t0
            status = "OK " if ok else "FAIL"
            msg = (f"[{done}/{total}] {status} {run_name} seed {seed:03d}  "
                   f"({elapsed/60:.1f} min elapsed)")
            if not ok:
                msg += f"  ERROR: {err}"
            print(msg, flush=True)

    print(f"\nDONE: {done - failed}/{total} ok, {failed} failed, "
          f"{(time.perf_counter() - t0)/60:.1f} min total.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
