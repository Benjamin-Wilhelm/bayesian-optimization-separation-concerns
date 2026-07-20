#!/usr/bin/env python
"""Seed-parallel driver for the Experiment C non-smoke configs.

Every ``(config, seed)`` pair is an independent, self-seeded job
(``_fixed_one_seed`` / ``_pareto_one_seed`` each call ``torch.manual_seed(seed)``),
so scheduling them across a process pool yields byte-identical results to the
sequential runner while saturating the idle cores.

Each worker is pinned to a single BLAS/OMP thread; parallelism comes from
running many one-seed jobs at once, not from threading inside one seed.

Each launch creates a FRESH timestamped run dir per tag and keeps it, so
re-launching never overwrites a previous run (the date makes every launch
unique).  Pass ``--reuse-existing`` to instead fill the newest existing dir
for a tag, to resume a partial run.  Seed files are one-per-file, so
concurrent writes never collide.

Usage::

    .venv/bin/python -m scripts.EXPC.run_parallel_seeds --max-workers 26
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

from benchmarks import multi_objective as mo_benchmarks  # noqa: E402
from common.logging import RunLogger  # noqa: E402
import experiments.exp_c_multiobjective.run as R  # noqa: E402
from experiments.exp_c_multiobjective.config import validate_config  # noqa: E402

EXPERIMENT = "exp_c_multiobjective"
CONFIG_ROOT = _ROOT / "experiments" / "exp_c_multiobjective" / "configs"

# (relative config path, explicit run tag).  Explicit tags disambiguate the
# identical filenames in the two sub-dirs and bypass the swapped pareto tags.

CONFIGS = [
    # --- log-beta schedule (original).  Explicit tags also FIX the swapped
    #     d2/d6 tags inside the pareto_front/*.yaml files. ---
    ("fixed_scalarization/LOG_BETA/zdt1_d2.yaml", "fixed_zdt1_d2_logbeta"),
    ("fixed_scalarization/LOG_BETA/zdt1_d6.yaml", "fixed_zdt1_d6_logbeta"),
    ("fixed_scalarization/LOG_BETA/zdt2_d2.yaml", "fixed_zdt2_d2_logbeta"),
    ("fixed_scalarization/LOG_BETA/zdt2_d6.yaml", "fixed_zdt2_d6_logbeta"),
    ("pareto_front/zdt1_d2.yaml", "pareto_zdt1_d2_logbeta"),
    ("pareto_front/zdt1_d6.yaml", "pareto_zdt1_d6_logbeta"),
    # NB: pareto_front/zdt2_d2.yaml actually holds input_dim=6 and zdt2_d6.yaml
    # holds input_dim=2 (mis-named source files); tags below match the CONTENT.
    ("pareto_front/zdt2_d2.yaml", "pareto_zdt2_d6_logbeta"),
    ("pareto_front/zdt2_d6.yaml", "pareto_zdt2_d2_logbeta"),
    # --- constant beta = 8 ---
    ("fixed_scalarization/beta=8/zdt1_d2.yaml", "fixed_zdt1_d2_beta8"),
    ("fixed_scalarization/beta=8/zdt1_d6.yaml", "fixed_zdt1_d6_beta8"),
    ("fixed_scalarization/beta=8/zdt2_d2.yaml", "fixed_zdt2_d2_beta8"),
    ("fixed_scalarization/beta=8/zdt2_d6.yaml", "fixed_zdt2_d6_beta8"),
    ("pareto_front/beta=8/zdt1_d2.yaml", "pareto_zdt1_d2_beta8"),
    ("pareto_front/beta=8/zdt1_d6.yaml", "pareto_zdt1_d6_beta8"),
    # Same content/filename swap as the log-beta pareto zdt2 files above.
    ("pareto_front/beta=8/zdt2_d2.yaml", "pareto_zdt2_d6_beta8"),
    ("pareto_front/beta=8/zdt2_d6.yaml", "pareto_zdt2_d2_beta8"),
]


class _NullLog:
    """Silent logger stand-in (per-seed workers do not write the run log)."""

    def info(self, *a, **k):
        pass

    warning = error = debug = info


def _run_one_seed(mode: str, config: dict, run_dir: str, bench_cfg: dict,
                  seed: int):
    """Execute a single seed and persist its raw JSON.  Runs in a worker."""
    torch.set_num_threads(1)
    benchmark = mo_benchmarks.build_benchmark(bench_cfg)
    log = _NullLog()
    try:
        if mode == "fixed_scalarization":
            raw = R._fixed_one_seed(config, benchmark, seed, log)
        elif mode == "pareto_front":
            raw = R._pareto_one_seed(config, benchmark, seed, log)
        else:
            raise ValueError(f"Unknown experiment_mode '{mode}'.")
        RunLogger(Path(run_dir)).save_seed_result(
            benchmark.name, seed, raw.to_dict()
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
    for rel, tag in CONFIGS:
        config = yaml.safe_load((CONFIG_ROOT / rel).read_text(encoding="utf-8"))
        validate_config(config)
        mode = config["experiment_mode"]
        logger = _resolve_run_dir(tag, args.results_root,
                                  reuse=args.reuse_existing)
        logger.save_config(config)
        logger.save_metadata(extra={
            "experiment": EXPERIMENT,
            "experiment_mode": mode,
            "benchmark": config["benchmark"]["name"],
            "tag": tag,
            "seeds": [int(config["seeds"]["start"]), int(config["seeds"]["end"])],
            "budget": int(config["budget"]),
            "launcher": "run_parallel_seeds",
        }, repo_root=_ROOT)
        s0, s1 = int(config["seeds"]["start"]), int(config["seeds"]["end"])
        bench_name = config["benchmark"]["name"]
        raw_dir = logger.run_dir / "raw" / bench_name
        queued = skipped = 0
        for seed in range(s0, s1 + 1):
            # Real resume: when filling an existing dir, don't recompute a seed
            # whose result file is already on disk. Off by default (fresh dir
            # has none), so a normal launch still queues every seed.
            if args.reuse_existing and (raw_dir / f"seed_{seed:03d}.json").exists():
                skipped += 1
                continue
            jobs.append((mode, config, str(logger.run_dir),
                         config["benchmark"], seed))
            queued += 1
        note = f" ({skipped} already done, skipped)" if skipped else ""
        print(f"queued {tag}: {queued} seeds in {s0}..{s1}{note} "
              f"-> {logger.run_dir.name}", flush=True)

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
            msg = f"[{done}/{total}] {status} {run_name} seed {seed:03d}  ({elapsed/60:.1f} min elapsed)"
            if not ok:
                msg += f"  ERROR: {err}"
            print(msg, flush=True)

    print(f"\nDONE: {done - failed}/{total} ok, {failed} failed, "
          f"{(time.perf_counter() - t0)/60:.1f} min total.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
