"""End-to-end smoke check of the Experiment B pipeline.

Runs the three stages in order on the 1-D Forrester benchmark (raw oriented
scale, good prior), using
``experiments/exp_b_prior_guided/configs/smoke_forrester.yaml``:

    run.py      -> raw/<benchmark>/seed_NNN.json
    process.py  -> processed/per_seed/<benchmark>/seed_NNN_processed.json
    analyze.py  -> processed/aggregate/<benchmark>/*.json

Used as a fast pre-flight before launching larger configs.  Not a unit test;
intentionally minimal.

Usage::

    python scripts/smoke_test_exp_b_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.exp_b_prior_guided import analyze, process, run


def main() -> None:
    config = (
        _PROJECT_ROOT
        / "experiments"
        / "exp_b_prior_guided"
        / "configs"
        / "smoke_forrester.yaml"
    )
    run_dir = run.main(["--config", str(config)])
    process.main(["--run-dir", str(run_dir)])
    analyze.main(["--run-dir", str(run_dir)])
    print(f"\nSmoke pipeline OK.  Outputs under: {run_dir}")


if __name__ == "__main__":
    main()
