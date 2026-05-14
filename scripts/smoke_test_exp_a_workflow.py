"""End-to-end smoke check of the new Experiment A pipeline.

Runs the three stages in order on the 1-D SineTrend benchmark, using
``experiments/exp_a_target_matching/configs/smoke_1d.yaml``:

    run.py      -> raw/<benchmark>/seed_NNN.json
    process.py  -> processed/per_seed/<benchmark>/seed_NNN_processed.json
    analyze.py  -> processed/aggregate/<benchmark>/*.json

Used as a fast pre-flight before launching larger configs.  Not a unit test;
intentionally minimal.

Usage::

    python scripts/smoke_test_exp_a_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.exp_a_target_matching import analyze, process, run


def main() -> None:
    config = (
        _PROJECT_ROOT
        / "experiments"
        / "exp_a_target_matching"
        / "configs"
        / "smoke_1d.yaml"
    )
    run_dir = run.main(["--config", str(config)])
    process.main(["--run-dir", str(run_dir)])
    analyze.main(["--run-dir", str(run_dir)])
    print(f"\nSmoke pipeline OK.  Outputs under: {run_dir}")


if __name__ == "__main__":
    main()
