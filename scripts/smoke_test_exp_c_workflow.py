"""End-to-end smoke check of the Experiment C pipeline (both modes).

Runs the three stages in order for a tiny fixed-scalarization config and a tiny
pareto-front config on ZDT1 (d=2)::

    run.py      -> raw/<benchmark>/seed_NNN.json
    process.py  -> processed/per_seed/<benchmark>/seed_NNN_processed.json
    analyze.py  -> figures/*.png|pdf + processed/aggregate_summary.json

Used as a fast pre-flight before launching larger configs.  Not a unit test.

Usage::

    python scripts/smoke_test_exp_c_workflow.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.exp_c_multiobjective import analyze, process, run

_CONFIGS = _PROJECT_ROOT / "experiments" / "exp_c_multiobjective" / "configs"


def _workflow(config: Path) -> None:
    print(f"\n=== {config.name} ===")
    run_dir = run.main(["--config", str(config)])
    process.main(["--run-dir", str(run_dir)])
    analyze.main(["--run-dir", str(run_dir)])
    print(f"OK: {run_dir}")


def main() -> None:
    _workflow(_CONFIGS / "fixed_scalarization" / "smoke_zdt1_d2.yaml")
    _workflow(_CONFIGS / "pareto_front" / "smoke_zdt1_d2.yaml")
    print("\nExperiment C smoke workflow completed.")


if __name__ == "__main__":
    main()
