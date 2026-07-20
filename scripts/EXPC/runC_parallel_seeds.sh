#!/usr/bin/env bash
# Seed-parallel launcher for the Experiment C non-smoke configs.
#
# Unlike runC_all_cpu.sh (one process per config, seeds run sequentially),
# this wraps run_parallel_seeds.py, which schedules every (config, seed) job
# across a process pool so all idle cores are used.  Each worker is pinned to
# a single BLAS/OMP thread; parallelism comes from running many one-seed jobs
# at once.
#
# Usage (from the repo root):
#   bash scripts/EXPC/runC_parallel_seeds.sh [MAX_WORKERS]
#   nohup bash scripts/EXPC/runC_parallel_seeds.sh 26 &   # detached
#
# MAX_WORKERS defaults to (number of cores - 2).
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make `scripts` importable as a package regardless of the caller's cwd.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LOG_DIR="logsC_cpu_runs"
DEFAULT_WORKERS="$(( $(nproc) - 2 ))"
[[ "$DEFAULT_WORKERS" -lt 1 ]] && DEFAULT_WORKERS=1
MAX_WORKERS="${1:-$DEFAULT_WORKERS}"

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/parallel_seeds.out"

echo "Launching seed-parallel EXPC run: ${MAX_WORKERS} workers -> ${LOG_FILE}"

"$PYTHON" -m scripts.EXPC.run_parallel_seeds --max-workers "$MAX_WORKERS" \
  > "${LOG_FILE}" 2>&1

status=$?
if [[ "$status" -ne 0 ]]; then
  echo "Parallel run exited with status ${status}. Check ${LOG_FILE}."
  exit "$status"
fi
echo "All raw runs finished. See ${LOG_FILE}."
