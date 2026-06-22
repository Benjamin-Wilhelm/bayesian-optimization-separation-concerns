#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="results/exp_b_prior_guided/runs"

# Usage:
#   bash scripts/process_latest_short_runs.sh
#   bash scripts/process_latest_short_runs.sh 5
#
# Argument optionnel :
#   $1 = nombre de runs récents à traiter
N_LATEST="${1:-1}"

if ! [[ "$N_LATEST" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_LATEST must be a strictly positive integer."
  echo "Usage: bash $0 [N_LATEST]"
  exit 1
fi

# Détection robuste du Python à utiliser.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "ERROR: no Python interpreter found."
  echo "Expected one of:"
  echo "  .venv/bin/python"
  echo "  .venv/Scripts/python.exe"
  echo "  python in PATH"
  exit 1
fi

echo "Using Python: $PYTHON"

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "ERROR: results directory not found: $RESULTS_ROOT"
  exit 1
fi

echo "============================================================"
echo "Looking for the ${N_LATEST} latest short runs"

mapfile -t matches < <(
  find "$RESULTS_ROOT" \
    -maxdepth 1 \
    -type d \
    \( \
      -name "*double_well_1d_bad_prior" \
      -o -name "*double_well_1d_good_prior" \
      -o -name "*branin_2d_bad_prior" \
      -o -name "*branin_2d_good_prior" \
      -o -name "*hartmann_6d_bad_prior" \
      -o -name "*hartmann_6d_good_prior" \
      -o -name "*hartmann_3d_bad_prior" \
      -o -name "*hartmann_3d_good_prior" \
      -o -name "*forrester_1d_good_prior" \
      -o -name "*forrester_1d_bad_prior" \
      -o -name "*diag_double_well_standardized" \
    \) \
    -printf "%T@ %p\n" \
    | sort -nr \
    | head -n "$N_LATEST" \
    | awk '{$1=""; sub(/^ /,""); print}'
)

if [[ ${#matches[@]} -eq 0 ]]; then
  echo "ERROR: no matching run found."
  exit 1
fi

echo "Selected runs:"
for run_dir in "${matches[@]}"; do
  echo "  - $run_dir"
done

for run_dir in "${matches[@]}"; do
  echo "============================================================"
  echo "Processing $run_dir"
  "$PYTHON" -m experiments.exp_b_prior_guided.process \
    --run-dir "$run_dir"

  echo "Analyzing $run_dir"
  "$PYTHON" -m experiments.exp_b_prior_guided.analyze \
    --run-dir "$run_dir"

  echo "Done: $run_dir"
done

echo "============================================================"
echo "The ${#matches[@]} latest available runs have been processed and analyzed."