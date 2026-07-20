#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="results/exp_c_multiobjective/runs"

# Usage:
#   bash scripts/EXPC/processC_last_runs.sh
#   bash scripts/EXPC/processC_last_runs.sh 8
#
# Argument optionnel :
#   $1 = nombre de runs récents à traiter
#
# Un batch EXPC complet (runC_all_cpu.sh) produit 8 run-dirs partageant le
# même timestamp (fixed/pareto x zdt1/zdt2 x d2/d6), d'où le défaut à 8.
#
# Cette étape calcule les métriques par seed (processed/per_seed) via
# experiments.exp_c_multiobjective.process. Les figures sont générées
# ensuite par plotC_last_runs.sh (experiments.exp_c_multiobjective.analyze).
N_LATEST="${1:-8}"

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
echo "Looking for the ${N_LATEST} latest EXPC runs"

mapfile -t matches < <(
  find "$RESULTS_ROOT" \
    -maxdepth 1 \
    -type d \
    \( \
      -name "*fixed_zdt1_d2" \
      -o -name "*fixed_zdt1_d6" \
      -o -name "*fixed_zdt2_d2" \
      -o -name "*fixed_zdt2_d6" \
      -o -name "*pareto_zdt1_d2" \
      -o -name "*pareto_zdt1_d6" \
      -o -name "*pareto_zdt2_d2" \
      -o -name "*pareto_zdt2_d6" \
      -o -name "*smoke_zdt1_d2" \
      -o -name "*smoke_zdt2_d2" \
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
  "$PYTHON" -m experiments.exp_c_multiobjective.process \
    --run-dir "$run_dir"

  echo "Done: $run_dir"
done

echo "============================================================"
echo "The ${#matches[@]} latest available runs have been processed."
echo "Run plotC_last_runs.sh to generate the figures."
