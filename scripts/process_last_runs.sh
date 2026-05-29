#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="results/exp_a_target_matching/runs"

# Détection robuste du Python à utiliser.
# Ordre :
# 1. venv Linux/macOS
# 2. venv Windows, utile si lancé depuis Git Bash sur Windows
# 3. python disponible globalement
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

# Vérifie que le dossier de résultats existe.
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "ERROR: results directory not found: $RESULTS_ROOT"
  exit 1
fi

patterns=(
  "*linear_1d_short"
  "*exponential_1d_short"
  "*gaussian_trap_1d_short"
  "*damped_oscillatory_trap_1d_short"
)

for pattern in "${patterns[@]}"; do
  echo "============================================================"
  echo "Looking for latest run matching: ${pattern}"

  # mapfile évite les soucis avec les espaces dans les chemins.
  mapfile -t matches < <(find "$RESULTS_ROOT" -maxdepth 1 -type d -name "$pattern" -printf "%T@ %p\n" \
    | sort -nr \
    | awk '{$1=""; sub(/^ /,""); print}')

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "WARNING: no run found for pattern ${pattern}"
    continue
  fi

  run_dir="${matches[0]}"

  echo "Selected run: $run_dir"

  echo "Processing $run_dir"
  "$PYTHON" -m experiments.exp_a_target_matching.process --run-dir "$run_dir"

  echo "Analyzing $run_dir"
  "$PYTHON" -m experiments.exp_a_target_matching.analyze --run-dir "$run_dir"

  echo "Done: $run_dir"
done

echo "============================================================"
echo "All available runs processed and analyzed."