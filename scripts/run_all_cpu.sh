#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_DIR="experiments/exp_a_target_matching/configs"
LOG_DIR="logs_cpu_runs"
MAX_PARALLEL=2

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"              # Linux / macOS
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"      # Windows Git Bash
else
  PYTHON="python"                        # fallback
fi

echo "Using Python: ${PYTHON}"
"${PYTHON}" --version

mkdir -p "${LOG_DIR}"

CONFIGS=(
  
  "exponential_1d_short.yaml"


)

run_one() {
  local cfg="$1"
  local name="${cfg%.yaml}"
  local log_file="${LOG_DIR}/${name}.out"

  {
    echo "============================================================"
    echo "Running raw BO run for ${name}"
    echo "Config: ${CONFIG_DIR}/${cfg}"
    echo "Started at: $(date)"
    echo "Python: ${PYTHON}"
    echo "============================================================"

    "${PYTHON}" -m experiments.exp_a_target_matching.run \
      --config "${CONFIG_DIR}/${cfg}" \
      --tag "${name}"

    echo "Finished raw BO run for ${name} at: $(date)"
  } > "${log_file}" 2>&1
}

for cfg in "${CONFIGS[@]}"; do
  echo "Launching ${cfg}"

  run_one "${cfg}" &

  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 5
  done
done

wait

echo "All raw runs finished."