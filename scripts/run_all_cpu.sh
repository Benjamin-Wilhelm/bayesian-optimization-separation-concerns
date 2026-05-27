#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_DIR="experiments/exp_a_target_matching/configs"
LOG_DIR="logs_cpu_runs"
MAX_PARALLEL=2

mkdir -p "${LOG_DIR}"

CONFIGS=(
  "sine_trend_1d_full.yaml"
  "forrester_1d_full.yaml"
  "branin_2d_full.yaml"
  "hartmann_6d_full.yaml"
)

run_pipeline_one() {
  local cfg="$1"
  local name="${cfg%.yaml}"
  local log_file="${LOG_DIR}/${name}.out"

  {
    echo "============================================================"
    echo "Running pipeline for ${name}"
    echo "Config: ${CONFIG_DIR}/${cfg}"
    echo "Started at: $(date)"
    echo "============================================================"

    output=$(python -m experiments.exp_a_target_matching.run \
      --config "${CONFIG_DIR}/${cfg}" \
      --tag "${name}" 2>&1)

    echo "$output"

    run_dir=$(echo "$output" | grep -o "results/exp_a_target_matching/runs/[^ ]*" | tail -n 1)

    if [[ -z "${run_dir}" ]]; then
      echo "ERROR: Could not detect run_dir from run.py output."
      exit 1
    fi

    echo "Detected run_dir: ${run_dir}"

    echo "Processing ${run_dir}"
    python -m experiments.exp_a_target_matching.process \
      --run-dir "${run_dir}"

    echo "Analyzing ${run_dir}"
    python -m experiments.exp_a_target_matching.analyze \
      --run-dir "${run_dir}"

    echo "Finished ${name} at: $(date)"
  } > "${log_file}" 2>&1
}

for cfg in "${CONFIGS[@]}"; do
  echo "Launching ${cfg}"

  run_pipeline_one "${cfg}" &

  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 5
  done
done

wait

echo "All pipelines finished."