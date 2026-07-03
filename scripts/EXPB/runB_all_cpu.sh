#!/usr/bin/env bash
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_DIR="experiments/exp_b_prior_guided/configs"
LOG_DIR="logsB_cpu_runs"
MAX_PARALLEL=8

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
fi

mkdir -p "$LOG_DIR"

#Uncomment the config of which you wanna create data
CONFIGS=(
  
  "branin_2d_bad_prior.yaml"
  "branin_2d_good_prior.yaml"
  "double_well_1d_good_prior.yaml"
  "double_well_1d_bad_prior.yaml"
  "hartmann_6d_bad_prior.yaml"
  "hartmann_6d_good_prior.yaml"
  "forrester_1d_bad_prior.yaml"
  "forrester_1d_good_prior.yaml"
  #"sanity_double_well_1d.yaml"
)

run_one() {
  local cfg="$1"
  local name="${cfg%.yaml}"
  local log_file="${LOG_DIR}/${name}.out"

  echo "Launching ${name}"

  "$PYTHON" -m experiments.exp_b_prior_guided.run \
    --config "${CONFIG_DIR}/${cfg}" \
    --tag "${name}" \
    > "${log_file}" 2>&1 &
}

running_pids=()

for cfg in "${CONFIGS[@]}"; do
  run_one "$cfg"
  running_pids+=("$!")

  while [[ "${#running_pids[@]}" -ge "$MAX_PARALLEL" ]]; do
    new_pids=()

    for pid in "${running_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      else
        wait "$pid"
      fi
    done

    running_pids=("${new_pids[@]}")
    sleep 5
  done
done

failed=0

for pid in "${running_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "Some runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All raw runs finished."


