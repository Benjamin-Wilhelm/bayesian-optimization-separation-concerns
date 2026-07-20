#!/usr/bin/env bash
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_ROOT="experiments/exp_c_multiobjective/configs"
LOG_DIR="logsC_cpu_runs"
MAX_PARALLEL=8

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
fi

mkdir -p "$LOG_DIR"

# All non-smoke configs.  Entries are "relative_config_path|tag".
# The explicit tag (mode + benchmark + dim) gives an unambiguous run-dir name
# and sidesteps the config `tag:` field (identical filenames live in both
# sub-dirs, and some pareto configs carry a swapped tag).
CONFIGS=(
  "fixed_scalarization/zdt1_d2.yaml|fixed_zdt1_d2"
  "fixed_scalarization/zdt1_d6.yaml|fixed_zdt1_d6"
  "fixed_scalarization/zdt2_d2.yaml|fixed_zdt2_d2"
  "fixed_scalarization/zdt2_d6.yaml|fixed_zdt2_d6"
  "pareto_front/zdt1_d2.yaml|pareto_zdt1_d2"
  "pareto_front/zdt1_d6.yaml|pareto_zdt1_d6"
  "pareto_front/zdt2_d2.yaml|pareto_zdt2_d2"
  "pareto_front/zdt2_d6.yaml|pareto_zdt2_d6"
)

run_one() {
  local entry="$1"
  local cfg="${entry%%|*}"
  local tag="${entry##*|}"
  local log_file="${LOG_DIR}/${tag}.out"

  echo "Launching ${tag}  (${cfg})"

  "$PYTHON" -m experiments.exp_c_multiobjective.run \
    --config "${CONFIG_ROOT}/${cfg}" \
    --tag "${tag}" \
    > "${log_file}" 2>&1 &
}

running_pids=()

for entry in "${CONFIGS[@]}"; do
  run_one "$entry"
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
