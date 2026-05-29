for pattern in \
  "*linear_1d_short" \
  "*exponential_1d_short" \
  "*gaussian_trap_1d_short" \
  "*damped_oscillatory_trap_1d_short"
do
  run_dir=$(ls -td results/exp_a_target_matching/runs/${pattern} 2>/dev/null | head -n 1)

  if [[ -z "$run_dir" ]]; then
    echo "WARNING: no run found for pattern ${pattern}"
    continue
  fi

  echo "Processing $run_dir"
  .venv/Scripts/python.exe -m experiments.exp_a_target_matching.process --run-dir "$run_dir"

  echo "Analyzing $run_dir"
  .venv/Scripts/python.exe -m experiments.exp_a_target_matching.analyze --run-dir "$run_dir"
done