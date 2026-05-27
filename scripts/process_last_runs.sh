for run_dir in $(ls -td results/exp_a_target_matching/runs/* | head -n 4); do
  echo "Processing $run_dir"
  python -m experiments.exp_a_target_matching.process --run-dir "$run_dir"

  echo "Analyzing $run_dir"
  python -m experiments.exp_a_target_matching.analyze --run-dir "$run_dir"
done