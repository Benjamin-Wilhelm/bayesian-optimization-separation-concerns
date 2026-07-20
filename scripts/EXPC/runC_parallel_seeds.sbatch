#!/usr/bin/env bash
#SBATCH --job-name=expc_seeds
#SBATCH --partition=SallesInfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logsC_cpu_runs/slurm_%j.out
#SBATCH --error=logsC_cpu_runs/slurm_%j.out
#
# Detached seed-parallel launcher for Experiment C.
#
# Runs run_parallel_seeds.py on a COMPUTE node under SLURM, so it no longer
# dies when the login session closes and gets a real CPU/RAM allocation
# instead of competing on the shared login node `acromion`.  (The previous
# direct `bash` launch on the login node was killed ~54 min in.)
#
# Usage (from the repo root):
#   sbatch scripts/EXPC/runC_parallel_seeds.sbatch                 # fresh run
#   sbatch scripts/EXPC/runC_parallel_seeds.sbatch --reuse-existing  # resume
#
# Any extra args (e.g. --reuse-existing) are forwarded to the Python driver.
# Follow progress:  tail -f logsC_cpu_runs/slurm_<jobid>.out
set -u

# One BLAS/OMP thread per worker; parallelism is across seeds, not within one.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Resolve the repo root. Under SLURM the batch script is copied to a spool
# dir, so BASH_SOURCE is useless here; SLURM_SUBMIT_DIR is the directory sbatch
# was launched from (the repo root). Fall back to BASH_SOURCE for a plain
# `bash` run outside SLURM.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || { echo "cannot cd to repo root '$ROOT'" >&2; exit 1; }
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p logsC_cpu_runs

PYTHON="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
fi

# Match the worker pool to the cores SLURM actually granted this job.
WORKERS="${SLURM_CPUS_PER_TASK:-26}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-<none>}  workers=${WORKERS}"
echo "cmd: ${PYTHON} -m scripts.EXPC.run_parallel_seeds --max-workers ${WORKERS} $*"

srun "$PYTHON" -m scripts.EXPC.run_parallel_seeds \
  --max-workers "$WORKERS" "$@"
