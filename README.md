# Bayesian Optimization — Separation of Concerns

This repository investigates **Separation of Concerns (SoC)** in Gaussian-process
Bayesian optimization: learn the underlying objective from observations, then
express targets, domain preferences, or multi-objective trade-offs in the
acquisition. Moving those concerns into the GP can change the modeled function
and its smoothness or prior assumptions.

Three experiments compare SoC-aligned **RIGHT** methods with **WRONG** alternatives:
**A — Target Matching**, **B — Prior-Guided BO**, and **C — Multi-Objective BO**.
These are experimental labels; performance and calibration are evaluated across seeds.

## Setup

Run commands from the directory containing this README,
`bayesian-optimization-separation-concerns/`. The inspected Linux environment uses
Python **3.9.25**, PyTorch **2.8.0**, BoTorch **0.7.0**, and GPyTorch **1.13**.
The code requires the legacy `FixedNoiseGP` API and relies on BoTorch 0.7's default kernel.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r Mrequirements.txt 'torch==2.8.0' 'botorch==0.7.0' 'gpytorch==1.13'
```

**`Mrequirements.txt`** lists NumPy, SciPy, pandas, PyTorch, GPyTorch, BoTorch,
PyYAML, Matplotlib, scikit-learn, tqdm and pytest without pins; the versions above
are not a full lock. `scienceplots`/seaborn are optional; figures do not require LaTeX.

Examples use CPU. A/B expose `--device cpu|cuda|auto`; C accepts the option but
builds CPU tensors without applying it. Shell launchers use Bash/Linux; SLURM is optional.

## Repository map

```text
common/                         GP/design/acquisition utilities, priors, metrics, diagnostics, statistics, logging
benchmarks/                     Scalar problems, maximization wrapper, two-objective ZDT benchmarks
acquisitions/                   Target matching, prior-guided UCB, scalarized UCB
data/                           Raw and processed JSON schemas for A, B, C
experiments/
  exp_a_target_matching/        run.py → process.py → analyze.py → plot.py
  exp_b_prior_guided/           run.py → process.py → analyze.py → plot.py
  exp_c_multiobjective/         run.py → process.py → analyze.py (includes figures)
    configs/                   fixed_scalarization/{LOG_BETA,beta=8}/, pareto_front/ (also beta=8/)
scripts/EXPA/, EXPB/, EXPC/      CPU batch/seed launchers and postprocessing wrappers
scripts/smoke_test_exp_*_workflow.py  End-to-end checks that create experimental data
tests/                          Scientific and implementation checks
results/<experiment>/runs/     Saved executions, derived data, figures, logs
```

Each experiment has YAML `configs/`; `hpc/` is a scaffold. SLURM lives under `scripts/EXPC/`.

## Shared modeling and reproducibility conventions

- Observations are noisy; synthetic ground truth is reserved for evaluation.
  A/B store `y_obs` and `f_noiseless`; C stores noisy `model_y` separately from
  noiseless `raw_y` and `oriented_y`.
- All experiments use `torch.double`, scrambled Sobol initial designs, and sequential
  BO (`q=1`). Hyperparameters are fitted afresh at each step by exact marginal
  log-likelihood. The default kernel is scaled Matérn 5/2 with ARD in BoTorch 0.7;
  `model.kernel` also supports `matern32`, `matern12`, and `rbf` (`common/models.py`).
- **Input normalization** acts inside the GP. A/C use `Normalize(d=...)` with
  bounds learned from the training inputs; B supplies fixed problem bounds.
  Designs and acquisition searches still use the native domain bounds.
- **Output standardization** uses `Standardize` on current training targets;
  posterior outputs return to the target scale. It is separate from absolute-value
  transforms, orientation flips, and scalarization. Policies differ below.
- GP `noise_std=s` becomes a constant **variance** tensor `train_Yvar = s²` in
  `FixedNoiseGP`; BoTorch also standardizes that variance when standardizing outputs.
  `noise_std=None` selects inferred-noise `SingleTaskGP`. GPyTorch may clamp tiny
  fixed variances for numerical stability.
- Acquisition optimization uses BoTorch `optimize_acqf`; `num_restarts` and
  `raw_samples` control multistart initialization. `BetaSchedule` supports constant,
  logarithmic $\beta_t=\beta_0+c\log(t+1)$, and `matern_gamma` schedules.
  A/B put these settings at the YAML root; C nests them under `acquisition`.
- All `run.py` entry points accept `--config`, `--tag`, `--results-root`, and
  `--device`. Seeds are selected in YAML, not by a run-time `--seed` option.
  A/B use seed lists; C uses inclusive `{start, end}` ranges.

## Experiment A — Target Matching

### Principle

Minimize $|f(x)-T|$ using observations $y=f(x)+\epsilon$. RIGHT fits a GP on
`y_obs`; WRONG fits a GP on `abs(y_obs - target)`. BoTorch maximizes both scores:

$$
a_R(x)=-|\mu_f(x)-T|+\sqrt{\beta_t}\,\sigma_f(x),\qquad
a_W(x)=-\mu_g(x)+\sqrt{\beta_t}\,\sigma_g(x).
$$

The target enters RIGHT only at acquisition time. WRONG absorbs it into the
training labels, potentially introducing kinks where the objective crosses $T$.

### Code organization and workflow

| Program | Inputs and main dependencies | Outputs |
| --- | --- | --- |
| `run.py` | YAML; `benchmarks`, shared GP builder, `acquisitions/target_matching.py` | Paired raw seed JSON: initial design, raw observations, BO trajectories, full fitted GP snapshots |
| `process.py` | Run `config.json` and raw JSON; saved states, shared metrics/diagnostics | Per-seed metrics, convergence curves, held-out calibration and optional predictions; reconstructs GPs without MLL fitting |
| `analyze.py` | Processed seeds and saved config; `common/statistics.py` | Across-seed performance/calibration summaries, bootstrap curves, statistical tests |
| `plot.py` | Saved config, processed/aggregate data, raw trajectories/snapshots and benchmark geometry | PNG/PDF figures under `figures/<benchmark>/` |

### Experimental protocol

Configured problems include linear, exponential, sine-trend, Forrester,
Gaussian-trap variants and damped oscillatory trap in 1D, Branin in 2D, and
Hartmann in 6D. Configurations may override the native bounds.
Representative **base configs** (outside kernel/schedule subdirectories):

| Config | Initial points + BO evaluations | Seeds | Target; exploration | Restarts / raw samples |
| --- | --- | --- | --- | --- |
| `branin_2d_short.yaml` | 4 + 50 | 1–100 | $T=5$; $\beta=2$ | 25 / 2048 |
| `hartmann_6d_full.yaml` | 12 + 50 | 1–100 | $T=-2.5$; $\beta=8$ | 30 / 4096 |
| `gaussian_trap_1d_short.yaml` | 1 + 50 | 1–100 | $T=0.38$; $\beta=2$ | 20 / 1024 |

Here `budget` counts **additional BO evaluations**; total cost is
`initial_design_size + budget`. Methods share initial inputs and observations.
The base scientific configs above are noiseless; `noise_std` controls Gaussian
observation noise. For nested benchmark configs, it belongs inside `benchmark`.
Both methods currently use fixed noise equal to the benchmark noise, as noise is currently set to 0; To further study, the effect of noise, one could try to infer noise variance for the wrong model, as the transformation of the objective folds the noise, leading to another source of misspecification. Reusing a fixed variance after the
absolute-value transform is a modeling approximation for noisy WRONG runs (except when it is set to 0 obviously.

Input normalization and output standardization are enabled for both methods by
the builder defaults. `model.kernel` selects the kernel; the runner does not
forward YAML normalization/standardization flags. `RBF/`, `MATERN_1_2/`,
`ADAPTATIVE_LOG_BETA/`, and `ADAPTATIVE_MATERN_BETA/` provide alternative configs
whose budgets and schedules can differ from the base configs.

The main metric is target simple regret $\min_{s\le n}|f(x_s)-T|-d^{*}$, with
$d^{*}=\min_x|f(x)-T|$ from the benchmark oracle, or zero if unavailable. Secondary
metrics include cumulative regret, AUC, hitting time and failure rate. Deterministic
global/near-target grids support NLPD, coverage, PIT, residuals, bound violations
and optional joint diagnostics. RIGHT's modeled quantity is $f$; WRONG's is
$|f-T|$; additional decision diagnostics compare target distances. Calibration
replays saved **pre-query** models, including the last snapshot, without a final refit.

### Commands and reproducibility

```bash
python -m experiments.exp_a_target_matching.run \
  --config experiments/exp_a_target_matching/configs/branin_2d_short.yaml --device cpu
# Replace <RUN_DIR> with the newly created results/exp_a_target_matching/runs/... path.
RUN_DIR='<RUN_DIR>'
python -m experiments.exp_a_target_matching.process --run-dir "$RUN_DIR"
python -m experiments.exp_a_target_matching.analyze --run-dir "$RUN_DIR"
python -m experiments.exp_a_target_matching.plot --run-dir "$RUN_DIR"

# Alternative raw generation: seed-parallel CPU execution of the driver's CONFIGS.
python -m scripts.EXPA.run_parallel_seeds --max-workers 8
```

Processing supports repeatable `--seed 1 --seed 2` and `--benchmark branin_2d`
filters, plus `--device`. Analysis/plotting accept `--benchmark`; analysis accepts
`--statistic`, `--n-bootstrap`, `--confidence`, `--convergence-band-type`,
`--calibration-band-type`, and `--residual-band-type` (`pointwise`, `uniform`, `both`).
Paired Wilcoxon tests are primary; Mann–Whitney and effect sizes are secondary.
Parallel/batch variants generate raw data only (see below); A has no dedicated SLURM script.

## Experiment B — Prior-Guided BO

### Principle

Keep the objective on its natural output scale, oriented to maximization by a
sign flip where needed. A Gaussian spatial preference
$\pi(x)=\exp[-\tfrac12\sum_j((x_j-c_j)/\ell_j)^2]$ is evaluated in native input
coordinates. All methods train on the same kind of raw noisy objective values;
none trains on residuals.

| Method | Location of the preference |
| --- | --- |
| `vanilla_ucb` | Baseline: $\mu+\sqrt{\beta_t}\sigma$ |
| `alpha_pi_ucb` (RIGHT) | Acquisition: $\mu+[1+\frac{\alpha}{1-\alpha}\pi(x)]\sqrt{\beta_t}\sigma$, $0\le\alpha<1$ |
| `lambda_pi_fixed_ucb`, `lambda_pi_adaptive_ucb` (WRONG) | GP mean $m(x)=\lambda f_{\rm prior}(x)$; ordinary UCB; fixed or MLL-learned $\lambda$ |
| `lambda_pi_fixed_standardized_ucb`, `lambda_pi_adaptive_standardized_ucb` | Same prior-in-mean comparison with standardized observations and z-scored $\pi$ |

The raw prior mean is $f_{\rm prior}=f_{\rm lower}+(f_{\rm upper}-f_{\rm lower})\pi$,
using the benchmark's oriented output range, distinct from RIGHT's dimensionless weight.

### Code organization and workflow

| Program | Inputs and main dependencies | Outputs |
| --- | --- | --- |
| `run.py` | YAML; `benchmarks/oriented.py`, `common/priors.py`, `acquisitions/prior_guided.py` | Paired raw seed JSON with shared initial data, GP snapshots, preference values and learned $\lambda$ |
| `process.py` | Saved config/raw JSON; prior-aware GP reconstruction, shared metrics/diagnostics | Per-seed regret, convergence, global/prior-region calibration; no GP refitting |
| `analyze.py` | Processed seeds and config; shared statistics | Aggregate summaries/curves and comparisons against the configured baseline (`vanilla_ucb` by default) |
| `plot.py` | Saved config, processed/aggregate data, raw trajectories/snapshots, benchmark and prior | PNG/PDF performance, calibration and prior diagnostics in `figures/<benchmark>/` |

### Experimental protocol

Good/bad priors cover Forrester and double Gaussian well (1D), Branin (2D), and
Hartmann (3D/6D). Base scientific configs use seeds 1–50 and `noise_std: 0.0`.
Initial/BO counts are 1/50 for 1D, 4/50 for Branin, 6/60 for Hartmann-3, and 12/80
for Hartmann-6. `budget` excludes the initial design, shared within each seed.

Base configs use $\beta=8$, usually $\alpha=0.5$ (double well: 0.25),
`lambda_init: 1.0`, and typically 20 restarts / 1024 raw samples.
`ADAPTATIVE_LOG_BETA/` and `ADAPTATIVE_BETA_MATERN/` change the schedule;
`TEST_ALPHA/`, `TEST_PRIOR_CENTER/`, and `TEST_PRIOR_LENGTHSCALE/` vary preferences.
The YAML `methods` list determines which of the six supported methods run.

The default kernel is Matérn 5/2. `model.use_fixed_noise: true` fixes the top-level
`noise_std` on the oriented objective scale; setting it to `false` infers noise.
`model.normalize_input` defaults to true and uses fixed domain bounds.
Standard GP methods honor `model.standardize_output` (default true). Raw
`lambda_pi_*_ucb` methods force output standardization **off**; their
`*_standardized_ucb` counterparts force it **on** and z-score $\pi$ over current
training inputs, with a deterministic Sobol fallback when its spread degenerates.
Orientation never rescales the objective to $[0,1]$.

Evaluation uses noiseless oriented simple regret $f^*-\max_{s\le n}f(x_s)$,
cumulative instantaneous regret, AUC, and hitting time. Calibration is on the
objective for every method, globally and on the masked test-grid region
$\{x:\pi(x)\ge\texttt{prior_region_threshold}\}$. NLPD, coverage, PIT, residuals,
bound violations and optional joint diagnostics replay pre-query snapshots as in A.

### Commands and reproducibility

```bash
# Small local workflow: Forrester, seeds 0 and 1, 3 initial + 4 BO evaluations.
python -m experiments.exp_b_prior_guided.run \
  --config experiments/exp_b_prior_guided/configs/smoke_forrester.yaml --device cpu
RUN_DIR='<RUN_DIR>'  # results/exp_b_prior_guided/runs/<generated_run_id>
python -m experiments.exp_b_prior_guided.process --run-dir "$RUN_DIR" --verify
python -m experiments.exp_b_prior_guided.analyze --run-dir "$RUN_DIR"
python -m experiments.exp_b_prior_guided.plot --run-dir "$RUN_DIR"

# Alternative raw generation across seeds/configs selected in the driver.
python -m scripts.EXPB.run_parallel_seeds --max-workers 8
```

For scientific runs, choose a good/bad-prior YAML. Processing supports A's filters;
`--verify` checks stored values/regret against analytic re-evaluation at saved inputs.
Analysis shares A's bootstrap/band options and adds `--baseline METHOD`: comparisons
use paired Wilcoxon plus secondary unpaired tests/effect sizes. B has no dedicated SLURM script.

## Experiment C — Multi-Objective BO

### Principle

ZDT objectives are minimized in their original convention; optimization uses
$F=-f_{\rm raw}$ for maximization. RIGHT fits one independent GP per objective
(`ModelListGP`) and scalarizes the objective-wise UCB vector. WRONG fits a scalar
GP on $g_\lambda=\rho_\lambda(\texttt{model_y})$ and maximizes ordinary scalar UCB:

$$
a_R(x;\lambda)=\rho_\lambda\!\left((\mu_j(x)+\sqrt{\beta_t}\sigma_j(x))_{j=1}^m\right),
\qquad a_W(x)=\mu_g(x)+\sqrt{\beta_t}\sigma_g(x).
$$

Scalarizations are weighted sum $\rho_\lambda(y)=\sum_j\lambda_j y_j$ and radial
hypervolume $\rho_\lambda(y)=[\min_j \max(0,y_j-z_j)/\lambda_j]^m$.
**Fixed scalarization** compares `right`/`wrong` at fixed preferences.
**Pareto front** compares `right_algorithm_1` (one shared posterior across online
preferences) with `right_algorithm_2`/`wrong_algorithm_2` (independent subruns,
each with a fixed preference). Algorithm 2's RIGHT/WRONG subruns share preferences
and initial observations; observations are not pooled between subruns.

### Code organization and workflow

| Program/module | Inputs and main dependencies | Outputs |
| --- | --- | --- |
| `run.py` | YAML validated by `config.py`; ZDT benchmarks; `methods.py` GP/acquisition logic; shared scalarization/preference sampling | Raw paired seed JSON containing fixed cases or global Pareto trajectories, preferences, budgets and pairing hashes |
| `process.py` | Raw JSON and metadata; analytic benchmarks, hypervolume/regret/Pareto diagnostics | Verified per-seed metrics; re-evaluates saved inputs, never rebuilds GPs or reruns BO |
| `analyze.py` | Processed seeds for aggregation; raw data for objective/decision-space figures; `plotting.py`, `trajectory_figures.py`, `final_comparisons.py` | Aggregate JSON, paired bootstrap comparisons, PNG/PDF figures directly under `figures/` |

There is no separate C `plot.py`: analysis generates the figures. Older processed
Pareto files need reprocessing to supply the current dominance/IGD diagnostics.

### Experimental protocol

ZDT1 (convex front) and ZDT2 (concave front with unsupported interior points) have
two objectives, with configured input dimensions 2 or 6 on $[0,1]^d$.
`model_y = -raw_y + noise`; `benchmark.normalization: fixed` is informational:
there is no additional fixed output rescaling. GP input normalization and output
standardization default to true and can be disabled independently under `model`.
The kernel defaults to Matérn 5/2. Standard configs use seeds 1–50,
20 restarts / 1024 raw samples, and either $\beta_t=8+\log(t+1)$ or constant 8.

| Mode | Budget/design and preferences |
| --- | --- |
| Fixed | Total `budget` 50 in 2D (5 initial), 200 in 6D (12 initial), **per method and scalarization/angle case**; linear and hypervolume; angles 10°, 30°, 45°, 60°, 80° with $\lambda=(\cos\theta,\sin\theta)$ |
| Pareto | Total `budget` 500 **per method including every initial point**; Algorithm 1 starts with 5 points; Algorithm 2 splits the budget evenly over 10 subruns with 2 initial points each; stored global order is round-robin, including initialization |

Pareto preferences are uniform on the positive unit sphere. Algorithm 1 draws
one per BO step; Algorithm 2 draws one per subrun. A separate RNG stream supplies
2048 evaluation preferences per seed, shared across methods and independent of
optimization preferences. Default subrun designs differ across subruns; RIGHT
and WRONG Algorithm 2 reuse the same subrun designs/initial observations.

**Noise policy:** `noise.std` is the per-objective Gaussian **standard deviation**
on the oriented scale (currently `1.0e-6` in the supplied configs).
`model.use_fixed_noise` defaults to **true**, as implemented in
`methods.py::make_gp_builder`. It passes `max(noise.std, 1e-6)` to
`GPModelConfig.noise_std`, so `common/models.py` constructs `FixedNoiseGP` with
`train_Yvar = max(noise.std, 1e-6)²`. This is the chosen policy for known synthetic
noise; explicit `false` selects noise inference with `SingleTaskGP`.
GPyTorch may clamp this tiny variance further. WRONG's scalar GP reuses the same
variance after scalarization rather than propagating noise through $\rho_\lambda$;
for nonlinear scalarization this is an approximation to transformed noise.

Fixed-mode metrics are noiseless scalarized simple/instantaneous/cumulative
regret. Processing recomputes the scalar optimum on a 1,000,000-point analytic
front. Pareto metrics include exact 2D hypervolume, hypervolume regret, empirical
Bayes regret, dominance diagnostics and IGD; front-based reference values use
the configured resolution (normally 10,000). The fixed oriented hypervolume
reference is $z=(-1.1,-1.1)$ by default. C's trajectory bands show median and
25–75% **inter-seed percentiles**, with dashed means; final comparisons separately
use 95% paired seed-block bootstrap CIs (5,000 replicates). Fixed-mode summaries
also report paired Wilcoxon tests. Linear scalarization's inability to recover
ZDT2's interior front is a scalarization limitation shared by both architectures.

### Commands and reproducibility

```bash
# Small fixed-preference workflow (6 total evaluations per method/case, 2 seeds).
python -m experiments.exp_c_multiobjective.run \
  --config experiments/exp_c_multiobjective/configs/fixed_scalarization/LOG_BETA/smoke_zdt1_d2.yaml \
  --tag smoke_fixed_zdt1_d2 --device cpu
RUN_DIR='<RUN_DIR>'  # results/exp_c_multiobjective/runs/<generated_run_id>
python -m experiments.exp_c_multiobjective.process --run-dir "$RUN_DIR"
python -m experiments.exp_c_multiobjective.analyze --run-dir "$RUN_DIR"  # includes figures

# Alternative raw generation: small Pareto workflow; then process/analyze its own RUN_DIR.
python -m experiments.exp_c_multiobjective.run \
  --config experiments/exp_c_multiobjective/configs/pareto_front/smoke_zdt1_d2.yaml \
  --tag smoke_pareto_zdt1_d2 --device cpu

# Full selected configuration suite, parallel across seeds (raw generation only).
python -m scripts.EXPC.run_parallel_seeds --max-workers 8 --schedule logbeta

# Same seed-parallel driver under SLURM; uses allocated CPUs as worker count.
mkdir -p logsC_cpu_runs
sbatch scripts/EXPC/runC_parallel_seeds.sbatch --schedule logbeta
```

`--schedule` accepts `all` (default), `logbeta`, or `beta8`. SLURM defaults to
partition `SallesInfo`, 26 CPUs, 48 GB, and two days; adapt with `sbatch` options.
It generates raw data only. C processing accepts only `--run-dir`, without seed/worker filters.
Analysis accepts repeatable `--run-dir` for Pareto subrun-count ablation overlays;
outputs go to the first run directory, whose processed seeds supply its summaries.

## Execution variants and outputs

Seed-parallel drivers reuse the sequential per-seed routines, raw schema and pairing.
Protocol/budget are equivalent; bitwise identity across hardware, dependency versions
or BLAS thread settings is not guaranteed. Drivers use CPU with math-library threads
defaulting to one per worker.

Config selection is embedded in `CONFIGS`/`CONFIG_DIR`, with **no `--config` option**:
A selects Hartmann-6 from `ADAPTATIVE_MATERN_BETA/`; B selects eight configs from
`ADAPTATIVE_LOG_BETA/`; C selects 16 fixed/Pareto configs before schedule filtering.
Each creates a fresh run per tag.
All accept `--results-root` and `--reuse-existing`; the latter selects the newest
matching run. C skips existing seed files; A/B currently rerun and overwrite
them, despite resume wording in their docstrings.

`runA_all_cpu.sh`/`runB_all_cpu.sh` launch their embedded config lists concurrently,
with sequential seeds inside each process. `runA_parallel_seeds.sh`,
`runB_parallel_seeds.sh`, and `runC_parallel_seeds.sh` wrap the Python seed drivers
(optional positional worker count). These are generation variants, not later
pipeline stages. `processA_last_runs.sh`/`processB_last_runs.sh` perform processing
and aggregation; `plotA_last_runs.sh`/`plotB_last_runs.sh` plot. For C,
`processC_last_runs.sh` processes and `plotC_last_runs.sh` analyzes/plots.
Explicit run-directory commands above avoid these wrappers' tag-based selection.

```text
results/<experiment>/runs/<timestamp>_<tag>/
  config.json, metadata.json                Parsed config; command, Git state, environment
  raw/<benchmark>/seed_NNN.json             All methods paired within one seed file
  processed/per_seed/<benchmark>/
    seed_NNN_processed.json                Derived per-seed metrics (and A/B calibration)
  processed/aggregate/<benchmark>/          A/B cross-seed summaries and bootstrap curves
  figures/<benchmark>/                     A/B PNG/PDF figures
  figures/                                 C PNG/PDF figures
  logs/                                    Stage logs; parallel launcher logs may be external
```

A/B aggregate files are `final_performance.json`, `convergence_curves.json`,
`calibration_summary.json`, `calibration_curves.json`, and `statistical_tests.json`
(curve files depend on `aggregate` settings). C instead writes
`processed/aggregate_summary.json` and `processed/paired_bootstrap_comparisons.json`.
Keep raw data, saved config, code and dependency versions for reproducibility.
A/B snapshots require a compatible GP stack. Aggregation uses processed seeds;
plotting can also read raw data. Postprocessing does not rerun BO.

## Configuration caveats

- A's `smoke_1d.yaml` currently runs **Branin-2D**, 20 seeds and 50 BO steps with
  20 initial points; it is not a tiny 1D smoke test despite its name/comments.
- In C's `pareto_front/` and `pareto_front/beta=8/`, ZDT1 d2/d6 tags are swapped;
  ZDT2's `zdt2_d2.yaml` contains **6D**, and `zdt2_d6.yaml` contains **2D**.
  Inspect `benchmark.input_dim`; use explicit `--tag`. The Python parallel driver
  already assigns tags matching the contents.
- `scripts/EXPC/runC_all_cpu.sh` references missing
  `fixed_scalarization/zdt*.yaml` paths. Use the documented Python driver or the
  actual `LOG_BETA/` and `beta=8/` configs. C's latest-run wrappers do not match
  the Python driver's `_logbeta`/`_beta8` tag suffixes; use explicit `--run-dir`.
