# Bayesian Optimization — Separation of Concerns

## 1. Overview and scientific motivation

This repository studies **Separation of Concerns (SoC)** in Gaussian-process
Bayesian optimization (BO): learn the unknown phenomenon from observations,
and express targets, domain preferences, and multi-objective trade-offs in the
acquisition function.

Putting a decision preference into the training labels or GP mean can change
the modeled function, its smoothness, or its prior assumptions. The experiments
investigate the consequences for optimization and probabilistic calibration.
**RIGHT** and **WRONG** name the experimental conditions; they do not assert
that one method always wins or that a preference-informed model is inherently
invalid.

Three experiments share GP utilities, seed-level logging, and postprocessing:

| Experiment | Question | Implementation |
| --- | --- | --- |
| A — Target Matching | Where should a target value enter BO? | [exp_a_target_matching](experiments/exp_a_target_matching/) |
| B — Prior-Guided BO | Should a spatial preference shape the acquisition or the GP mean? | [exp_b_prior_guided](experiments/exp_b_prior_guided/) |
| C — Multi-Objective BO | Where should objective preferences enter, and when does sharing observations help? | [exp_c_multiobjective](experiments/exp_c_multiobjective/) |

The current runners use analytic synthetic benchmarks. They distinguish noisy
observations used by BO from noiseless values used only for evaluation.

## 2. Three experiments

### A — Target matching

Given a target $T$, minimize $g(x)=\lvert f(x)-T\rvert$ using noisy observations
$y_{\mathrm{obs}}=f(x)+\epsilon$.

**RIGHT** fits a GP to raw observations. **WRONG** fits a GP to their absolute
target distances. Both acquisitions are maximized:

$$
\begin{aligned}
a_R(x) &= -\lvert\mu_f(x)-T\rvert
          +\sqrt{\beta_t}\,\sigma_f(x),\\
a_W(x) &= -\mu_g(x)+\sqrt{\beta_t}\,\sigma_g(x).
\end{aligned}
$$

The implemented method names are `right_target_matching_ucb` and
`wrong_gp_on_abs_lcb`. A target crossing can introduce a kink into the
transformed function even when the underlying objective is smooth.

### B — Prior-guided BO

Objectives are oriented to **maximization**, with a sign flip where required
and no rescaling of their natural output range. A Gaussian spatial preference
is evaluated in native input coordinates:

$$
\pi(x)=\exp\!\left[
-\frac{1}{2}\sum_{j=1}^{d}
\left(\frac{x_j-c_j}{\ell_j}\right)^2
\right].
$$

**RIGHT**, `alpha_pi_ucb`, keeps a generic GP and weights its exploration bonus:

$$
a_R(x)=\mu_f(x)+
\left(1+\frac{\alpha}{1-\alpha}\pi(x)\right)
\sqrt{\beta_t}\,\sigma_f(x),
\qquad 0\leq\alpha<1.
$$

The vanilla baseline uses ordinary UCB. The raw prior-in-mean **WRONG**
conditions use ordinary UCB with a GP whose mean is

$$
m(x)=\lambda f_{\mathrm{prior}}(x),
\qquad
f_{\mathrm{prior}}(x)=f_{\mathrm{lower}}
 +(f_{\mathrm{upper}}-f_{\mathrm{lower}})\pi(x).
$$

Here the output bounds belong to the oriented benchmark. The coefficient
$\lambda$ is fixed or learned by marginal likelihood. The posterior models
the objective directly; training labels are not residuals.

| Method | Condition |
| --- | --- |
| `vanilla_ucb` | Generic GP, ordinary UCB |
| `alpha_pi_ucb` | Generic GP, preference in the acquisition |
| `lambda_pi_fixed_ucb` | Raw-scale prior mean, fixed coefficient |
| `lambda_pi_adaptive_ucb` | Raw-scale prior mean, learned coefficient |
| `lambda_pi_fixed_standardized_ucb` | Standardized prior-mean variant, fixed coefficient |
| `lambda_pi_adaptive_standardized_ucb` | Standardized prior-mean variant, learned coefficient |

The standardized variants z-score $\pi$ in model coordinates instead of using
the raw-scale mean above. They standardize observations; raw prior-mean
variants disable output standardization.

### C — Multi-objective BO

The original ZDT objectives are minimized. The implementation uses their
sign-flipped maximization view $F=-f_{\mathrm{raw}}$.

**RIGHT** fits an independent GP to each noisy oriented objective.
**WRONG** fits a scalar GP to scalarized noisy observations, with a preference
$\lambda$ fixed within that GP. Their maximized acquisitions are

$$
\begin{aligned}
A_{j,t}(x)&=\mu_{j,t}(x)+\sqrt{\beta_t}\,\sigma_{j,t}(x),\\
a_R(x;\lambda)&=\rho_\lambda\\left(
 A_{1,t}(x),\ldots,A_{m,t}(x)\right),\\
a_W(x;\lambda)&=\mu_{g_\lambda,t}(x)
 +\sqrt{\beta_t}\,\sigma_{g_\lambda,t}(x).
\end{aligned}
$$

Two monotone scalarizations are implemented:

$$
\rho_\lambda^{\mathrm{lin}}(y)=\sum_{j=1}^{m}\lambda_j y_j,
\qquad
\rho_\lambda^{\mathrm{HV}}(y)=
\left[
\min_j\frac{\max(0,y_j-z_j)}{\lambda_j}
\right]^m.
$$

The hypervolume reference point is fixed at $z=(-1.1,-1.1)$ in maximization
coordinates. Weights are clamped away from zero only for numerical safety.

**Fixed preference** isolates the scalarization location. **Pareto front**
compares three methods at equal global evaluation budget:

| Method | Observation sharing |
| --- | --- |
| `right_algorithm_1` | One shared set of objective GPs; a new preference at each BO step |
| `right_algorithm_2` | Independent RIGHT subruns, each with a fixed preference |
| `wrong_algorithm_2` | Independent WRONG subruns using the same preferences and initial data as RIGHT Algorithm 2 |

WRONG is coherent for a fixed preference. Combining labels from changing
preferences in a scalar GP without a preference input would model a different
problem; that method is not implemented.

## 3. Installation and Quick Start

Run commands from the repository root. The inspected environment uses:

| Component | Version |
| --- | --- |
| Python | 3.9.25 |
| PyTorch | 2.8.0 |
| BoTorch | 0.7.0 |
| GPyTorch | 1.13 |
| NumPy / SciPy | 2.0.2 / 1.13.1 |
| Matplotlib / PyYAML | 3.9.4 / 6.0.3 |

The code uses the legacy `FixedNoiseGP` API. Updating BoTorch requires
compatibility work; installing the latest version is not equivalent.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r Mrequirements.txt
```

[Mrequirements.txt](Mrequirements.txt) pins the numerical, GP, plotting,
configuration, and testing dependencies to the validated Python 3.9 environment,
including the directly imported `linear-operator` package. This is **not a
complete lock** of transitive dependencies or platform-specific wheels.
`scienceplots` and seaborn are optional, with tested versions noted in the
requirements file; plotting does not require LaTeX.

A small Experiment A example uses two seeds, three initial points, and three
additional BO evaluations per method. It also enables the RKHS diagnostics:

```bash
python -m experiments.exp_a_target_matching.run \
  --config experiments/exp_a_target_matching/configs/ablations/smoke_lengthscale.yaml \
  --device cpu

# Set this to the run directory printed by run.py.
RUN_DIR='results/exp_a_target_matching/runs/<generated_run_id>'
python -m experiments.exp_a_target_matching.process --run-dir "$RUN_DIR"
python -m experiments.exp_a_target_matching.analyze --run-dir "$RUN_DIR"
python -m experiments.exp_a_target_matching.plot --run-dir "$RUN_DIR"
```

The smoke configuration is a technical check, not evidence for a scientific
conclusion. Seeds and budgets are selected in YAML.

## 4. Repository structure

```text
common/                       GP builders, designs, metrics, diagnostics, statistics, logging
benchmarks/                   Scalar synthetic functions, orientation wrapper, ZDT problems
acquisitions/                 Target matching, prior-guided UCB, scalarized UCB
data/                         Raw and processed result schemas
experiments/
  exp_a_target_matching/      run → process → analyze → plot; configs and ablation comparison
  exp_b_prior_guided/         run → process → analyze → plot; prior configurations
  exp_c_multiobjective/       run → process → analyze (also generates figures)
scripts/
  EXPA/, EXPB/, EXPC/         CPU launchers and postprocessing wrappers
  launcher_utils.py          Shared YAML loading for seed-parallel launchers
hpc/                          Scaffold; the current SLURM launcher is under scripts/EXPC/
tests/                        Scientific invariants, diagnostics, and CLI regression tests
results/<experiment>/runs/    Independent config executions, paired seeds, derived outputs
```

Models remain generic. Decision-specific formulas belong in acquisitions;
runners generate raw results, processors compute metrics, and plotting reads
saved results. The experiment is the scientific unit, a config execution is a
run, and a seed contains one trajectory for every compared method.

## 5. Running experiments

### Sequential execution of one YAML

The three entry points accept `--config`, `--tag`, `--results-root`,
and `--device`. These commands generate raw data only:

```bash
python -m experiments.exp_a_target_matching.run \
  --config experiments/exp_a_target_matching/configs/branin_2d_short.yaml --device cpu

python -m experiments.exp_b_prior_guided.run \
  --config experiments/exp_b_prior_guided/configs/branin_2d_good_prior.yaml --device cpu

python -m experiments.exp_c_multiobjective.run \
  --config experiments/exp_c_multiobjective/configs/pareto_front/zdt1_d2.yaml --device cpu
```

Those are scientific configurations and can be expensive. For technical checks,
use A's Quick Start, B's `configs/smoke_forrester.yaml`, or C's
`configs/pareto_front/smoke_zdt1_d2.yaml`. A/B support CPU, CUDA, and automatic
device selection. C currently constructs CPU tensors even when another device
is requested; use CPU for C.

### Parallel execution across seeds

The Python launchers use the same per-seed routines as the sequential runners.
`--config PATH` selects **one** YAML without editing launcher source.
Relative paths are resolved from the current directory; absolute paths and
`.yaml`/`.yml` files are accepted.

```bash
python -m scripts.EXPA.run_parallel_seeds \
  --config experiments/exp_a_target_matching/configs/branin_2d_short.yaml \
  --max-workers 8

python -m scripts.EXPB.run_parallel_seeds \
  --config experiments/exp_b_prior_guided/configs/branin_2d_good_prior.yaml \
  --max-workers 8

python -m scripts.EXPC.run_parallel_seeds \
  --config experiments/exp_c_multiobjective/configs/pareto_front/zdt1_d2.yaml \
  --max-workers 8
```

Workers run on CPU; math-library thread counts default to one unless already
set in the environment. Seed pairing, RNG policy, GP fitting, and raw schemas
are inherited from the experiment runners. Bitwise identity across dependency
versions, hardware, and thread settings is not guaranteed.

| Option | Meaning |
| --- | --- |
| `--config PATH` | Run one YAML instead of the default batch |
| `--max-workers N` | Maximum worker processes; default is CPU count minus two, with a minimum of one |
| `--results-root PATH` | Alternative output root |
| `--reuse-existing` | Select the newest run matching the effective tag; see resume differences below |
| `--schedule all\|logbeta\|beta8` | C only: filter its default batch; mutually exclusive with `--config` |

An explicit YAML uses its `tag`, falling back to its filename stem. C's
default batch instead uses explicit mode/dimension/schedule tags, so an
explicit-config invocation may have a different tag from the corresponding
batch job.

### Batch execution and resume behavior

Omitting `--config` preserves the existing selections:

| Driver | Default batch |
| --- | --- |
| A | Hartmann-6 from `configs/ADAPTATIVE_MATERN_BETA/` |
| B | Eight good/bad-prior configs from `configs/ADAPTATIVE_LOG_BETA/` |
| C | Sixteen fixed/Pareto configs across log-beta and constant-beta schedules |

```bash
python -m scripts.EXPA.run_parallel_seeds --max-workers 8
python -m scripts.EXPB.run_parallel_seeds --max-workers 8
python -m scripts.EXPC.run_parallel_seeds --schedule logbeta --max-workers 8
```

Ordinary launches use timestamped run directories. With `--reuse-existing`,
**C skips completed seed files; A/B recompute all selected seeds and can
overwrite their raw JSON**. All three save config and metadata again. Resume
only with the same scientific settings; config compatibility is not checked
against the previous saved run. Avoid simultaneous launches with the same tag:
timestamps have one-second resolution.

### Shell launchers (`.sh`)

Run these scripts with **Bash from the repository root**. They select
`.venv/bin/python` when available, then `.venv/Scripts/python.exe`, then Python
on `PATH`. They use Bash/Linux utilities such as `nproc` and `find`.

**Parallel seeds.** These wrappers call the Python seed-parallel drivers using
their default config batches. The optional first positional argument is the
worker count; without it, they use `nproc - 2`, with a minimum of one.

```bash
bash scripts/EXPA/runA_parallel_seeds.sh 8
bash scripts/EXPB/runB_parallel_seeds.sh 8
bash scripts/EXPC/runC_parallel_seeds.sh 8
```

Each wrapper forces one BLAS/OMP thread per worker and redirects Python output
to its experiment's log file:

| Script | Driver | Log file |
| --- | --- | --- |
| [runA_parallel_seeds.sh](scripts/EXPA/runA_parallel_seeds.sh) | `scripts.EXPA.run_parallel_seeds` | `logs_cpu_runs/parallel_seeds.out` |
| [runB_parallel_seeds.sh](scripts/EXPB/runB_parallel_seeds.sh) | `scripts.EXPB.run_parallel_seeds` | `logsB_cpu_runs/parallel_seeds.out` |
| [runC_parallel_seeds.sh](scripts/EXPC/runC_parallel_seeds.sh) | `scripts.EXPC.run_parallel_seeds` | `logsC_cpu_runs/parallel_seeds.out` |

```bash
# Follow an ongoing launch in another terminal.
tail -f logsC_cpu_runs/parallel_seeds.out

# Detached execution of C's default batch, where permitted on the machine.
nohup bash scripts/EXPC/runC_parallel_seeds.sh 8 > logsC_launcher.out 2>&1 &
```

These `.sh` wrappers currently accept **only the positional worker count**:
they do not forward `--config`, `--schedule`, `--reuse-existing`, or
`--results-root`. Use the Python commands above for those options. The fixed log
filename is overwritten on each launch; save it separately if needed.

**Parallel configurations.** The `all_cpu.sh` scripts start independent
sequential runners concurrently, rather than distributing individual seeds.
Their config lists and concurrency limits are embedded in the shell files.

| Script | Current selection and status |
| --- | --- |
| [runA_all_cpu.sh](scripts/EXPA/runA_all_cpu.sh) | Hartmann-6 from `ADAPTATIVE_MATERN_BETA/`; one log per config in `logs_cpu_runs/` |
| [runB_all_cpu.sh](scripts/EXPB/runB_all_cpu.sh) | Eight good/bad-prior configs from `ADAPTATIVE_LOG_BETA/`; one log per config in `logsB_cpu_runs/` |
| [runC_all_cpu.sh](scripts/EXPC/runC_all_cpu.sh) | Legacy batch; its fixed-mode config paths are missing. Use the seed-parallel Python driver or its shell wrapper instead. |
| [run_test.sh](scripts/EXPB/run_test.sh) | B's three double-well prior-center configs in `TEST_PRIOR_CENTER/`; this generates experimental data, despite its name. |

```bash
# These launch the scientific batches selected inside each script.
bash scripts/EXPA/runA_all_cpu.sh
bash scripts/EXPB/runB_all_cpu.sh
```

For processing and plotting existing runs with `.sh` scripts, see
[Shell postprocessing shortcuts](#shell-postprocessing-shortcuts).

### SLURM

C has an existing [SLURM launcher](scripts/EXPC/runC_parallel_seeds.sbatch)
that forwards arguments to its Python driver and uses the allocated CPU count:

```bash
mkdir -p logsC_cpu_runs
sbatch --partition=YOUR_PARTITION --cpus-per-task=8 \
  scripts/EXPC/runC_parallel_seeds.sbatch \
  --config experiments/exp_c_multiobjective/configs/pareto_front/zdt1_d2.yaml
```

Submit from the repository root with the Python environment available on the
compute node. Adjust partition, memory, and time limits for the cluster.
The supplied defaults are site-specific. A/B have no dedicated SLURM scripts.

## 6. Experiment-specific protocols

### Shared modeling and evaluation conventions

BO uses noisy observations; synthetic evaluation uses noiseless benchmark
values. A/B distinguish `y_obs` from `f_noiseless`. C distinguishes
noisy `model_y` from noiseless minimization `raw_y` and maximization
`oriented_y`. Methods never train on held-out noiseless diagnostic values.

The runners use double precision, scrambled Sobol initial designs, and
sequential acquisitions with one candidate per step. GP hyperparameters are
fitted by exact marginal log-likelihood at each step. The supported kernels are
`matern52` (default), `matern32`, `matern12`, and `rbf`.
With BoTorch 0.7, the default is a scaled Matérn 5/2 kernel with ARD.

Input normalization and output standardization are internal GP transforms;
they are separate from objective orientation or scalarization. A/C normally
learn normalization bounds from training inputs; B uses fixed domain bounds.
A's ablations can use fixed domain bounds via
`model.normalize_with_problem_bounds: true`.

Known noise standard deviation $s$ becomes a variance $s^2$ in
`FixedNoiseGP`; GPyTorch can impose a numerical floor. Output
standardization also transforms that variance. Reusing this noise model after
absolute-value or nonlinear scalarization is an approximation to transformed
noise, especially near a kink or truncation.

Acquisition optimization uses multistart `optimize_acqf`.
`num_restarts` and `raw_samples` control initialization. Exploration
schedules include constant, logarithmic, and `matern_gamma` policies. The
logarithmic convention is $\beta_t=\beta_0+c\log(t+1)$.

### A — Benchmarks, budget, and metrics

Available problems include linear, exponential, sine-trend, Forrester,
Gaussian-trap variants, and damped oscillatory trap in 1D, Branin in 2D,
and Hartmann in 6D. Representative base configs are:

| Config under `experiments/exp_a_target_matching/configs/` | Initial + additional BO evaluations | Seeds |
| --- | --- | --- |
| `branin_2d_short.yaml` | 4 + 50 | 1–100 |
| `hartmann_6d_full.yaml` | 12 + 50 | 1–100 |
| `gaussian_trap_1d_short.yaml` | 1 + 50 | 1–100 |

A/B's `budget` counts **additional BO evaluations**. Methods share initial
inputs and noisy observations within each seed. A's two methods both use fixed
benchmark noise. Most base scientific configurations use zero observation
noise, subject to the GP's numerical noise floor. A's runner enables GP input
normalization and output standardization by builder defaults; general YAML
flags for disabling these are not forwarded.

Define $d^\star=\min_x\lvert f(x)-T\rvert$. The main synthetic metric is

$$
R_n^{\mathrm{target}}=
\min_{1\leq s\leq n}\lvert f_{\mathrm{noiseless}}(x_s)-T\rvert-d^\star.
$$

Cumulative target regret sums the **instantaneous** target distances minus
$d^\star$, not best-so-far regrets. Secondary summaries include AUC, hitting
time, and failure rate. Processing uses the benchmark's target-distance oracle
when available, otherwise zero as the reference; the latter is justified only
when the target is attainable.

Calibration uses held-out global and near-target grids. RIGHT is evaluated
against the raw function; WRONG's native calibration concerns its transformed
function. Common target-distance diagnostics are described in Section 7.

### B — Priors, budget, and metrics

Good/bad-prior configs cover Forrester and double Gaussian well in 1D, Branin
in 2D, and Hartmann in 3D/6D. Base scientific configurations use seeds 1–50,
zero observation noise, and these initial/additional BO counts: 1/50 in 1D,
4/50 for Branin, 6/60 for Hartmann-3, and 12/80 for Hartmann-6.
The YAML method list controls which comparisons run.

For an oriented maximization objective, evaluation uses

$$
r_t=f^\star-f_{\mathrm{noiseless}}(x_t),
\qquad
R_n^{\mathrm{simple}}=\min_{t\leq n}r_t,
\qquad
R_n^{\mathrm{cumulative}}=\sum_{t=1}^{n}r_t.
$$

Calibration is measured globally and in the masked prior region where
$\pi(x)$ exceeds `calibration.prior_region_threshold`.
`model.use_fixed_noise` selects fixed or inferred noise. Raw prior-mean
methods disable output standardization; standardized variants enable it and
z-score the prior over training inputs, with a deterministic fallback grid.
The raw prior mean uses known synthetic output-range information, which must
be supplied or estimated independently for a real application.

### C — Preferences, budget, and metrics

ZDT1 and ZDT2 each have two objectives, with input dimension 2 or 6 on
$[0,1]^d$. ZDT1 has a convex front; the interior of ZDT2's concave front is
unsupported by positive linear scalarizations. Failure to select that interior
with a weighted sum is a scalarization limitation.

| Mode | Config location | Standard total budget per method |
| --- | --- | --- |
| Fixed, log beta | `configs/fixed_scalarization/LOG_BETA/` | 50 in 2D, 200 in 6D, **per scalarization/angle case** |
| Fixed, constant beta | `configs/fixed_scalarization/beta=8/` | Same budget convention |
| Pareto, log beta | `configs/pareto_front/` | 500 |
| Pareto, constant beta | `configs/pareto_front/beta=8/` | 500 |

Unlike A/B, C's `budget` **includes all initial evaluations**. Fixed mode
starts with 5 points in 2D or 12 in 6D. It uses angles
$\theta\in\{10,30,45,60,80\}$ degrees with
$\lambda=(\cos\theta,\sin\theta)$ for both scalarization families.

Pareto Algorithm 1 starts with 5 points. Algorithm 2 divides the same global
budget across 10 independent subruns, each starting with 2 points. RIGHT and
WRONG Algorithm 2 share the subrun preferences and initial observations.
Stored anytime curves interleave subruns in round-robin budget order,
including initialization; this is not physical chronology between subruns.

Optimization preferences are uniform on the positive unit sphere. A separate
RNG stream draws 2,048 evaluation preferences per seed, shared across methods.
Standard schedules are logarithmic beta starting at 8 or constant beta 8.
Known per-objective noise is `noise.std: 1.0e-6`; fixed noise is the default,
with model standard deviation at least $10^{-6}$. WRONG reuses that model
noise after scalarization. C's `benchmark.normalization: fixed` does not
apply additional objective rescaling.

For fixed preference, $v_t=\rho_\lambda(F(x_t))$ is evaluated noiselessly:

$$
R_n^{\mathrm{simple}}=\rho_\lambda^\star-\max_{t\leq n}v_t,
\qquad
R_n^{\mathrm{cumulative}}=
\sum_{t=1}^{n}\left(\rho_\lambda^\star-v_t\right).
$$

Processing recomputes the reference optimum on a dense analytic front.
Pareto evaluation includes exact 2D hypervolume regret, empirical Bayes regret
over independent preferences, dominance distances, dominated-query fraction,
and IGD. The reference point and normalization scales are fixed across methods.
HV regret measures the accumulated set; cumulative dominance regret measures
the quality of individual queries. IGD measures front coverage and is not
universally monotone when dominated archive points are removed.

See [Experiment C's protocol](experiments/exp_c_multiobjective/README.md) for
pairing checks, reference grids, tolerances, and figure selection.

## 7. Lengthscale and RKHS ablations

Experiment A's [ablation guide](experiments/exp_a_target_matching/ABLATIONS.md)
describes a study applicable to **every synthetic Experiment A benchmark**,
including Branin-2D and Hartmann-6D. For each benchmark separately, cross
**Matérn 5/2 vs Matérn 1/2** with **MLL-optimized vs fixed lengthscales**,
comparing RIGHT and WRONG in each of the four arms.

Keep that benchmark's target, bounds, initial design, budget, noise, seeds,
exploration schedule, and optimizer settings identical across arms. Use fixed
domain normalization in all four arms. Only the kernel family and lengthscale
policy differ; mean and outputscale remain trainable. `paired_method_rng: true`
resets the method RNG to the seed, preserving a common RNG initialization,
not identical adaptive query points.

The checked-in `linear_1d_*` configs are **pilot examples**, not a restriction
on the ablation. Their fixed lengthscale of 0.2 in normalized input units is a
preselected example, not a universal choice for every benchmark. Existing
benchmark configs are not automatically expanded into four runs.

### Enable diagnostics for a run

Norm/error diagnostics can also be computed for an ordinary run without
executing the four-arm comparison. Add this block before processing:

```yaml
generalization:
  enabled: true
  n_test: 2048
  seed: 123
```

The default grid is `linspace` in 1D and `sobol` in higher dimensions.
An explicit `grid: linspace` requires 1D. The same deterministic held-out
grid is shared across methods and snapshots. Native-function RMSE and latent
posterior variance have different output units for RIGHT and WRONG and should
not be ranked directly across those methods.

### Effective RKHS norm

For a saved GP, let $z$ be its training targets in internal model coordinates,
$m$ its fitted prior mean, $K$ its fitted training covariance including
outputscale, and $D$ its actual likelihood noise diagonal:

$$
\alpha=(K+D)^{-1}\bigl(z-m(X)\bigr),
\qquad
B_{\mathrm{eff}}^2=\alpha^\top K\alpha.
$$

The implementation uses a linear solve. This is the norm of the **centered
posterior mean** in the fitted kernel, excluding the prior mean. It is not an
imposed norm budget or a norm of the unknown true objective.

For output standardization $y=a+sz$ and fitted kernel
$K=\sigma_{\mathrm{out}}^2K_\ell$, the centered raw-output mean has norm

$$
B_{\mathrm{unit}}=s\,\sigma_{\mathrm{out}}\,B_{\mathrm{eff}}
$$

in the unit-outputscale kernel $k_\ell$. This removes output-amplitude
conventions. It provides a common reference only when kernel family,
lengthscale, and input coordinates agree; it does not make norms across
different Matérn families or MLL-adapted lengthscales directly comparable.

### Common target-distance error

Both methods are evaluated against the same noiseless function:

$$
\begin{aligned}
g(x)&=\lvert f_{\mathrm{noiseless}}(x)-T\rvert,\\
\widehat g_R(x)&=\lvert\mu_f(x)-T\rvert,
&\widehat g_W(x)&=\mu_g(x),\\
\mathrm{RMSE}_g&=
\sqrt{\frac{1}{M}\sum_{i=1}^{M}
\left(\widehat g(x_i)-g(x_i)\right)^2}.
\end{aligned}
$$

RIGHT uses a plug-in mean prediction, not the posterior expectation of an
absolute value. WRONG predictions are not clipped. Near-target RMSE restricts
the **same grid** to points satisfying $g(x)\leq\varepsilon$.
`generalization.near_target_epsilon` overrides the top-level
`near_target_epsilon`; the threshold and selected point count are saved.
Without a threshold or with an empty subset, the local error is unavailable.

### Generate individual-run and cross-run figures

```bash
python -m experiments.exp_a_target_matching.process --run-dir "$RUN_DIR"
python -m experiments.exp_a_target_matching.analyze --run-dir "$RUN_DIR"
python -m experiments.exp_a_target_matching.plot --run-dir "$RUN_DIR"

# Only the saved RKHS panels; preserves other figures and avoids the
# initial-design GP fits used by ordinary posterior plots.
python -m experiments.exp_a_target_matching.plot --run-dir "$RUN_DIR" --rkhs-only
```

Normal `plot.py` adds these files to the other run figures when the
aggregated diagnostics are available:

| Figure basename, PNG and PDF | Content |
| --- | --- |
| `05_gp_generalization` | Lengthscales, native RMSE, mean latent variance |
| `05_gp_rkhs_complexity` | All ARD lengthscales, fitted/unit-outputscale norms, common global/near-target RMSE |

Both diagnostic figures retain every saved ARD lengthscale component, with
dimension labels. Snapshot diagnostics describe models **before** the
corresponding query.

For comparisons across separately executed arms of the **same benchmark**:

```bash
python -m experiments.exp_a_target_matching.compare_ablations \
  --run-dir "$RUN_MATERN52_MLL" --run-dir "$RUN_MATERN52_FIXED" \
  --run-dir "$RUN_MATERN12_MLL" --run-dir "$RUN_MATERN12_FIXED" \
  --output-dir results/exp_a_target_matching/ablation_comparison
```

For the included 1D example, use configs `linear_1d_matern52_mll.yaml`,
`linear_1d_matern52_fixed.yaml`, `linear_1d_matern12_mll.yaml`, and
`linear_1d_matern12_fixed.yaml` under `configs/ablations/`.
The comparison validates matching settings, seed IDs, and initial observations,
and writes `lengthscale_ablations` and `rkhs_complexity_ablations` in
PNG/PDF. The same command supports higher-dimensional benchmarks: supply
their four run directories and use a separate output directory per benchmark.
Changing kernel or lengthscale policy requires new BO trajectories; reprocessing
a single existing run only adds diagnostics for its already fitted models.

Existing runs with sufficient GP snapshots and training data can be reprocessed
without rerunning BO. Enable generalization in their saved `config.json`,
retain a record of this diagnostic-only edit, and process **all** seeds before
aggregating. The processors reconstruct saved GPs without refitting.

### Interpretation

A fixed-RKHS approximation barrier concerns a fixed kernel space and a
norm-constrained approximation family:

$$
\left\lbrace h \in \mathcal{H}_k \mid \lVert h \rVert_{\mathcal{H}_k} \leq B \right\rbrace.
$$

MLL changes lengthscale and therefore kernel geometry. A shorter lengthscale
does **not** necessarily imply a larger RKHS norm. Even fixed-lengthscale arms
still learn amplitude and mean. These plots investigate associations between
complexity and approximation error along adaptive trajectories; they do not
establish causality, an asymptotic barrier, or theoretical rates. Small pilots
are descriptive checks, not robust inferential evidence.

## 8. Results and visualization

### Processing pipeline

For A/B, use the experiment's `process`, `analyze`, then `plot`
module. For example:

```bash
RUN_DIR='results/exp_b_prior_guided/runs/<generated_run_id>'
python -m experiments.exp_b_prior_guided.process --run-dir "$RUN_DIR" --verify
python -m experiments.exp_b_prior_guided.analyze --run-dir "$RUN_DIR"
python -m experiments.exp_b_prior_guided.plot --run-dir "$RUN_DIR"
```

C generates figures in `analyze.py`; **there is no separate C `plot.py`**:

```bash
RUN_DIR='results/exp_c_multiobjective/runs/<generated_run_id>'
python -m experiments.exp_c_multiobjective.process --run-dir "$RUN_DIR"
python -m experiments.exp_c_multiobjective.analyze --run-dir "$RUN_DIR"

# Reorganize already generated C figures and refresh selected panels.
python -m experiments.exp_c_multiobjective.analyze \
  --run-dir "$RUN_DIR" --refresh-selection
```

A/B processing supports repeatable `--seed` and `--benchmark` filters.
Their analysis supports statistic, bootstrap count, confidence level, and
pointwise/uniform band options; B also accepts `--baseline`.
C accepts repeatable `--run-dir` for an Algorithm-2 subrun-count overlay;
its main summaries use the first run. Use compatible runs differing in subrun
count, not unrelated benchmark configurations.

### Shell postprocessing shortcuts

The `last_runs.sh` scripts automate later pipeline stages; they **do not
generate new BO trajectories**. The optional positional argument selects how
many of the most recently modified **matching** run directories to process.
It must be a positive integer. A/B default to one run; C defaults to eight.

| Experiment | Script | Stages executed |
| --- | --- | --- |
| A | [processA_last_runs.sh](scripts/EXPA/processA_last_runs.sh) | `process` then `analyze` |
| A | [plotA_last_runs.sh](scripts/EXPA/plotA_last_runs.sh) | `plot` |
| B | [processB_last_runs.sh](scripts/EXPB/processB_last_runs.sh) | `process` then `analyze` |
| B | [plotB_last_runs.sh](scripts/EXPB/plotB_last_runs.sh) | `plot` |
| C | [processC_last_runs.sh](scripts/EXPC/processC_last_runs.sh) | `process` only |
| C | [plotC_last_runs.sh](scripts/EXPC/plotC_last_runs.sh) | `analyze`, including figures |

```bash
# A: process/analyze, then plot the latest matching run.
bash scripts/EXPA/processA_last_runs.sh 1
bash scripts/EXPA/plotA_last_runs.sh 1

# B: process/analyze, then plot the latest matching run.
bash scripts/EXPB/processB_last_runs.sh 1
bash scripts/EXPB/plotB_last_runs.sh 1

# C: process, then analyze/plot the latest eight matching runs.
bash scripts/EXPC/processC_last_runs.sh 8
bash scripts/EXPC/plotC_last_runs.sh 8
```

Selection uses hardcoded tag patterns under `results/<experiment>/runs/`.
It is not an unrestricted selection of the last runs: custom tags, A's
ablation tags, and C's `_logbeta`/`_beta8` suffixes may not match. In particular,
the current C wrappers do not select the schedule-suffixed runs produced by
its default Python batch. The examples above apply to matching tags; use the
explicit Python `--run-dir` commands for a specific run or another results
root. These shortcuts do not forward diagnostic options such as
`--rkhs-only` or `--refresh-selection`.

### Output layout

```text
results/<experiment>/runs/<UTC_timestamp>_<tag>/
  config.json
  metadata.json                              command, platform, Git state, seeds, ...
  raw/<benchmark>/seed_NNN.json               all compared methods for one seed
  processed/per_seed/<benchmark>/seed_NNN_processed.json
  processed/aggregate/<benchmark>/           A/B summaries and bootstrap curves
  processed/aggregate_summary.json           C summary
  processed/paired_bootstrap_comparisons.json C paired statistics
  figures/<benchmark>/                       A/B PNG/PDF
  figures/                                   C main PNG/PDF
    supplementary/                           C secondary views
    archive/                                 C legacy replaced/mixed-metric figures
    figure_selection.json                    C file-move manifest
  logs/                                      stage logs
```

A/B aggregate files include `final_performance.json`,
`convergence_curves.json`, `calibration_summary.json`,
`calibration_curves.json`, and `statistical_tests.json`.
A additionally writes `generalization_curves.json` when enabled. Curve
availability depends on the saved configuration.

### Figures and uncertainty

A's standard figures cover benchmark geometry, posterior/acquisition
snapshots, query trajectories, target-distance convergence, final
distributions, paired comparisons, calibration, and GP hyperparameters.
B adds prior geometry, learned prior coefficients, and query/prior diagnostics.
Geometry and posterior views depend on dimension and available snapshots.
Calibration includes NLPD, coverage, PIT, standardized residuals, confidence
bound violations, and optional joint diagnostics.

For held-out values of a specified modeled quantity, marginal Gaussian NLPD is

$$
\mathrm{NLPD}=\frac{1}{M}\sum_{i=1}^{M}
\left[\frac{(q_i-\mu_i)^2}{2\sigma_i^2}
+\frac{1}{2}\log(2\pi\sigma_i^2)\right].
$$

Here $q_i$ is the evaluation truth in the prediction's units; its definition
must match the diagnostic (raw objective, transformed distance, or observation).
Coverage is the fraction inside the specified predictive intervals; PIT values
are predictive CDF values at the evaluation truth. Joint NLPD additionally uses
posterior correlations. Latent and noisy-observation diagnostics are distinct.

C's main selection retains fixed-preference query maps at **every angle**,
simple/cumulative regret panels and aggregates, all-seed preference maps,
aggregate distributions, and separate paired-bootstrap comparisons.
Pareto views retain query maps, final fronts, HV regret, cumulative dominance
regret, dominated-query fraction, IGD, and selected final/paired summaries.
Secondary figures remain in `supplementary/`; legacy mixed-metric plots and
old best-so-far panels are archived without deletion.

A/B primarily use paired Wilcoxon comparisons, with additional effect sizes
and unpaired tests. C reports mean paired RIGHT-minus-WRONG differences with
95% percentile bootstrap intervals from 5,000 seed-block replications.
Fixed-preference aggregates average angles within each seed before resampling.
Negative regret differences favor RIGHT. Algorithm 1 vs WRONG Algorithm 2 is
paired by seed but does not share an initial design; RIGHT vs WRONG Algorithm 2
has the stricter shared-subrun pairing.

C's trajectory bands are inter-seed IQRs, not confidence intervals; its fixed
aggregate band is the min–max range of per-angle medians. A's generalization
bands are pointwise bootstrap intervals for seed means. Always distinguish
trajectory variability from uncertainty on a mean.

## 9. Limitations and reproducibility notes

- Save raw observations, config snapshots, code revision, and dependency versions.
  A/B GP snapshots require a compatible GP stack. A Git hash does not capture
  uncommitted edits; metadata also records the dirty state.
- Current experiments are synthetic. Real-data metrics must not assume
  `f_noiseless`, exact optima, true fronts, or known objective ranges.
- Data-dependent normalization, output standardization, mean fitting, and noise
  floors affect GP interpretation. Native errors and fitted-kernel norms across
  different modeled functions or kernels are not automatically comparable.
- Most model diagnostics reconstruct pre-query snapshots; they are not a
  separately fitted posterior after the final evaluation. Ordinary A/B plotting
  may fit an initial-design GP for visualization; A's `--rkhs-only` path does not.
- A's historical `configs/smoke_1d.yaml` actually specifies Branin-2D,
  20 seeds, 20 initial points, and 50 BO steps. Use the Quick Start config instead.
- The legacy `scripts/EXPC/runC_all_cpu.sh` references missing fixed-mode
  config paths. Use C's Python launcher. The latest-run shell wrappers may not
  match schedule-suffixed C tags; explicit `--run-dir` is more reliable.
- A/B ordinary plotting clears existing PNG/PDF files in its benchmark figure
  directory before exporting the regenerated set. Keep manually curated files
  elsewhere. A's `--rkhs-only` preserves other files.
- Incomplete runs summarize available seeds only. A single seed cannot support
  an identifiable paired bootstrap interval; pilot uncertainty is limited.
- CPU/GPU and numerical library differences can change optimization paths.
  The launchers preserve RNG logic; they do not promise cross-platform bitwise
  reproduction.

Lightweight verification commands, without scientific sweeps:

```bash
python -m compileall common acquisitions experiments scripts
python -m scripts.EXPA.run_parallel_seeds --help
python -m scripts.EXPB.run_parallel_seeds --help
python -m scripts.EXPC.run_parallel_seeds --help
pytest
```

The `scripts/smoke_test_exp_*_workflow.py` utilities execute small BO
workflows and create results; they are separate from help/compilation checks.
