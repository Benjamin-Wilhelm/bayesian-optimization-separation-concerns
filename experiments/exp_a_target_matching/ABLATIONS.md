# Lengthscale and generalization ablations

These optional experiments retain the absolute target-matching objective.
RIGHT trains on `y_obs`; WRONG trains on `abs(y_obs - target)`. Synthetic
evaluation uses `f_noiseless` or `abs(f_noiseless - target)` only. Existing
configurations retain their previous model fitting and acquisition behavior.

The four-arm study applies to **every synthetic benchmark in Experiment A**,
not only to 1D functions. The checked-in linear configs below are pilot
examples. Both per-run diagnostics and `compare_ablations.py` support Branin-2D,
Hartmann-6D, and the other registered synthetic benchmarks. Comparisons are
within one benchmark at a time; results from different problems are not pooled.

For another benchmark, copy its own YAML into four separately tagged configs.
Keep target, bounds, initial design, budget, noise, seeds, beta schedule and
optimizer settings identical across those four configs. Set
`model.normalize_with_problem_bounds: true`, `paired_method_rng: true`, and
`generalization.enabled: true` in **all** arms. Cross `model.kernel: matern52`
and `matern12` with an absent/null `model.fixed_lengthscale` (MLL) or a
preselected positive scalar/ARD vector (fixed). Select fixed values explicitly
for that study; the pilot's 0.2 is not a universal setting. Normalization and
RNG controls define the matched ablation; they do not modify existing configs.
No new configurations or BO runs are created automatically by plotting.

## Included linear 1D pilot

The files in `configs/ablations/` cross Matérn 5/2 and Matérn 1/2 with
MLL-optimized and fixed lengthscales. They reuse `linear_1d_short.yaml`'s
target (0.6), initial design size (1), BO budget (50 additional evaluations),
zero observation noise, beta (2), and acquisition optimizer settings. The
pilot uses the first three existing seeds. Full runs are not launched by
installation or processing. `smoke_lengthscale.yaml` is a separate technical
check with two seeds, three initial observations and three BO evaluations.

All four pilot configs use normalization to the known problem bounds.
`model.fixed_lengthscale: 0.2` freezes only the lengthscale, measured in these
normalized input units; mean and outputscale remain MLL optimized. The noise
policy is unchanged. The MLL controls use the same fixed input normalization,
so normalization does not confound the comparison. The value 0.2 is a
preselected example, not an optimum; repeat with other preselected values
before drawing conclusions about adaptation. ARD vectors are also supported.

`paired_method_rng: true` resets the Torch RNG to the seed before each method.
Across ablation configs this gives a common optimizer/noise RNG stream per
method, in addition to the shared initial inputs and noisy observations.
It does not force different adaptive trajectories to query identical points.
Legacy configurations keep their existing RNG behavior.

```bash
python -m experiments.exp_a_target_matching.run \
  --config experiments/exp_a_target_matching/configs/ablations/smoke_lengthscale.yaml

python -m experiments.exp_a_target_matching.process --run-dir RESULTS_RUN
python -m experiments.exp_a_target_matching.analyze --run-dir RESULTS_RUN
python -m experiments.exp_a_target_matching.plot --run-dir RESULTS_RUN
```

For each pilot arm, substitute one of these config names for the smoke config:

- `linear_1d_matern52_mll.yaml`
- `linear_1d_matern52_fixed.yaml`
- `linear_1d_matern12_mll.yaml`
- `linear_1d_matern12_fixed.yaml`

Generalization is opt-in through:

```yaml
generalization:
  enabled: true
  n_test: 2048
  seed: 123
```

When `grid` is omitted, the default is `linspace` in 1D and `sobol` in higher
dimensions. An explicit `grid: linspace` requires a 1D problem. The fixed dense
1D grid includes the domain endpoints; Sobol uses the benchmark's test-set
interface and a fixed seed. Each grid is shared by all methods, iterations and
matched arms. Generalization grids are separate from existing calibration
grids; their definition is stored with the reports. Processing can also be
enabled on existing snapshots by explicitly adding this block to a saved run
config, without repeating BO.

`process.py` restores every snapshot without fitting. At iteration t the model
has seen `initial_design_size + t - 1` observations; diagnostics describe the
model **before** query t. It saves native RMSE, mean **latent** posterior variance
(without observation noise), normalized RMSE, and the stored hyperparameters
under each processed method's `generalization` list. Normalized RMSE divides
by the standard deviation of the modeled noiseless function on the fixed grid;
it is null for constant functions. Native RMSE and variance have different
units for RIGHT and WRONG and should not be directly ranked across methods.

`analyze.py` writes `generalization_curves.json`, with means over seeds and
pointwise bootstrap confidence intervals (whole trajectories are resampled).
`plot.py` adds `05_gp_generalization.png` and `.pdf`; existing hyperparameter
plots remain available. Snapshots store fixed-lengthscale and input-bound
settings so reconstruction preserves both predictions and fitting policy.
Legacy snapshots and processed files remain readable.

## RKHS complexity and common target-distance error

Reprocess saved runs with generalization enabled, then rerun `analyze.py` and
`compare_ablations.py` below. Neither processing nor comparison fits a model
or reruns BO. The new fields are additive; older processed reports still load,
and plots omit unavailable diagnostics. All seeds must be reprocessed before
aggregating a new diagnostic; partial upgrades are not silently pooled.

For an individual run, this plots only the new saved diagnostics, without
fitting initial-design GPs for the older posterior figures or clearing files:

```bash
python -m experiments.exp_a_target_matching.plot --run-dir RESULTS_RUN --rkhs-only
```

Each snapshot report now contains:

- `effective_rkhs_norm` and `effective_rkhs_norm_squared`: the norm B and B²
  of the **centered** posterior mean in that snapshot's fitted kernel.
- `effective_rkhs_norm_unit_outputscale`: the raw centered posterior mean's
  norm in the corresponding unit-outputscale base kernel (explained below).
- `target_distance_rmse`: error against the shared noiseless truth
  `g = abs(f_noiseless - target)`. RIGHT predicts `abs(mu_f - target)`;
  WRONG predicts `mu_g`, including any negative predictions without clipping.
  These are plug-in mean predictions, not posterior expectations of `abs(f-T)`.
- `target_distance_rmse_near_target`: the same error restricted to points
  with `abs(f_noiseless - target) <= near_target_epsilon` on the **same grid**.
  The epsilon defaults to the run's top-level `near_target_epsilon`; an explicit
  `generalization.near_target_epsilon` overrides it. The threshold and point
  count are saved in `near_target`. With no threshold or an empty subset,
  local RMSE is null; no points are resampled to fill the region. Check the
  count and use an adequately dense grid for local interpretation.

Native `rmse`, `normalized_rmse` and latent variance remain unchanged. Common
distance RMSE has the same units and truth for both methods. These synthetic
evaluation values never enter the training targets or acquisition.

### Norm convention and numerical computation

In the GP's internal output coordinates, let `z` be the training observations
(raw or absolute-distance observations according to the method), after any
output standardization; `m` the learned prior mean; `K` the training covariance
including learned outputscale; and `D` the actual likelihood training noise
diagonal. We compute

```text
r = z - m(X)
alpha = solve(K + D, r)
B_eff² = alpha.T @ K @ alpha
```

The solve uses Cholesky factorization, with numerical jitter only if needed.
`rkhs_context` records the solve jitter in internal covariance units, raw prior
mean, outcome mean/std, raw outputscale, and raw noise range. The likelihood's
actual noise includes any library floor, even for a configured zero-noise run.
Fixed and inferred noise are both supported. Inputs use exactly the saved
model's normalization and hyperparameters.

For an output transform `y = a + s*z`, the raw kernel is `s²*K`, the raw noise
is `s²*D`, and the centered raw mean is `s*K(x,X)*alpha`. Its norm in that raw
kernel is also B_eff: standardization is accounted for consistently, without
scaling the residual alone. The raw prior mean `a + s*m(x)` is **excluded**.
An arbitrary prior mean need not belong to the kernel's RKHS, so this finite
kernel expansion does not in general determine the norm of the entire
uncentered posterior mean. The diagnostic measures the fitted correction to
the prior; changing prior means can also affect approximation error.

### Connection to the fixed-RKHS approximation barrier

Theorem 3.2 concerns approximation within a **fixed RKHS**. A trajectory of
fitted GPs is not a norm-constrained optimization experiment, and neither
monotonic norm growth nor a theoretical rate is implied by these diagnostics.
MLL changes the lengthscale and thereby the kernel geometry: effective norms
across such kernels cannot be directly ranked as approximation complexity.
A shorter lengthscale does **not** necessarily imply a larger norm.

Even the fixed-lengthscale arms still optimize outputscale and prior mean,
and their outcome standardization changes with the data. They therefore do
not literally hold the entire fitted kernel and prior fixed. To remove the
amplitude convention, for `K = outputscale*k_ell` we additionally report

```text
B_unit = s * sqrt(outputscale) * B_eff
```

This is the norm of the centered **raw-output** mean under `k_ell` with unit
outputscale. Within a fixed-lengthscale arm, and between RIGHT and WRONG
using the same kernel family, lengthscale and fixed input coordinates, it
provides a common reference RKHS for their centered expansions. It is still
not a norm of the full prediction including the learned prior. Unit-outputscale
norms across Matérn families or MLL-adapted lengthscales remain different-RKHS
quantities. Do not combine or normalize them into a single complexity ranking.

The most informative comparison is common (especially near-target) error
alongside norm under a fixed reference kernel, within each Matérn family.
Inspect whether WRONG uses more norm for comparable error, or retains local
error while its norm grows, then inspect whether MLL changes lengthscale and
error together. These are empirical associations consistent with or against
an approximation-complexity explanation, not proof of its mechanism. BO also
changes training inputs, noise, mean and amplitude; a small paired pilot
cannot establish causality, an asymptotic barrier, or theoretical rates.

To overlay processed arms of any one benchmark, with identical protocols and
initial data (the included pilot is one example):

```bash
python -m experiments.exp_a_target_matching.compare_ablations \
  --run-dir RUN_MATERN52_MLL --run-dir RUN_MATERN52_FIXED \
  --run-dir RUN_MATERN12_MLL --run-dir RUN_MATERN12_FIXED \
  --output-dir results/exp_a_target_matching/ablation_comparison
```

This writes PNG/PDF overlays for lengthscale, native RMSE, latent variance,
and best observed-in-the-synthetic-evaluation noiseless target distance, with
one row per method. These diagnostics test the proposed mechanism; an
association between short lengthscales and error along adaptive BO trajectories
alone does not establish causality, since the sampled inputs also differ.
It also writes `rkhs_complexity_ablations.png` and `.pdf`, with one row per
method and panels for all ARD lengthscales, fitted-kernel norm, unit-outputscale norm,
common target-distance RMSE, and near-target distance RMSE. Curves are means
over seeds with pointwise bootstrap confidence intervals, resampling whole
seed trajectories. `plot.py` adds the same diagnostic panels for an individual
run as `05_gp_rkhs_complexity.png` and `.pdf`. With two or three pilot seeds,
bands should be read as descriptive checks, not robust inferential evidence.
