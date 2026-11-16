# Experiment C — Multi-objective Bayesian optimization and Separation of Concerns

This experiment extends the **Separation of Concerns (SoC)** principle to
multi-objective Bayesian optimization (BO).  A vector-valued objective
`F(x) = (f_1(x), ..., f_m(x))` is combined with a *preference* `lambda` through
a monotone scalarization `g_lambda(x) = rho_lambda(F(x))`.  The scientific
question is **where** the preference should enter: in the probabilistic model,
or only in the acquisition.

The whole experiment is a single shared multi-objective pipeline with two
explicitly separated runners:

```python
run_fixed_scalarization(...)   # question 1
run_pareto_front(...)          # question 2
```

## 1. RIGHT vs WRONG

**RIGHT (SoC-compliant).**  Model each objective separately with its own
generic GP (a `ModelListGP` of `m` single-output GPs), trained on the raw
oriented objectives `F`.  The preference `lambda` never enters the GP targets.
The objective-wise UCBs

```
A_{j,t}(x) = mu_{j,t}(x) + sqrt(beta_t) * sigma_{j,t}(x)
```

are combined **after** the acquisition vector is built:

```
a^RIGHT_t(x; lambda) = rho_lambda( A_{1,t}(x), ..., A_{m,t}(x) ).
```

The models therefore describe the objective map `F`, independently of the
preference.

**WRONG.**  For a **fixed** `lambda`, transform each observation to
`g_lambda(x_t) = rho_lambda(F(x_t))` and train a single scalar GP on it, then
maximize a standard scalar UCB `mu^g_t + sqrt(beta_t) * sigma^g_t`.  WRONG is
statistically coherent *as long as `lambda` is fixed within the run*.

**Why WRONG is never used with a varying preference.**  If `lambda_t` changed
each iteration, the pairs `(x_t, rho_{lambda_t}(F(x_t)))` would no longer be
noisy observations of a single fixed latent function of `x`, so a scalar GP on
them is ill-posed.  A *contextual* WRONG on `G(x, lambda) = rho_lambda(F(x))`
would be coherent, but it is a **different method** (it augments the input
space and asks the GP to learn an analytically-known dependence on `lambda`);
it is out of scope here.  This is enforced structurally: WRONG only appears
with a fixed `lambda` (`fixed_scalarization`, and `wrong_algorithm_2` where each
subrun keeps a constant preference).

## 2. The two scientific questions

### 2.1 `fixed_scalarization` — where to scalarize?

At a fixed `(benchmark, scalarization, lambda, seed)`, RIGHT and WRONG use
**exactly the same** initial design, initial observations, budget, noise,
kernel, `beta_t` schedule, and acquisition-optimizer settings.  The *only*
scientific difference is the location of the scalarization.  A `pairing_hash`
(of the shared initial design + `lambda`) is stored so `process.py` can verify
the pairing.

Two scalarizations are compared: the **weighted sum** and the **hypervolume**
(Golovin–Zhang) scalarization.  Two front geometries are studied:

- **ZDT1** — convex front, every point is *supported* (reachable by maximizing
  some positive weighted sum);
- **ZDT2** — concave front with **non-supported** points: no positive weighted
  sum has its maximum in the interior of the front.

> On ZDT2 the weighted sum cannot reach the non-supported region.  This is a
> limitation of the **linear scalarization family**, not of RIGHT or WRONG.
> The figures and tables let you test this empirically; it is **not** hard-coded
> into the analysis.

### 2.2 `pareto_front` — approximating the whole front

Three methods, all at **equal global objective budget** `N`:

- **RIGHT Algorithm 1** — one shared set of per-objective GPs for the whole
  run.  Each iteration draws a new `lambda_t ~ Uniform(S_+^{m-1})`, builds the
  objective-wise UCBs, scalarizes, optimizes, observes `F(x_t)`, and updates
  **the same** GPs.  An observation taken under one preference informs every
  future preference.
- **RIGHT Algorithm 2** — draw `L` fixed preferences once; run `L` independent
  RIGHT subruns (own GPs, own initial design, own data, no sharing).  The union
  of all subruns' observations forms the final multi-objective set.
- **WRONG Algorithm 2** — the same `L` preferences and the same per-subrun
  initial designs as RIGHT Algorithm 2, but each subrun trains a scalar WRONG
  GP on `rho_{lambda_ell}(F(x))` (fixed `lambda`, so WRONG is coherent).

Comparisons:

```
RIGHT Alg. 1 vs RIGHT Alg. 2   -> effect of sharing information across preferences
RIGHT Alg. 2 vs WRONG Alg. 2   -> RIGHT vs WRONG at multiple fixed scalarizations
RIGHT Alg. 1 vs WRONG Alg. 2   -> overall SoC gain (less diagnostic)
```

Algorithm 1 can win for two distinct reasons the metrics keep separable:
**statistical sharing** (one vector observation reduces uncertainty on every
`rho_lambda(F)` at once) and **geometric overlap** (a single front point is
good for a whole region of preferences).  Neither is a budget artifact — the
global budget is identical.

## 3. Preferences (`lambda`)

Directions are drawn **uniformly on the positive orthant of the unit sphere**
`S_+^{m-1}` via `lambda = |g| / ||g||`, `g ~ N(0, I)` (drawing Uniform(0,1)
coordinates and normalizing would *not* be uniform on the sphere).  The
`fixed_scalarization` experiment uses a deterministic 2-D grid
`lambda = (cos theta, sin theta)` with `theta in {10, 30, 45, 60, 80}` degrees.
The same directions are reused across the linear and hypervolume
scalarizations.  All samplers accept a `torch.Generator` and are reproducible.

## 4. Orientation, normalization, reference point (all fixed)

ZDT functions are minimization; the pipeline runs in **maximization** via the
centralized flip `F = -f_raw` (never re-coded in the methods).  Every result
stores three views: `raw_y` (ZDT minimization, for the raw figures),
`oriented_y` (noiseless maximization, used by the metrics), and `model_y` (what
the GPs train on = `oriented_y` + optional noise).

Normalization is **fixed** (independent of the observations of any method):
for ZDT1/ZDT2 both objectives are `O(1)` on the front, so the orientation flip
alone suffices; no adaptive/online normalization is ever applied.

The hypervolume reference point `z` is a **fixed** point in the oriented
convention, `z = (-1.1, -1.1)`, dominated by the entire true front (which lies
in `[-1, 0]^2` after orientation) but not by every dominated point of the
domain.  It never depends on the trajectory.  Dominance, the reference point,
and the hypervolume are all computed in the same (oriented, maximization)
convention.

## 5. Metrics

**Scalarized (fixed lambda)**, with `v_t = rho_lambda(F(x_t))` and reference
optimum `rho* = max over the dense true front`:

```
best-so-far      b_t = max_{s<=t} v_s
simple regret    rho* - b_t
inst. regret     rho* - v_t
cumulative       R_t = sum_{s<=t} (rho* - v_s)      # sum of INSTANTANEOUS regrets
```

> Critical invariant: cumulative regret accumulates the **instantaneous**
> regrets, never the best-so-far simple regrets.

**Pareto front.**  The 2-D hypervolume is computed **exactly** (rectangle
sweep, never Monte-Carlo) and is monotone non-decreasing along a prefix.  With
`HV*` the hypervolume of the dense true front, the hypervolume regret is
`HV* - HV_z(Y_t)`.  The **empirical Bayes regret** averages the scalarized
regret over an independent set of `M` evaluation preferences `~lambda_j`
(distinct RNG seed, disjoint from every optimization stream, identical across
all compared methods):

```
R_Bayes(X_t) = (1/M) sum_j [ max over front rho_{~lambda_j}(F)
                             - max_{x in X_t} rho_{~lambda_j}(F(x)) ].
```

The hypervolume identity `HV_z(Y) = c_m * E_lambda[ max_y rho^HV_lambda(y - z) ]`
with `c_2 = pi/4` is verified in the tests and used only as a diagnostic; it is
never used for the main 2-D curves.

## 6. Budgets and scheduling

Algorithm 1 spends the whole budget `N` on one shared run (initial design
counted).  Algorithm 2 splits `N` across `L` subruns as evenly as possible:

```
base = N // L;  budgets[i] = base + (1 if i < N % L else 0)   # sum == N
```

Each subrun's initial design counts in its local (and the global) budget; a
local budget smaller than the per-subrun initial design raises an explicit
error.  Curves are drawn against the **global objective budget** using a
deterministic **round-robin** schedule: subrun 0 step 1, subrun 1 step 1, …,
subrun `L-1` step 1, subrun 0 step 2, …  Initial-design points are the first
local steps and are interleaved the same way.  The same order is used for RIGHT
and WRONG Algorithm 2.  The round-robin changes only the *anytime* curves, not
the final set.

> Never compare Algorithm 1 at `N` evaluations against Algorithm 2 at `L * N`.
> `process.py` checks `sum(local budgets) == N`.

The `L` ablation uses `L in {5, 10, 20}` (run three configs; `analyze.py`
overlays them when given several `--run-dir`).

## 7. Result layout

```
results/exp_c_multiobjective/runs/<run_id>/
  config.json                                  # resolved config
  metadata.json                                # git commit, command, seeds, ...
  raw/<benchmark>/seed_NNN.json                # one file per (benchmark, seed)
  processed/per_seed/<benchmark>/seed_NNN_processed.json
  processed/aggregate_summary.json
  figures/*.png  *.pdf
  logs/run.log
```

The raw files store every queried `x`, all three objective views, the fixed
`lambda`/`lambda_sequence`, the shared `evaluation_lambdas`, `rho_star` /
`hv_star`, pairing hashes, per-subrun budgets, GP hyperparameters, and the seeds
used.  All Experiment C metrics are model-independent, so `process.py` and
`analyze.py` re-evaluate the analytic benchmark at the stored `x` and never
rebuild a GP or rerun BO.

## 8. Commands

Runs (any config):

```bash
python -m experiments.exp_c_multiobjective.run \
    --config experiments/exp_c_multiobjective/configs/fixed_scalarization/zdt1_d2.yaml
python -m experiments.exp_c_multiobjective.run \
    --config experiments/exp_c_multiobjective/configs/pareto_front/zdt1_d2.yaml
```

Processing (recompute metrics + verification):

```bash
python -m experiments.exp_c_multiobjective.process \
    --run-dir results/exp_c_multiobjective/runs/<run_id>
```

Figures + aggregate summary (pass several `--run-dir` for the L ablation):

```bash
python -m experiments.exp_c_multiobjective.analyze \
    --run-dir results/exp_c_multiobjective/runs/<run_id>
```

Tests and smoke workflow:

```bash
pytest tests/test_exp_c_*.py
python scripts/smoke_test_exp_c_workflow.py
```

## 9. Available configs

```
configs/
  fixed_scalarization/{smoke_zdt1_d2, zdt1_d2, zdt2_d2, zdt1_d6, zdt2_d6}.yaml
  pareto_front/{smoke_zdt1_d2, zdt1_d2, zdt2_d2}.yaml
```

For the `L` ablation, copy a `pareto_front` config and set
`lambda.n_scalarizations_algorithm_2` to 5 / 10 / 20.

## Additional query and coverage diagnostics

Run `process.py` again on saved results, then `analyze.py` using the commands
above. These steps only re-evaluate noiseless synthetic objectives at saved
queries; they do not refit a GP or run BO. Original metrics and figures retain
their definitions. Existing raw files and optimization configurations are not
modified. Old processed Pareto files need reprocessing for the new diagnostics.

Fixed-preference figures now include paired RIGHT/WRONG objective-space query
paths and strict incumbent improvements for every angle, including the shared
initial design in stored evaluation order. The first available seed is used
for trajectories and compact first/middle/last-angle summaries. Both panels
share limits and iteration color normalization. Gold crosses indicate continuous analytical ZDT optima (including tied
optima); other benchmarks use a dense-front fallback. Stars mark
final incumbents. Preference maps show those final incumbents for each angle,
for the representative seed and all available seeds. The interior of ZDT2's
concave front cannot be selected by positive linear scalarizations; this is a
scalarization limitation rather than an algorithm failure.

Pareto processed methods contain a new `diagnostics` dictionary:

- `dominance_regret`: per-query normalized additive L-infinity distance to
  domination by the continuous analytical ZDT front, calculated by bisection.
- `cumulative_dominance_regret` and `average_dominance_regret`: prefix sum and
  prefix sum divided by the number of global evaluations.
- `dominated_query_fraction`: fraction with dominance distance greater than
  **1e-8 normalized units**. This tolerance-based diagnostic measures strict
  additive domination, not the severity of domination.
- `igd`: mean normalized Euclidean distance from a fixed reference front to
  the accumulated nondominated archive. The 1,001-point reference is uniformly
  resampled in normalized arc length from a 10,001-point analytical front.

Both ZDT objectives have a fixed front span of one, so scales are `(1, 1)`
for every method and seed; no observed-range normalization is used. All these
metrics use noiseless **minimization** values. Existing hypervolume and scalar
metrics continue to use their **maximization** convention. The diagnostic
reference and tolerance are recorded in processed JSON. Initial observations
count toward every prefix. Algorithm 2 uses the stored global round-robin
budget convention; lines describe budgets, not physical chronology across
independent subruns. No new ordering is created.

**IGD is not universally monotone for a nondominated archive.** Adding an
observation can remove an older dominated archive member that was closer to
some reference points in Euclidean distance. IGD is retained exactly as
specified, without a running-minimum transform. Tests verify monotonicity for
archive extensions without this removal effect and include a counterexample
to the unconditional claim.

Main curves retain the median and 25–75% inter-seed IQR, and add a lighter
dashed mean. The aggregate fixed-preference curve still uses the mean of
per-angle medians and the min–max range over angles; its dashed curve is the
mean over seeds and angles. Neither variability band is a confidence interval.
Final regret boxplots include seed points. `paired_bootstrap_comparisons.json`
and separate difference figures report 95% percentile confidence intervals
for **mean(RIGHT − WRONG)**, using 5,000 paired seed bootstrap replications and
a deterministic random seed of zero. Negative differences favor RIGHT for
regrets. Fixed-preference aggregates average angles within each seed and
resample whole seed blocks. Results include the seed IDs, replication count,
and an `identifiable` flag (false for a single available seed). The Alg. 1 vs
WRONG Alg. 2 contrast is paired by seed, but does not share an initial design;
RIGHT vs WRONG Alg. 2 retains the stricter paired subrun design.

Every figure is saved in PNG and PDF under the existing run's `figures/`.
Runs without completed raw seed files cannot yet be analyzed. Summaries of
incomplete runs describe only the available complete seeds, not all configured
seeds. MaxGap is not implemented; IGD remains the coverage diagnostic.
