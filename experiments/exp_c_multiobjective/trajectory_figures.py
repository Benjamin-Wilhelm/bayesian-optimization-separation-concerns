"""Fixed-preference noiseless objective trajectories from saved paired queries."""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from common.serialization import load_json
from common.multi_objective_metrics import scalarized_trajectory_values
from common.pareto_diagnostics import incumbent_indices
from data.raw_multiobjective import FixedSeedData
from experiments.exp_c_multiobjective import plotting
from experiments.exp_c_multiobjective.process import _build_benchmark, _verify_fixed


def case_queries(case, method, benchmark):
    """Initial design followed by stored BO steps; no objective-space sorting."""
    steps = case.methods[method].trajectory
    iterations = [s.iteration for s in steps]
    if iterations != sorted(iterations) or len(set(iterations)) != len(iterations):
        raise ValueError("Stored fixed trajectory is not chronological")
    X = torch.tensor(case.initial_X + [s.x for s in steps], dtype=torch.double)
    raw_y = benchmark.evaluate_raw(X)
    values = scalarized_trajectory_values(-raw_y, torch.tensor(case.lambda_vector),
                                          case.scalarization, benchmark.reference_point)
    return raw_y.numpy(), values.numpy()


def scalarization_optima(case, benchmark, front_tensor):
    """Continuous ZDT optima; a dense shared-front fallback for other problems."""
    w = np.asarray(case.lambda_vector, dtype=float)
    if benchmark.name in ('zdt1', 'zdt2') and (w > 0).all():
        def h(x):
            return 1-np.sqrt(x) if benchmark.name == 'zdt1' else 1-x*x
        if case.scalarization == 'linear':
            if benchmark.name == 'zdt1':
                xs = [min(1., (w[1]/(2*w[0]))**2)]
            else:
                xs = [0., 1.] if np.isclose(w[0], w[1], rtol=0, atol=1e-12) else [0. if w[1] < w[0] else 1.]
        else:
            ref = benchmark.reference_point.numpy()
            lo, hi = 0., 1.
            for _ in range(60):
                x = (lo+hi)/2
                if (-x-ref[0])/w[0] > (-h(x)-ref[1])/w[1]:
                    lo = x
                else:
                    hi = x
            xs = [(lo+hi)/2]
        return np.column_stack([xs, h(np.asarray(xs))])
    values = scalarized_trajectory_values(-front_tensor, torch.tensor(case.lambda_vector),
                                          case.scalarization, benchmark.reference_point).numpy()
    return front_tensor.numpy()[np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))]


def _draw(ax, y, values, front, optimum, *, incumbent, norm):
    indices = incumbent_indices(values) if incumbent else np.arange(len(y))
    points = y[indices]
    ax.plot(front[:, 0], front[:, 1], 'k-', lw=1, label="true Pareto front")
    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lines = LineCollection(segments, cmap='viridis', norm=norm, alpha=.6, linewidths=1)
        lines.set_array(indices[1:]+1)
        ax.add_collection(lines)
        # A few arrows keep long trajectories readable.
        for j in np.unique(np.linspace(0, len(points)-2, min(6, len(points)-1)).astype(int)):
            ax.annotate('', xy=points[j+1], xytext=points[j],
                        arrowprops=dict(arrowstyle='->', color='0.35', alpha=.6, lw=.8))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=indices+1,
                         cmap='viridis', norm=norm, s=20, zorder=3)
    best = y[int(np.argmax(values))]
    ax.scatter(*best, marker='*', s=130, c='red', edgecolors='k', label="final best", zorder=5)
    ax.scatter(optimum[:, 0], optimum[:, 1], marker='X', s=65, c='gold',
               edgecolors='k', label="reference optimum", zorder=4)
    ax.set_xlabel('$f_1$ (minimize)'); ax.set_ylabel('$f_2$ (minimize)')
    return scatter


def fixed_trajectory_figures(run_dir, fig_dir):
    paths = sorted((run_dir/'raw').glob('*/seed_*.json'))
    if not paths:
        return
    representative = FixedSeedData.from_dict(load_json(paths[0]))
    _verify_fixed(representative)
    benchmark = _build_benchmark(representative.metadata)
    front_tensor = benchmark.true_pareto_front_raw(100001)
    front = front_tensor.numpy()
    seed = representative.metadata['seed']
    data = {}
    optima = {}
    for case in representative.cases:
        key = (case.scalarization, case.lambda_angle_degrees)
        data[key] = {m: case_queries(case, m, benchmark) for m in ('right', 'wrong')}
        optima[key] = scalarization_optima(case, benchmark, front_tensor)
        combined = np.vstack([front[::100], *[y for y, _ in data[key].values()]])
        lower, upper = combined.min(0), combined.max(0)
        margin = np.maximum(upper-lower, .1)*.04
        norm = Normalize(1, max(len(y) for y, _ in data[key].values()))
        for incumbent, suffix in ((False, 'queries'), (True, 'incumbents')):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
            for ax, method in zip(axes, ('right', 'wrong')):
                y, values = data[key][method]
                scatter = _draw(ax, y, values, front[::100], optima[key], incumbent=incumbent, norm=norm)
                ax.set_title(plotting.style_for(method)['label']); ax.legend(fontsize=7)
                ax.set_xlim(lower[0]-margin[0], upper[0]+margin[0])
                ax.set_ylim(lower[1]-margin[1], upper[1]+margin[1])
            fig.colorbar(scatter, ax=axes, label='evaluation (initial design included)')
            fig.suptitle(f'{key[0]}, angle {key[1]:g}°, seed {seed} — {suffix}')
            plotting.save_fig(fig, fig_dir, f'fixed_{key[0]}_angle{key[1]:g}_{suffix}_seed{seed:03d}')

    for scal in sorted({key[0] for key in data}):
        keys = sorted(k for k in data if k[0] == scal)
        chosen = [keys[i] for i in sorted(set((0, len(keys)//2, len(keys)-1)))]
        fig, axes = plt.subplots(2, len(chosen), figsize=(5*len(chosen), 8), squeeze=False)
        combined = np.vstack([front[::100], *[data[k][m][0] for k in chosen for m in ('right','wrong')]])
        lower, upper = combined.min(0), combined.max(0)
        margin = np.maximum(upper-lower, .1)*.04
        norm = Normalize(1, len(data[chosen[0]]['right'][0]))
        for col, key in enumerate(chosen):
            for row, method in enumerate(('right', 'wrong')):
                ax = axes[row, col]
                scatter = _draw(ax, *data[key][method], front[::100], optima[key], incumbent=False, norm=norm)
                ax.set_title(f'{method.upper()}, {key[1]:g}°')
                ax.set_xlim(lower[0]-margin[0], upper[0]+margin[0])
                ax.set_ylim(lower[1]-margin[1], upper[1]+margin[1])
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label='evaluation')
        plotting.save_fig(fig, fig_dir, f'fixed_{scal}_trajectory_summary_seed{seed:03d}')

        # Reduced-opacity seed map only draws final incumbents, not every query.
        for all_seeds in (False, True):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(front[:, 0], front[:, 1], 'k-', lw=1, label='true Pareto front')
            angle_norm = Normalize(min(k[1] for k in keys), max(k[1] for k in keys))
            for key in keys:
                ax.scatter(optima[key][:, 0], optima[key][:, 1], c=np.full(len(optima[key]), key[1]),
                           cmap='plasma', norm=angle_norm, marker='X', s=70, edgecolors='k')
            for path in (paths if all_seeds else paths[:1]):
                raw = FixedSeedData.from_dict(load_json(path)); _verify_fixed(raw)
                for case in raw.cases:
                    if case.scalarization != scal:
                        continue
                    for method, marker in (('right','o'), ('wrong','^')):
                        y, values = case_queries(case, method, benchmark)
                        point = y[np.argmax(values)]
                        scatter = ax.scatter(*point, c=[case.lambda_angle_degrees], cmap='plasma',
                                             norm=angle_norm, marker=marker, s=40,
                                             alpha=.25 if all_seeds else .9)
            for marker, label in [('o','RIGHT'), ('^','WRONG'), ('X','reference optimum')]:
                ax.scatter([], [], c='0.4', marker=marker, label=label)
            ax.legend(); ax.set_xlabel('$f_1$ (minimize)'); ax.set_ylabel('$f_2$ (minimize)')
            ax.set_title(f'{scal} — final preference map' + (' (all seeds)' if all_seeds else f' (seed {seed})'))
            if benchmark.name == 'zdt2' and scal == 'linear':
                ax.text(.02, .02, 'Linear scalarization cannot select the interior of the ZDT2 front.',
                        transform=ax.transAxes, fontsize=7)
            fig.colorbar(scatter, ax=ax, label='preference angle (degrees)')
            suffix = 'all_seeds' if all_seeds else f'seed{seed:03d}'
            plotting.save_fig(fig, fig_dir, f'fixed_{scal}_preference_map_{suffix}')
