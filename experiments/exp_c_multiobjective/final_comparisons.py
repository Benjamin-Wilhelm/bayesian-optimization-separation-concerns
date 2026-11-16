"""Final-budget distributions and paired seed-block bootstrap comparisons."""
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from common.statistics import paired_bootstrap_difference
from experiments.exp_c_multiobjective import plotting


def _distribution(values, fig_dir, name, title):
    methods = list(values)
    fig, ax = plt.subplots(figsize=(max(6, 2*len(methods)), 4.5))
    arrays = [np.asarray(values[m], dtype=float) for m in methods]
    ax.boxplot(arrays, labels=[plotting.style_for(m)['label'] for m in methods], showfliers=False)
    rng = np.random.default_rng(0)
    for i, (method, array) in enumerate(zip(methods, arrays), 1):
        ax.scatter(i+rng.uniform(-.12, .12, len(array)), array, s=14, alpha=.5,
                   color=plotting.style_for(method)['color'])
    ax.set_ylabel(title); ax.set_title(f'Final budget — {title}')
    plotting.save_fig(fig, fig_dir, name)


def final_comparisons(processed, fig_dir):
    """Match by explicit seed ID and case; never flatten seed-angle pairs."""
    seed_ids = [int(p['metadata']['seed']) for p in processed]
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError('Duplicate seed IDs in paired comparison')
    result = dict(difference_sign='RIGHT minus WRONG; negative favors RIGHT for regret',
                  n_bootstrap=5000, confidence=.95, comparisons=[])
    mode = processed[0]['mode']
    groups = defaultdict(dict)
    if mode == 'fixed_scalarization':
        expected = {(c['scalarization'], float(c['lambda']['angle_degrees'])) for c in processed[0]['cases']}
        for p in processed:
            cases = {(c['scalarization'], float(c['lambda']['angle_degrees'])): c for c in p['cases']}
            if set(cases) != expected:
                raise ValueError('Seed blocks must contain the same preference angles')
            for (scal, angle), case in cases.items():
                for metric in ('simple_regret', 'cumulative_regret'):
                    groups[(scal, angle, metric)][p['metadata']['seed']] = {
                        m: case['methods'][m][metric][-1] for m in ('right', 'wrong')}
        for (scal, angle, metric), seeds in sorted(groups.items()):
            values = {m: [seeds[s][m] for s in sorted(seeds)] for m in ('right','wrong')}
            name = f'fixed_{scal}_angle{angle:g}_final_{metric}'
            _distribution(values, fig_dir, name, f'{scal}, {angle:g}°: {metric}')
            result['comparisons'].append(dict(scalarization=scal, angle=angle, metric=metric,
                seed_ids=sorted(seeds), **paired_bootstrap_difference(values['right'], values['wrong'])))
        for scal in sorted({k[0] for k in groups}):
            angles = sorted({k[1] for k in groups if k[0] == scal})
            for metric in ('simple_regret','cumulative_regret'):
                blocks = {m: np.asarray([[groups[(scal,a,metric)][s][m] for a in angles]
                                          for s in sorted(seed_ids)]) for m in ('right','wrong')}
                _distribution({m:b.mean(1) for m,b in blocks.items()}, fig_dir,
                              f'fixed_{scal}_aggregate_final_{metric}', f'{scal}: mean {metric} over angles')
                result['comparisons'].append(dict(scalarization=scal, aggregate='mean over angles within seed',
                    angles=angles, metric=metric, seed_ids=sorted(seed_ids),
                    **paired_bootstrap_difference(blocks['right'], blocks['wrong'])))
    else:
        methods = sorted(processed[0]['methods'])
        if any(sorted(p['methods']) != methods for p in processed):
            raise ValueError('All seeds must contain the same methods')
        metrics = ('hypervolume_regret','empirical_bayes_regret','cumulative_dominance_regret',
                   'average_dominance_regret','igd')
        for metric in metrics:
            values = {}
            for m in methods:
                values[m] = []
                for p in sorted(processed, key=lambda p:int(p['metadata']['seed'])):
                    entry = p['methods'][m]
                    curve = entry.get(metric, entry.get('diagnostics', {}).get(metric))
                    if curve is None:
                        raise ValueError('New diagnostics missing; rerun process.py')
                    if len(curve) != int(p['metadata']['budget']):
                        raise ValueError('Unequal total evaluation budgets')
                    values[m].append(curve[-1])
            _distribution(values, fig_dir, f'pareto_final_{metric}', metric)
            for right in ('right_algorithm_1','right_algorithm_2'):
                if right in values and 'wrong_algorithm_2' in values:
                    result['comparisons'].append(dict(metric=metric, right=right, wrong='wrong_algorithm_2',
                        seed_ids=sorted(seed_ids), **paired_bootstrap_difference(values[right], values['wrong_algorithm_2'])))
    # CIs are separate from trajectory variability bands.
    for metric in sorted({c['metric'] for c in result['comparisons']}):
        comparisons = [c for c in result['comparisons'] if c['metric'] == metric]
        fig, ax = plt.subplots(figsize=(9, max(4, .35*len(comparisons))))
        labels = []
        for i, c in enumerate(comparisons):
            ax.plot([c['lower'], c['upper']], [i,i], color='#0072B2')
            ax.scatter(c['estimate'], i, color='#0072B2', s=18)
            labels.append(f"{c.get('scalarization', c.get('right',''))} {c.get('angle','aggregate')}")
        ax.axvline(0, color='0.5', ls='--')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Mean paired difference RIGHT − WRONG (95% bootstrap CI)')
        ax.set_title(f'Final {metric}: seed-block bootstrap (5,000 replications)')
        plotting.save_fig(fig, fig_dir, f'{mode}_paired_bootstrap_{metric}')
    return result
