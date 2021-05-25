from IPython import embed
from IPython.display import display, set_matplotlib_formats, HTML
from abc import ABC
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
set_matplotlib_formats('retina')
sns.set(rc={"figure.figsize": (6, 4)})
sns.set_style('ticks')
sns.set_context('notebook', rc={"lines.linewidth": 2.5})



def display_heading(text, level=1):
    display(HTML('<h{}>{}</h{}>'.format(level, text, level)))

def display_side_by_side(*dfs):
    display(HTML(
        '<div style="display: flex; flex-direction: row; gap: 12px;">' + 
        ''.join(['<div>' + df._repr_html_() + '</div>' for df in dfs]) +
        '</div>'
    ))

import random
import numpy as np


import ccnlab.benchmarks.classical as classical
import ccnlab.evaluation as evaluation
from ccnlab.baselines.basic import RandomModel, RW, Kalman, TD 

random.seed(0)

#exps = classical.registry('*')
#exps = classical.registry('*Competition*')[:6]
exps = classical.registry('*ContinuousVsPartial*') + classical.registry('*Generalization*') + [classical.registry('*HigherOrder*')[0]] + classical.registry('*Overshadowing*')
#embed()
model_names = ['Rescorla-Wagner', 'Kalman filtering', 'Temporal difference\nlearning']
figsize=(3, 3)
fig, axes = plt.subplots(len(exps), 1+len(model_names), figsize=(figsize[0] * len(exps), figsize[1] * (1+len(model_names))))

scores = np.zeros((len(exps), len(model_names)))
is_ratio = np.zeros((len(exps), len(model_names)))
for e, exp in enumerate(exps):
    display_heading(exp.name, level=2)
    print(classical.repr_spec(exp.spec))

    model_classes = [
            lambda: RW(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space)),
            lambda: Kalman(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space)),
            lambda: TD(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space), num_timesteps=len(trial))
    ]

    dfs = [exp.results]
    for m, model_class in enumerate(model_classes):
        exp.reset()
        for g, group in exp.stimuli.items():
            for subject in range(20):
                model = model_class()
                for i, trial in enumerate(group):
                    for t, timestep in enumerate(trial):
                        cs, ctx, us = exp.stimulus(g,i,t, vector=True)
                        res = model.act(cs, ctx, us, t)
                        exp.data[g][i][t]['response'].append(res)
        summary = exp.summarize()
        dfs.append(summary)
        if len(list(exp.results.value)) == 2:
            scores[e,m] = evaluation.ratio_of_ratios(exp.results, summary)
            is_ratio[e,m] = 1
        else:
            scores[e,m] = evaluation.correlation(exp.results, summary)

    #embed()
    exp.multiplot(axes[e], dfs, names=['Empirical data'] + model_names, is_empirical=[True] + [False] * len(model_names), show_titles=(exp == exps[0]))

fig.tight_layout(pad=0.0)
plt.show()

for e, exp in enumerate(exps):
    columns = ['{:.2f}'.format(c) for c in scores[e]]
    for i in [np.argmax(scores[e])]:
        columns[i] = '\\textbf{' + columns[i] + '}'
    name =  exp.name.replace('_', ': ')
    if is_ratio[e,0]:
        assert(all(is_ratio[e]))
        name = name + '*'
    name = '\\makecell[tl]{'+ name + '}'
    print(name + ' & ' + ' & '.join(columns) + '\\\\')
