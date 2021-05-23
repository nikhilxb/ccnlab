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
import ccnlab.baselines.basic as models

exp = classical.registry('*acquisition*')[0]
model = models.RandomModel()
g, group = list(exp.stimuli.items())[0]
i, trial = list(enumerate(group))[0]
t, timestep = list(enumerate(trial))[0]
cs, ctx, us = timestep
res = model.act(cs, ctx, us, t)

rw = models.RW(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = rw.act(cs, ctx, us, t)

kalman = models.Kalman(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = kalman.act(cs, ctx, us, t)


td = models.TD(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space), num_timesteps=len(trial))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = td.act(cs, ctx, us, t)

embed()
    
for exp in classical.registry('*acquisition*'):
    display_heading(exp.name, level=2)
    print(classical.repr_spec(exp.spec))
    
    for g, group in exp.stimuli.items():
        for subject in range(1):
            model = models.RandomModel()
            for i, trial in enumerate(group):
                for t, timestep in enumerate(trial):
                    cs, ctx, us = timestep
                    res = model.act(cs, ctx, us, t)
                    exp.data[g][i][t]['response'].append(res)

    results = exp.results
    summary = exp.summarize()
    print('correlation:', evaluation.correlation(results, summary))
    display_side_by_side(results, summary)
    exp.plot(show='both')
    plt.show()
