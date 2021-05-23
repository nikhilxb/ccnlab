from IPython import embed
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt


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

class RW:
    def __init__(self, cs_dim, ctx_dim, alpha=0.1):
        self.alpha = alpha
        self.cs_dim = cs_dim
        self.ctx_dim = ctx_dim
        self.reset()

    def reset(self):
        self.w = np.zeros((self.cs_dim + self.ctx_dim,))

    def act(self, cs, ctx, us):
        return self._update(x=np.array(cs + ctx), r=us)

    def _update(self, x, r):
        v = self.w.dot(x)  # value = reward prediction
        rpe = r - v  # reward prediction error
        self.w = self.w + self.alpha * rpe * x  # weight update
        return v  # CR = value


class Kalman:
    def __init__(self, cs_dim, ctx_dim, alpha=0.3, tau2=0.01, sigma_r2=1, sigma_w2=1):
        self.alpha = alpha
        self.cs_dim = cs_dim
        self.ctx_dim = ctx_dim
        self.tau2 = tau2  # diffusion/transition variance
        self.sigma_r2 = sigma_r2  # noise variance
        self.sigma_w2 = sigma_w2  # prior variance
        self.Q = self.tau2 * np.identity(self.cs_dim + self.ctx_dim)  # transition covariance
        self.reset()

    def reset(self):
        self.w = np.zeros((self.cs_dim + self.ctx_dim,))  # mean weights
        self.S = self.sigma_w2 * np.identity(self.cs_dim + self.ctx_dim)  # weight covariance

    def act(self, cs, ctx, us):
        return self._update(x=np.array(cs + ctx), r=us)

    def _update(self, x, r):
        v = self.w.dot(x)  # value = reward prediction
        rpe = r - v  # reward prediction error
        S = self.S + self.Q  # prior covariance
        R = x.dot(S).dot(x) + self.sigma_r2  # residual covariance
        k = S.dot(x) / R  # Kalman gain
        self.w = self.w + k * rpe  # weight update
        self.S = S - k.dot(x) * S  # posterior covariance
        return v  # CR = value

class RandomModel:
    """Produces response with probability that changes linearly with each US."""
    def __init__(self, start=0.2, delta=0.1, min_prob=0.1, max_prob=0.9):
        self.prob = start
        self.start = start
        self.delta = delta
        self.min_prob = min_prob
        self.max_prob = max_prob
        
    def reset(self):
        self.prob = start
    
    def act(self, cs, ctx, us):
        if us > 0:
            self.prob = max(min(self.prob + self.delta, self.max_prob), self.min_prob)
            return 1
        if len(cs) > 0:
            return random.choices([1, 0], weights=[self.prob, 1-self.prob])[0]
        return 0

import ccnlab.benchmarks.classical as classical
import ccnlab.evaluation as evaluation

exp = classical.registry('*acquisition*')[0]
model = RandomModel()
g, group = list(exp.stimuli.items())[0]
i, trial = list(enumerate(group))[0]
t, timestep = list(enumerate(trial))[0]
cs, ctx, us = timestep
res = model.act(cs, ctx, us)

rw = RW(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = rw.act(cs, ctx, us)

kalman = Kalman(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = kalman.act(cs, ctx, us)

embed()
    
for exp in classical.registry('*acquisition*'):
    display_heading(exp.name, level=2)
    print(classical.repr_spec(exp.spec))
    
    for g, group in exp.stimuli.items():
        for subject in range(1):
            model = RandomModel()
            for i, trial in enumerate(group):
                for t, timestep in enumerate(trial):
                    cs, ctx, us = timestep
                    res = model.act(cs, ctx, us)
                    exp.data[g][i][t]['response'].append(res)

    results = exp.results
    summary = exp.summarize()
    print('correlation:', evaluation.correlation(results, summary))
    display_side_by_side(results, summary)
    exp.plot(show='both')
    plt.show()
