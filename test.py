from IPython import embed
from abc import ABC
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

class Model(ABC):

    def reset(self):
        ''' Erase all memory from the model '''

    def act(self, cs, ctx, us, t):
        ''' Perform an action given current stimulus input 
        :param cs: conditioned stimulus
        :param ctx: context
        :param us: unconditioned stimulus
        :param t: timestep within trial
        :return: (un)conditioned response
        '''

class RW(Model):
    def __init__(self, cs_dim, ctx_dim, alpha=0.3):
        self.alpha = alpha  # learning rate
        self.cs_dim = cs_dim
        self.ctx_dim = ctx_dim
        self.D = self.cs_dim + self.ctx_dim  # stimulus dimensions: concatenate punctate and contextual cues
        self.reset()

    def reset(self):
        self.w = np.zeros((self.D,))

    def act(self, cs, ctx, us, t):
        x=np.array(cs + ctx)
        self._update(x=x, r=us)
        return self.w.dot(x)  # CR = value

    def _update(self, x, r):
        rpe = r - self.w.dot(x)  # reward prediction error
        self.w = self.w + self.alpha * rpe * x  # weight update


class TD(Model):
    def __init__(self, cs_dim, ctx_dim, num_timesteps, alpha=0.3, gamma=0.98):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.cs_dim = cs_dim
        self.ctx_dim = ctx_dim
        self.T = num_timesteps
        self.D = self.cs_dim + self.ctx_dim  # stimulus dimensions: concatenate punctate and contextual cues
        self.reset()

    def reset(self):
        self.w = np.zeros((self.D * self.T,))
        self.last_x = np.zeros((self.D * self.T,))  # previous input
        self.last_r = 0  # previous reward

    def act(self, cs, ctx, us, t):
        if t == 0:
            self.last_x = np.zeros((self.D * self.T,))  # no previous input at initial timestep
        x = np.zeros((self.D * self.T,))
        x[t * self.D : (t + 1) * self.D] = cs + ctx  # complete serial compound representation
        self._update(x=x, r=us)
        if t == self.T:
            self._update(x=np.zeros((self.D * self.T,)), r=0)  # perform update with the last seen input
        return self.w.dot(x)  # CR = value

    def _update(self, x, r):
        # notice that we have to update for the previous input, because we don't have access to the next input
        last_rpe = self.last_r + self.gamma * self.w.dot(x) - self.w.dot(self.last_x)   # reward prediction error
        self.w = self.w + self.alpha * last_rpe * self.last_x  # weight update
        self.last_x = x
        self.last_r = r


class Kalman(Model):
    def __init__(self, cs_dim, ctx_dim, tau2=0.01, sigma_r2=1, sigma_w2=1):
        self.cs_dim = cs_dim
        self.ctx_dim = ctx_dim
        self.D = self.cs_dim + self.ctx_dim  # stimulus dimensions: concatenate punctate and contextual cues
        self.tau2 = tau2  # diffusion/transition variance
        self.sigma_r2 = sigma_r2  # noise variance
        self.sigma_w2 = sigma_w2  # prior variance
        self.Q = self.tau2 * np.identity(self.D)  # transition covariance
        self.reset()

    def reset(self):
        self.w = np.zeros((self.D,))  # mean weights
        self.S = self.sigma_w2 * np.identity(self.D)  # weight covariance

    def act(self, cs, ctx, us, t):
        x=np.array(cs + ctx)
        self._update(x=x, r=us)
        return self.w.dot(x)  # CR = value

    def _update(self, x, r):
        rpe = r - self.w.dot(x)  # reward prediction error
        S = self.S + self.Q  # prior covariance
        R = x.dot(S).dot(x) + self.sigma_r2  # residual covariance
        k = S.dot(x) / R  # Kalman gain
        self.w = self.w + k * rpe  # weight update
        self.S = S - k.dot(x) * S  # posterior covariance

class RandomModel(Model):
    """Produces response with probability that changes linearly with each US."""
    def __init__(self, start=0.2, delta=0.1, min_prob=0.1, max_prob=0.9):
        self.prob = start
        self.start = start
        self.delta = delta
        self.min_prob = min_prob
        self.max_prob = max_prob
        
    def reset(self):
        self.prob = start
    
    def act(self, cs, ctx, us, t):
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
res = model.act(cs, ctx, us, t)

rw = RW(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = rw.act(cs, ctx, us, t)

kalman = Kalman(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = kalman.act(cs, ctx, us, t)


td = TD(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space), num_timesteps=len(trial))
cs, ctx, us = exp.stimulus(g,i,t, vector=True)
res = td.act(cs, ctx, us, t)

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
                    res = model.act(cs, ctx, us, t)
                    exp.data[g][i][t]['response'].append(res)

    results = exp.results
    summary = exp.summarize()
    print('correlation:', evaluation.correlation(results, summary))
    display_side_by_side(results, summary)
    exp.plot(show='both')
    plt.show()
