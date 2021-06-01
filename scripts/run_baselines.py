import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import os

import ccnlab.benchmarks.classical as classical
import ccnlab.evaluation as evaluation
from ccnlab.baselines.basic import RescorlaWagner, KalmanFilter, TemporalDifference

GENERATE_PLOTS = True
GENERATE_TABLE = True
NUM_SUBJECTS = 20

EXP_NAMES = ['*']
EXP_NAMES = [
  'Acquisition_ContinuousVsPartial',
  'Extinction_ContinuousVsPartial',
  'Generalization_NovelVsInhibitor',
  'Generalization_AddVsRemove',
  'Competition_OvershadowingAndForwardBlocking',
  'Recovery_Overshadowing',
  'HigherOrder_SensoryPreconditioning',
]
# EXP_NAMES = [
#   # 'Acquisition_ContinuousVsPartial',
#   'PreExposure_LatentInhibitionVsPerceptualLearning',
# ]
MODEL_NAMES = [
  'Rescorla-Wagner',
  'Kalman Filter',
  'Temporal Difference',
]

sns.set(rc={"figure.figsize": (6, 4)})
sns.set_style('ticks')
sns.set_context('notebook', rc={"lines.linewidth": 2.5})
random.seed(0)

exps = classical.registry(*EXP_NAMES)

PLOT_WIDTH, PLOT_HEIGHT = (4, 2)
fig, axes = plt.subplots(
  len(exps),
  1 + len(MODEL_NAMES),
  figsize=(PLOT_WIDTH * (1 + len(MODEL_NAMES)), PLOT_HEIGHT * len(exps))
)
scores = np.zeros((len(exps), len(MODEL_NAMES)))
is_ratio = np.zeros((len(exps), len(MODEL_NAMES)))

for e, exp in enumerate(exps):
  print(exp.name)

  MODEL_CLASSES = [
    lambda: RescorlaWagner(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space)),
    lambda: KalmanFilter(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space)),
    lambda: TemporalDifference(
      cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space), num_timesteps=len(trial)
    ),
  ]

  dfs = [exp.empirical_results]
  for m, model_class in enumerate(MODEL_CLASSES):
    exp.reset()
    for g, group in exp.stimuli.items():
      for subject in range(NUM_SUBJECTS):
        model = model_class()
        for i, trial in enumerate(group):
          for t, timestep in enumerate(trial):
            cs, ctx, us = exp.stimulus(g, i, t, vector=True)
            res = model.act(cs, ctx, us, t)
            exp.data[g][i][t]['response'].append(res)
    simulated = exp.simulated_results()
    dfs.append(simulated)
    if len(list(exp.empirical_results.value)) == 2:
      scores[e, m] = evaluation.ratio_of_ratios(exp.empirical_results, simulated)
      is_ratio[e, m] = 1
    else:
      scores[e, m] = evaluation.correlation(exp.empirical_results, simulated)
    if np.isnan(scores[e, m]):
      continue

  # Assemble grid of plots.
  if GENERATE_PLOTS:
    col_names = ['Empirical Data'] + MODEL_NAMES
    assert len(dfs) == len(col_names)
    for plotfn in exp.plots:
      for m, df in enumerate(dfs):
        kind = 'empirical' if m == 0 else 'simulation'
        xlab = exp.meta.get('xlabel', None)
        # Show name as y-label for first col.
        category, name = exp.name.split('_')
        ylab = '{}\n{}'.format(category, name[:23]) if m == 0 else ''
        plotfn(df, axes[e][m], xlabel=xlab, ylabel=ylab, kind=kind)
        # Show title for first row only.
        if e == 0: axes[e][m].set_title(col_names[m], y=1.0, pad=14, fontsize=20)
        axes[e][m].get_yaxis().set_ticks([])

if GENERATE_PLOTS:
  fig.tight_layout(pad=0)
  outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure-baselines.png')
  plt.savefig(outpath)
  # plt.show()

if GENERATE_TABLE:
  last_category = ''
  for e, exp in enumerate(exps):
    columns = ['{:.2f}'.format(c) for c in scores[e]]
    for m in range(len(scores[e])):
      SCORE_THRESHOLD = 0.8
      if scores[e, m] > SCORE_THRESHOLD:
        columns[m] = '\\textbf{{{}}}'.format(columns[m])

    category, name = exp.name.split('_')
    prefix = '{} & \\texttt{{{}}}'.format(
      category if category != last_category else ' ' * len(category),
      name,
    )
    last_category = category
    if is_ratio[e, 0]:
      assert (all(is_ratio[e]))
      prefix += '*'
    print('{} & {} \\\\'.format(prefix, ' & '.join(columns)))
