import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from IPython import embed
from IPython.display import set_matplotlib_formats

import ccnlab.benchmarks.classical as classical
import ccnlab.evaluation as evaluation
from ccnlab.baselines.basic import RescorlaWagner, KalmanFilter, TemporalDifference

GENERATE_PLOTS = True
GENERATE_TABLE = True

set_matplotlib_formats('retina')
sns.set(rc={"figure.figsize": (6, 4)})
sns.set_style('ticks')
sns.set_context('notebook', rc={"lines.linewidth": 2.5})
random.seed(0)

# exps = classical.registry('*')

EXP_NAMES = [
  'Acquisition_ContinuousVsPartial', 'Extinction_ContinuousVsPartial',
  'Generalization_NovelVsInhibitor', 'Generalization_AddVsRemove',
  'HigherOrder_SensoryPreconditioning', 'Competition_OvershadowingAndForwardBlocking',
  'Recovery_Overshadowing'
]
exps = classical.registry(*EXP_NAMES)
exp_names = [name.split('_')[0] for name in EXP_NAMES]

embed()

model_names = ['Rescorla-Wagner', 'Kalman Filter', 'Temporal Difference']
figsize = (2.3, 3.3)
fig, axes = plt.subplots(
  len(exps),
  1 + len(model_names),
  figsize=(figsize[0] * len(exps), figsize[1] * (1 + len(model_names)))
)

scores = np.zeros((len(exps), len(model_names)))
is_ratio = np.zeros((len(exps), len(model_names)))
for e, exp in enumerate(exps):
  print(exp.name)

  model_classes = [
    lambda: RescorlaWagner(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space)), lambda:
    KalmanFilter(cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space)), lambda: TemporalDifference(
      cs_dim=len(exp.cs_space), ctx_dim=len(exp.ctx_space), num_timesteps=len(trial)
    )
  ]

  dfs = [exp.empirical_results]
  for m, model_class in enumerate(model_classes):
    exp.reset()
    for g, group in exp.stimuli.items():
      for subject in range(20):
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
      embed()
      pass

  embed()
  exp.multiplot(
    axes[e],
    dfs,
    names=['Empirical Data'] + model_names,
    is_empirical=[True] + [False] * len(model_names),
    show_titles=(exp == exps[0]),
    exp_name=exp_names[e]
  )

fig.tight_layout(pad=1.0)
if GENERATE_PLOTS:
  plt.show()

if GENERATE_TABLE:
  last_category = ''
  for e, exp in enumerate(exps):
    columns = ['{:.2f}'.format(c) for c in scores[e]]
    #for m in [np.argmax(scores[e])]:
    for m in range(len(scores[e])):
      if scores[e, m] > 0.8:
        columns[m] = '\\textbf{' + columns[m] + '}'

    category, name = exp.name.split('_')
    if category != last_category:
      prefix = category + ' & \\texttt{' + name + '}'
    else:
      prefix = ' & \\texttt{' + name + '}'
    last_category = category
    if is_ratio[e, 0]:
      assert (all(is_ratio[e]))
      prefix += '*'
    print('{} & {} \\\\'.format(prefix, ' & '.join(columns)))
