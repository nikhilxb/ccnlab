import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Discrimination_ReinforcedVsNonreinforced(cc.ClassicalConditioningExperiment):
  """A reinforced CS elicits significantly greater CR than a non-reinforced CS.

  Source: 4.1 - Figure 10
  """
  def __init__(self, n=250):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A+'), name='train-A'),
          cc.seq(cc.trial('B-'), name='train-B'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Campolattaro et al. (2008)',
    )
    self.results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'B'],
        data=[
          ['main', 1, 20, 31],
          ['main', 2, 19, 39],
          ['main', 3, 20, 48],
          ['main', 4, 21, 65],
          ['main', 5, 12, 78],
          ['main', 6, 13, 80],
          ['main', 7, 12, 81],
          ['main', 8, 13, 86],
          ['main', 9, 13, 81],
          ['main', 10, 13, 85],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def summarize(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['response'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['response'], ['B']),
          } if x['phase'] == 'train-B' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      25,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_PositivePatterning(cc.ClassicalConditioningExperiment):
  """Reinforced AB+ intermixed with non-reinforced A- and B- results in responding to AB that is
  stronger than the sum of the individual responses to A and B.

  Source: 4.2 - Figure 11
  """
  def __init__(self, n=480):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A-'), name='train-A'),
          cc.seq(cc.trial('B-'), name='train-B'),
          cc.seq(cc.trial('AB+'), name='train-AB'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Bellingham et al. (1985)',
    )
    self.results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'B', 'AB'],
        data=[
          ['main', 1, 11, 23, 21],
          ['main', 2, 41, 51, 53],
          ['main', 3, 61, 73, 98],
          ['main', 4, 54, 56, 93],
          ['main', 5, 63, 61, 93],
          ['main', 6, 46, 48, 93],
          ['main', 7, 36, 41, 86],
          ['main', 8, 38, 30, 85],
          ['main', 9, 28, 30, 85],
          ['main', 10, 31, 33, 83],
          ['main', 11, 31, 23, 84],
          ['main', 12, 31, 29, 85],
          ['main', 13, 28, 21, 88],
          ['main', 14, 24, 21, 88],
          ['main', 15, 24, 13, 80],
          ['main', 16, 29, 17, 88],
          ['main', 17, 24, 16, 80],
          ['main', 18, 21, 18, 80],
          ['main', 19, 26, 21, 84],
          ['main', 20, 26, 26, 87],
          ['main', 21, 24, 16, 84],
          ['main', 22, 25, 24, 91],
          ['main', 23, 16, 21, 86],
          ['main', 24, 21, 24, 83],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def summarize(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['response'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['response'], ['B']),
          } if x['phase'] == 'train-B' else {
            'AB': cc.conditioned_response(x['timesteps'], x['response'], ['A', 'B']),
          } if x['phase'] == 'train-AB' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      20,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_NegativePatterning(cc.ClassicalConditioningExperiment):
  """Non-reinforced AB- intermixed with reinforced A+ and B+ results in responding to AB that is
  weaker than the sum of the individual responses to A and B.

  Source: 4.3 - Figure 12
  """
  def __init__(self, n=480):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A+'), name='train-A'),
          cc.seq(cc.trial('B+'), name='train-B'),
          cc.seq(cc.trial('AB-'), name='train-AB'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Bellingham et al. (1985)',
    )
    self.results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'B', 'AB'],
        data=[
          ['main', 1, 38, 20, 26],
          ['main', 2, 51, 40, 81],
          ['main', 3, 78, 73, 93],
          ['main', 4, 81, 80, 93],
          ['main', 5, 89, 85, 93],
          ['main', 6, 88, 85, 83],
          ['main', 7, 88, 92, 73],
          ['main', 8, 78, 83, 66],
          ['main', 9, 88, 88, 58],
          ['main', 10, 91, 83, 51],
          ['main', 11, 85, 83, 53],
          ['main', 12, 80, 91, 48],
          ['main', 13, 96, 93, 48],
          ['main', 14, 96, 88, 36],
          ['main', 15, 96, 88, 36],
          ['main', 16, 88, 83, 36],
          ['main', 17, 86, 88, 27],
          ['main', 18, 80, 92, 23],
          ['main', 19, 88, 88, 28],
          ['main', 20, 88, 88, 26],
          ['main', 21, 80, 88, 28],
          ['main', 22, 84, 85, 26],
          ['main', 23, 80, 88, 31],
          ['main', 24, 73, 86, 26],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def summarize(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['response'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['response'], ['B']),
          } if x['phase'] == 'train-B' else {
            'AB': cc.conditioned_response(x['timesteps'], x['response'], ['A', 'B']),
          } if x['phase'] == 'train-AB' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      20,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_Biconditional(cc.ClassicalConditioningExperiment):
  """Biconditional discrimination between compounds (AC+/BD+ vs. AD-/BC-, where no single CS
  predicts reinforcement or non-reinforcement) is possible but harder than component discrimination
  between compounds (AC+/AD+ vs. BC-/BD-, where A and B predict reinforcment and non-reinforcement,
  respectively).

  Source: 4.7, 4.9 - Figure 16
  """
  def __init__(self, n_componennt=50, n_biconditional=100):
    super().__init__({
      'component':
        cc.seq(
          cc.seq(cc.trial('AC+'), name='train-AC'),
          cc.seq(cc.trial('AD+'), name='train-AD'),
          cc.seq(cc.trial('BC-'), name='train-BC'),
          cc.seq(cc.trial('BD-'), name='train-BD'),
          repeat=n_componennt,
        ),
      'biconditional':
        cc.seq(
          cc.seq(cc.trial('AC+'), name='train-AC'),
          cc.seq(cc.trial('AD-'), name='train-AD'),
          cc.seq(cc.trial('BC-'), name='train-BC'),
          cc.seq(cc.trial('BD+'), name='train-BD'),
          repeat=n_biconditional,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Saavedra (1975)',
    )
    self.results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', '+', '\u2212'],  # Use "minus" character for prettier plot.
        data=[
          ['component', 1, 34, 30],
          ['component', 2, 70, 39],
          ['component', 3, 90, 42],
          ['component', 4, 91, 33],
          ['component', 5, 91, 34],
          ['biconditional', 1, 48, 37],
          ['biconditional', 2, 65, 64],
          ['biconditional', 3, 85, 64],
          ['biconditional', 4, 88, 72],
          ['biconditional', 5, 86, 62],
          ['biconditional', 6, 94, 63],
          ['biconditional', 7, 91, 53],
          ['biconditional', 8, 88, 47],
          ['biconditional', 9, 94, 62],
          ['biconditional', 10, 93, 50],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def summarize(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            '+': cc.conditioned_response(x['timesteps'], x['response'], ['A', 'C']),
          } if x['phase'] == 'train-AC' else {
            ('+' if x['group'] == 'component' else '\u2212'):
              cc.conditioned_response(x['timesteps'], x['response'], ['A', 'D']),
          } if x['phase'] == 'train-AD' else {
            '\u2212': cc.conditioned_response(x['timesteps'], x['response'], ['B', 'C']),
          } if x['phase'] == 'train-BC' else {
            ('\u2212' if x['group'] == 'component' else '+'):
              cc.conditioned_response(x['timesteps'], x['response'], ['B', 'D']),
          } if x['phase'] == 'train-BD' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      10,
      trial_name='trial in phase'
    )
