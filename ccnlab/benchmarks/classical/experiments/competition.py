import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Competition_RelativeValidity(cc.ClassicalConditioningExperiment):
  """Conditioning to X is weaker when training consists of pairing X with stimuli A/B that are
  correlated with reinforcement, than when training consists of pairing X with stimuli A/B that
  are not correlated. 

  Source: 7.1 - Figure 29
  """
  def __init__(self, n_correlated=200, n_uncorrelated=100, n_test=10):
    super().__init__({
      'correlated':
        cc.seq(
          cc.seq(
            cc.trial('XA+'),
            cc.trial('XB-'),
            repeat=n_correlated,
            name='train',
          ), cc.seq(
            cc.trial('X'),
            repeat=n_test,
            name='test',
          )
        ),
      'uncorrelated':
        cc.seq(
          cc.seq(
            cc.trial('XA+'),
            cc.trial('XA-'),
            cc.trial('XB+'),
            cc.trial('XB-'),
            repeat=n_uncorrelated,
            name='train',
          ), cc.seq(
            cc.trial('X'),
            repeat=n_test,
            name='test',
          )
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Wagner et al. (1968)',
    )
    self.results = pd.melt(
      pd.DataFrame(columns=['group', 'X'], data=[
        ['correlated', 20],
        ['uncorrelated', 80],
      ]),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'])
    ]

  def summarize(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          'X': cc.conditioned_response(x['timesteps'], x['response'], ['X']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_OvershadowingAndForwardBlocking(cc.ClassicalConditioningExperiment):
  """Training AB+ results in weaker conditioning to A than training A+ alone (overshadowing).
  Training B+ -> AB+ results in even weaker conditioning to A (forward blocking).

  Source: 7.2, 7.5 - Figure 30
  """
  def __init__(self, n_train=20, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'overshadowing':
        cc.seq(
          cc.seq(cc.trial('C+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'forward blocking':
        cc.seq(
          cc.seq(cc.trial('B+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='elevation scores',
      citation='Holland and Fox (2003)',
    )
    self.results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['control', 65],
          ['overshadowing', 40],
          ['forward blocking', 12],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'])
    ]

  def summarize(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          'A': cc.conditioned_response(x['timesteps'], x['response'], ['A']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_BackwardBlocking(cc.ClassicalConditioningExperiment):
  """Training AB+ -> B+ results in weaker conditioning to A than training A+ alone (backward
  blocking).

  Source: 7.7 - Figure 33
  """
  def __init__(self, n_train=20, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('C+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'backward blocking':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('B+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='mean latency [log s]',
      citation='Miller and Matute (1996)',
    )
    self.results = pd.melt(
      pd.DataFrame(columns=['group', 'A'], data=[
        ['control', 1.55],
        ['backward blocking', 1.05],
      ]),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'])
    ]

  def summarize(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          'A': cc.conditioned_response(x['timesteps'], x['response'], ['A']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()
