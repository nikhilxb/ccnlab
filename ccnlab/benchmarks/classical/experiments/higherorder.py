import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class HigherOrder_SensoryPreconditioning(cc.ClassicalConditioningExperiment):
  """When BA- pairings are followed by A+ pairings, presentation of B may generate a response.

  Source: 11.1 - Figure 57
  """
  def __init__(self, n_nonreinforced=100, n_reinforced=20, n_test=1):
    super().__init__({
      'control':
        cc.seq(

        ),
      'sensory preconditioning':
        cc.seq(

        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response',
      citation='Brogden (1939)',
    )
    self.results = pd.melt(
      pd.DataFrame(
        columns=['group', 'B'],
        data=[
          
        ]
      ),
      id_vars=['group']
    )
    self.plots = [

    ]

  def summarize(self):
    return
