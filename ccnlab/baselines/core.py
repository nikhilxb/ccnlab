from abc import ABC, abstractmethod
import numpy as np
import math
import random


def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))


class Model(ABC):
  """Abstract class for all models."""
  @abstractmethod
  def reset(self):
    """Erase all memory from the model."""

  @abstractmethod
  def act(self, cs, ctx, us, t):
    """Perform an action given current stimulus input 
    :param cs: conditioned stimulus
    :param ctx: context
    :param us: unconditioned stimulus
    :param t: timestep within trial
    :return: (un)conditioned response
    """


class ValueBasedModel(Model):
  """Abstract class for models that calculate value through an RL-style framework."""
  def act(self, cs, ctx, us, t):
    v = self._value(cs, ctx, us, t)
    response = v
    # p = sigmoid(self.inverse_temperature * (v - self.offset))
    # p = v
    # p = min(1 - self.eps / 2., max(self.eps / 2., p))
    # response = float(random.random() < p)
    # ur = 1 if us else 0
    # response = ur if us else cr
    return response

  @abstractmethod
  def _value(self, cs, ctx, us, t):
    """Return value for current stimulus."""
