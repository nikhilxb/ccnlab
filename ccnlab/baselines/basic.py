import numpy as np
import random
from ccnlab.baselines.core import Model, BinaryResponseModel



class RW(BinaryResponseModel):
    def __init__(self, cs_dim, ctx_dim, alpha=0.3):
        super().__init__()
        self.alpha = alpha  # learning rate
        self.cs_dim = cs_dim
        self.ctx_dim = ctx_dim
        self.D = self.cs_dim + self.ctx_dim  # stimulus dimensions: concatenate punctate and contextual cues
        self.reset()

    def reset(self):
        self.w = np.zeros((self.D,))

    def _value(self, cs, ctx, us, t):
        x = np.array(cs + ctx)
        v = self.w.dot(x) # value before update
        self._update(x=x, r=us)
        #print(x, self.w, self.w.dot(x))
        return v

    def _update(self, x, r):
        rpe = r - self.w.dot(x)  # reward prediction error
        self.w = self.w + self.alpha * rpe * x  # weight update


class TD(BinaryResponseModel):
    def __init__(self, cs_dim, ctx_dim, num_timesteps, alpha=0.3, gamma=0.98):
        super().__init__()
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

    def _value(self, cs, ctx, us, t):
        if t == 0:
            self.last_x = np.zeros((self.D * self.T,))  # no previous input at initial timestep
        x = np.zeros((self.D * self.T,))
        x[t * self.D : (t + 1) * self.D] = cs + ctx  # complete serial compound representation
        v = self.w.dot(x) # value before update
        self._update(x=x, r=us)
        if t + 1 == self.T:
            self._update(x=np.zeros((self.D * self.T,)), r=0)  # perform update with the last seen input
        return v

    def _update(self, x, r):
        # notice that we have to update for the previous input, because we don't have access to the next input
        last_rpe = self.last_r + self.gamma * self.w.dot(x) - self.w.dot(self.last_x)   # reward prediction error
        self.w = self.w + self.alpha * last_rpe * self.last_x  # weight update
        self.last_x = x
        self.last_r = r


class Kalman(BinaryResponseModel):
    def __init__(self, cs_dim, ctx_dim, tau2=0.01, sigma_r2=1, sigma_w2=1):
        super().__init__()
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

    def _value(self, cs, ctx, us, t):
        x = np.array(cs + ctx)
        v = self.w.dot(x) # value before update
        self._update(x=x, r=us)
        return v  # CR = value

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
