from copy import copy
import numpy as np
import matplotlib.pyplot as plt

class ActionSpace:
  """The action space: [-1, 0, 1] (SELL, HOLD, BUY)"""

  def __init__(self):
    self.n = 3
    self.actions = [-1, 0, 1]

class State:
  """The state: Current prices and if has asset."""
  def __init__(self):
    self.prices = []
    self.asset = -1

  def as_array(self):
    return list(self.prices - 100) + [self.asset]


class Env:
  """The environment of my gym."""

  def __init__(self, n, testdata=True, N=200):
    self.N = N # Total number of steps
    self.n = n # Prices taken into account

    # compute prices
    x = np.linspace(- self.n / self.N, 1, self.N+self.n) # we compute n values additionally (before step 0), i.e. price starts indexing with self.n
    self.periods = 5
    self.prices = np.sin(self.periods * np.pi * x) + 100

    self.fig, (self.ax1, self.ax2) = plt.subplots(2)
    self.ax1.plot(range(self.N), self.prices[self.n:self.N+self.n])
    self.ax1.set_xlim([0, self.N])

    self.solvedReward = 0.9 * self.periods * (1 - 0.01)
    self.reset()

  def reset(self):
    self.step_ = 0
    self.state = State()
    self.state.prices = self.prices[self.step_:self.step_+self.n]
    self.cumulativereward = [0]
    self.actionhistory = [self.state.asset]
    return self.state.as_array()

  def step(self, action):
    action -= 1 # we get [0, 1, 2]

    self.step_ += 1

    # if we hold asset, compute reward
    if self.state.asset == 1:
      reward = (self.prices[self.step_+self.n] - self.prices[self.step_+self.n-1])
    else:
      reward = 0

    if action != 0:
      reward -= 0.01
      self.state.asset = action

    self.state.prices = self.prices[self.step_:self.step_+self.n]
    self.cumulativereward += [self.cumulativereward[-1] + reward]
    self.actionhistory += [action]

    done = (self.step_ == self.N-1)
    info = None
    return self.state.as_array(), reward, done, info

  def render(self, mode='pause'):
    self.ax2.clear()
    self.ax2.set_xlim([0, self.N])
    self.ax2.set_ylim([-self.periods, self.periods])

    x = np.arange(0, len(self.cumulativereward), 1)
    self.ax2.plot(x, self.cumulativereward, 'k--')

    a = np.array(self.actionhistory)
    sell = x[a == -1]
    buy  = x[a == 1]
    self.ax2.plot(sell, [0]*len(sell), 'r.')
    self.ax2.plot(buy,  [0]*len(buy),  'g.')

    if mode == 'pause':
      plt.pause(0.00001)
    elif mode == 'show':
      plt.show()
    else:
      raise

    screen = None
    return screen
