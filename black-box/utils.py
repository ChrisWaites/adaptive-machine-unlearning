from scipy.spatial.distance import cdist
import jax.numpy as np
from jax import random, ops
import matplotlib.pyplot as plt

class LookupTable:
  def __init__(self, tol=0.0):
    self.tol = tol
    self.default = 0

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
    return self

  def predict(self, X):
    dist = cdist(X, self.X_train)
    closest_idx = np.argmin(dist, 1)
    closest_dist = np.min(dist, 1)
    preds = self.y_train[closest_idx]
    preds = ops.index_update(preds, ops.index[closest_dist > self.tol], self.default)
    return preds


class Ensemble:
  def __init__(self, ModelClass, num_classes=3):
    self.ModelClass = ModelClass
    self.models = []
    self.one_hot = np.eye(num_classes + 1)

  def fit(self, X_y_pieces):
    self.models = [self.ModelClass().fit(Xi, yi) for Xi, yi in X_y_pieces]
    return self

  def predict(self, X):
    votes = sum([self.one_hot[model.predict(X)] for model in self.models])
    return votes


def sample_preds(rng, votes, sample, eps):
  if sample:
    if eps: # exp. mech
      noisy_scores = nn.softmax((eps / 2) * votes)
      temp, rng = random.split(rng)
      return random.categorical(temp, noisy_scores)
    else:
      return random.categorical(temp, votes / num_shards)
  else:
    if eps: # lnmax
      temp, rng = random.split(rng)
      noisy_scores = votes + (1 / eps) * random.laplace(temp, votes.shape)
      return np.argmax(noisy_scores, -1)
    else:
      return np.argmax(votes, -1)
  return sampled_votes


def shuffle_and_split_into_thirds(rng, X, y):
  X = random.permutation(rng, X)
  y = random.permutation(rng, y)
  return split_into_thirds(X, y)


def split_into_thirds(X, y):
  # TODO: use np.split
  split_idx = int(X.shape[0] / 3)
  X0, y0 = X[:split_idx], y[:split_idx]
  X1, y1 = X[split_idx:2*split_idx], y[split_idx:2*split_idx]
  X2, y2 = X[2*split_idx:], y[2*split_idx:]
  return (X0, y0), (X1, y1), (X2, y2)

def save_plot(x_int_a, x_int_b, y):
  x_a = [(lower + upper) / 2 for (upper, lower) in x_int_a]
  x_int_a = np.array([(x - lower, upper - x) for (x, (lower, upper)) in zip(x_a, x_int_a)]).T

  x_b = [(lower + upper) / 2 for (upper, lower) in x_int_b]
  x_int_b = np.array([(x - lower, upper - x) for (x, (lower, upper)) in zip(x_b, x_int_b)]).T

  y = np.array(y)

  linewidth = 2.2
  capthick = 2.2
  textsize = 17

  fig = plt.figure(figsize=(5, 4))
  ax = fig.add_subplot()
  ax.set_xlabel(r'E[Indicator]')
  ax.set_ylabel(r'Number of Deleted Points')
  ax.grid(True, linestyle='dashed', linewidth=linewidth)
  ax.set_rasterized(True)
  fig.tight_layout()

  ax.errorbar(x_a, y, xerr=x_int_a, color='red', ecolor='red', fmt=' ', capsize=3, capthick=capthick, linewidth=linewidth)
  ax.errorbar(x_b, y, xerr=x_int_b, color='blue', ecolor='blue', fmt=' ', capsize=3, capthick=capthick, linewidth=linewidth)

  """
  y_mins = [yi - yerri for yi, yerri in zip(y, yerr)]
  y_maxs = [yi + yerri for yi, yerri in zip(y, yerr)]
  yticks = np.linspace(min(y_mins), max(y_maxs), num=6)
  yticklabels = [str(round(yticki, 2)) for yticki in yticks]
  plt.yticks(yticks, labels=yticklabels)

  xticks = np.linspace(0.5, 1.0, num=4)
  xticklabels = [str(round(xticki, 2)) for xticki in xticks]
  plt.xticks(xticks, labels=xticklabels)

  ax.axvline(x=0.5, color='black', linestyle='-', linewidth=linewidth)
  plt.xlim(0.45, 1.05)
  """

  plt.rcParams.update({'font.size': textsize})
  plt.tight_layout()
  plt.savefig('out.png')
