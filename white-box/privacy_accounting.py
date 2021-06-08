import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(t, noise_multi, n, batch_size):
  """Compute mu from uniform subsampling."""

  c = batch_size * np.sqrt(t) / n
  return np.sqrt(2) * c * np.sqrt(
      np.exp(noise_multi**(-2)) * norm.cdf(1.5 / noise_multi) +
      3 * norm.cdf(-0.5 / noise_multi) - 2)


def compute_mu_poisson(t, noise_multi, n, batch_size):
  """Compute mu from Poisson subsampling."""

  return np.sqrt(np.exp(noise_multi**(-2)) - 1) * np.sqrt(t) * batch_size / n


def delta_eps_mu(eps, mu):
  """Compute dual between mu-GDP and (epsilon, delta)-DP."""
  return norm.cdf(-eps / mu +
                  mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
  """Compute epsilon from mu given delta via inverse dual."""

  def f(x):
    """Reversely solve dual by matching delta."""
    return delta_eps_mu(x, mu) - delta

  return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def compute_eps_uniform(t, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of uniform subsampling."""

  return eps_from_mu(
      compute_mu_uniform(t, noise_multi, n, batch_size), delta)


def compute_eps_poisson(t, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of Poisson subsampling."""

  return eps_from_mu(
      compute_mu_poisson(t, noise_multi, n, batch_size), delta)
