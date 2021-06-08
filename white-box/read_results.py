import sys
import json
import os
from statistics import mean, stdev
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from tabulate import tabulate
from jax import random


def get_experiment_results(results_dir, hashes=None):
  """
  Reads experiment directory and aggregates results across trial .json files into a unified dictionary.
  Dictionary maps hash of config dictionary to list of trials, each containing experiment information.
  """
  experiment_results = {}

  # For each experiment in the results directory
  for experiment_hash in os.listdir(results_dir):
    if not experiment_hash.startswith('.'):
      experiment_dir = os.path.join(results_dir, experiment_hash)

      # Skip if not in specified hashes
      if hashes and experiment_hash not in hashes:
        continue

      # For each trial in experiment directory
      trials, config = [], None
      for trial_path in os.listdir(experiment_dir):
        if not trial_path.startswith('.'):
          with open(os.path.join(experiment_dir, trial_path), 'r') as f:
            result = json.loads(f.read())

          # Either save config or trial
          if trial_path == 'config.json':
            config = result
          else:
            trials.append(result)

      if config:
        experiment_results[experiment_hash] = (config, trials)
  return experiment_results


def filter_results(experiment_results, expects_dict):
  """
  Filters experiment results dictionary for configs with specified values in expects_dict
  e.g. expects_dict = { "noise_multiplier": 1.2 } will only get experiments with this value for the noise multiplier.
  """
  filtered_experiment_results = {}
  for hash, (config, trials) in experiment_results.items():
    if 'optimizer' not in config:
      config['optimizer'] = '-'
    include = True
    for key, value in expects_dict.items():
      if config[key] != value:
        include = False
    if include:
      filtered_experiment_results[hash] = (config, trials)
  return filtered_experiment_results


def get_sorted_trials(trials):
  hash_to_indices = {}
  for i, trial in enumerate(trials):
    rng = np.array(trial['rng'], dtype=np.uint32)
    loggable_rng = tuple([int(x) for x in rng])
    trial_hash = '_'.join(map(str, loggable_rng))
    hash_to_indices[trial_hash] = i

  def get_start(hash_to_indices):
    for hash in hash_to_indices:
      parts = list(map(int, hash.split('_')))
      if parts[0] == 0:
        return np.array(parts, dtype=np.uint32)
    raise Exception('Can\t find start!')

  sorted_trials = []
  rng = get_start(hash_to_indices)
  for i in range(len(trials)):
    loggable_rng = tuple([int(x) for x in rng])
    trial_hash = '_'.join(map(str, loggable_rng))
    sorted_trials.append(trials[hash_to_indices[trial_hash]])
    rng, _ = random.split(rng)
  return sorted_trials


def get_summaries(dirname='results/', expects_dict=None, trial_limit=350):
  experiment_results = get_experiment_results(dirname)
  if expects_dict:
    experiment_results = filter_results(experiment_results, expects_dict)

  summaries, x, y, xerr, yerr, labels = [], [], [], [], [], []
  for hash, (config, trials) in experiment_results.items():
    label = str(config['noise_multiplier'])
    exp_shard_prediction_acc = mean(trial['shard_prediction_precision'] for trial in trials)
    std_shard_prediction_acc = stdev(trial['shard_prediction_precision'] for trial in trials)

    indicators = [trial['indicator'] for trial in trials]
    exp_indicator = mean(indicators)
    err_interval = proportion_confint(sum(indicators), len(indicators), alpha=0.05, method='binom_test')

    exp_before_deletion = mean([trial['ensemble_acc_before_deletion'] for trial in trials])
    std_before_deletion = 2 * stdev([trial['ensemble_acc_before_deletion'] for trial in trials])

    exp_after_deletion = mean([trial['ensemble_acc_after_deletion'] for trial in trials])
    std_after_deletion = 2 * stdev([trial['ensemble_acc_after_deletion'] for trial in trials])


    x.append(exp_indicator)
    xerr.append((exp_indicator - err_interval[0], err_interval[1] - exp_indicator))
    y.append(exp_after_deletion)
    yerr.append(std_after_deletion)
    labels.append(label)

    summaries.append([
      '[{:.3f}, {:.3f}]'.format(err_interval[0], err_interval[1]),
      '{:.3f} ± {:.3f}'.format(exp_after_deletion, std_after_deletion),
      '{:.3f} ± {:.3f}'.format(exp_before_deletion, std_before_deletion),
      config['noise_multiplier'],
      '{:.3f} ± {:.3f}'.format(exp_shard_prediction_acc, std_shard_prediction_acc),
      hash,
    ])

  save_plot(name, x, xerr, y, yerr, labels)

  headers = ['indicator', 'acc (after)', 'acc (bef)', 'noise', 'shard pred acc.', 'hash']
  summaries = reversed(sorted(summaries, key=lambda summary: summary[0])) # Sort by expected indicator, highest first

  return headers, summaries


def get_label_offset(name):
  if name == 'mnist (2)':
    return (0.007, 0.001)
  elif name == 'mnist (6)':
    return (0.007, 0.005)
  elif name == 'fmnist (2)':
    return (0.007, 0.005)
  elif name == 'fmnist (6)':
    return (0.007, 0.003)
  elif name == 'cifar (2)':
    return (0.007, 0.005)
  elif name == 'cifar (6)':
    return (0.007, 0.005)
  else:
    raise Exception('No label offset defined for experiment name {}'.format(name))


def save_plot(name, x, xerr, y, yerr, labels):
  linewidth = 2.2
  capthick = 2.2
  textsize = 17

  fig = plt.figure(figsize=(5, 4))
  ax = fig.add_subplot()
  ax.set_xlabel(r'Expectation of Indicator')
  ax.set_ylabel(r'Ensemble Test Accuracy')
  ax.grid(True, linestyle='dashed', linewidth=linewidth)
  ax.set_rasterized(True)
  fig.tight_layout()

  label_del = get_label_offset(name)

  xerr = np.array(xerr).T
  ax.errorbar(x, y, xerr=xerr, yerr=yerr, color='black', ecolor='black', fmt='o', capsize=3, capthick=capthick, linewidth=linewidth)
  for label, (xi, yi) in zip(labels, zip(x, y)):
    ax.annotate(label, (xi, yi), xytext=(xi + label_del[0], yi + label_del[1]), size=textsize)

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

  if not os.path.exists('plots'):
    os.makedirs('plots')

  plt.rcParams.update({'font.size': textsize})
  plt.tight_layout()
  plt.savefig('plots/' + name + '.png')


def print_summaries(dirname='results/', expects_dict=None):
  headers, summaries = get_summaries(dirname, expects_dict)
  print(tabulate(summaries, headers=headers, tablefmt='orgtbl'))


if __name__ == '__main__':
  for name, expects_dict in [
      ('cifar (6)', { 'model': 'cifar_smaller_conv', 'num_shards': 6 }),
      ('cifar (2)', { 'model': 'cifar_smaller_conv', 'num_shards': 2 }),

      ('fmnist (6)', { 'model': 'fmnist_conv', 'num_shards': 6 }),
      ('fmnist (2)', { 'model': 'fmnist_conv', 'num_shards': 2 }),

      ('mnist (6)', { 'model': 'mnist_conv', 'num_shards': 6 }),
      ('mnist (2)', { 'model': 'mnist_conv', 'num_shards': 2 }),
    ]:
    print(name + '\n')
    print_summaries('results/', expects_dict)
    print()

