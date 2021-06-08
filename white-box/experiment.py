from datasets import get_dataset
from jax import ops, nn
from privacy_accounting import compute_eps_uniform as compute_eps
from statistics import mean, stdev
from training import train, privately_train
from util import *
from task import get_task
import argparse
import jax.numpy as np
import json
import os
import pickle
import sys
import logging
import fire
import hashlib
from tqdm import tqdm
import logging


def select_points(confidences, cutoff, deletion_gap_quantile):
  most_confident_models = np.argmax(confidences, 1) # (num_points,): most confident model index for each point
  examples_pointing_to_targeted_models = (most_confident_models < cutoff) # (num_points,): binary, whether or not example points to targeted model

  top_two_model_indices = np.argsort(confidences, 1)[:, -2:]
  top_two_model_confidences = np.take_along_axis(confidences, top_two_model_indices, 1)
  gap = (top_two_model_confidences[:, 1] - top_two_model_confidences[:, 0]).reshape(-1)

  gap_targeted_models = gap[examples_pointing_to_targeted_models]

  logging.info('Targeted model gap stats: min={}, max={}, mean={}, std={}'.format(
    gap_targeted_models.min(), gap_targeted_models.max(),
    gap_targeted_models.mean(), gap_targeted_models.std()
  ))

  gap_threshold = np.quantile(gap_targeted_models, 1 - deletion_gap_quantile)
  selected_points = examples_pointing_to_targeted_models & (gap >= gap_threshold)

  return selected_points


def trial(
    rng,
    num_shards,
    num_slices,
    private,
    l2_norm_clip,
    noise_multiplier,
    iterations,
    batch_size,
    cutoff,
    step_size,
    examples_per_shard,
    sampling,
    model,
    deletion_gap_quantile,
    optimizer
  ):
  # Get config hash
  config_hash = hash_dict({
    'num_shards': num_shards,
    'num_slices': num_slices,
    'private': private,
    'l2_norm_clip': l2_norm_clip,
    'noise_multiplier': noise_multiplier,
    'iterations': iterations,
    'batch_size': batch_size,
    'cutoff': cutoff,
    'step_size': step_size,
    'examples_per_shard': examples_per_shard,
    'sampling': sampling,
    'model': model,
    'deletion_gap_quantile': deletion_gap_quantile,
    'optimizer': optimizer
  })

  # Get path for current trial
  loggable_rng = tuple([int(x) for x in rng])
  trial_hash = '_'.join(map(str, loggable_rng))
  trial_path = 'results/' + config_hash + '/' + trial_hash + '.json'

  # If identical trial has been run before, just return that
  if os.path.exists(trial_path):
    with open(trial_path, 'r') as f:
      return json.load(f)

  # Select training function
  if private:
    def train_fn(rng, params, predict, X, y):
      return privately_train(
        rng=rng, params=params, predict=predict, X=X, y=y, optimizer=optimizer,
        l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
        sampling=sampling, iterations=iterations, batch_size=batch_size,
        step_size=step_size, X_val=X_val, y_val=y_val
      )
  else:
    def train_fn(rng, params, predict, X, y):
      return train(
        rng=rng, params=params, predict=predict, X=X, y=y,
        optimizer=optimizer, sampling=sampling, iterations=iterations,
        batch_size=batch_size, step_size=step_size
      )

  # Keep track of metrics in json dictionary
  trial_json = { 'rng': loggable_rng }

  # Get dataset and task
  dataset, model_def = get_task(model)
  X_train, y_train, X_test, y_test = dataset
  dataset_rng = random.PRNGKey(0)

  # Shuffle test then extract val set
  temp, dataset_rng = random.split(dataset_rng)
  perm = random.permutation(temp, np.arange(0, X_test.shape[0]))
  X_test, y_test = X_test[perm], y_test[perm]
  split_idx = int(X_test.shape[0] / 2)
  X_val, y_val = X_test[:split_idx], y_test[:split_idx]
  X_test, y_test = X_test[split_idx:], y_test[split_idx:]

  # Shuffle dataset
  temp, dataset_rng = random.split(dataset_rng)
  perm = random.permutation(temp, np.arange(0, X_train.shape[0]))
  X_train, y_train = X_train[perm], y_train[perm]

  # Shard and slice dataset
  X_train, y_train = X_train[:num_shards*examples_per_shard], y_train[:num_shards*examples_per_shard]
  X, y = shard_and_slice(num_shards, num_slices, X_train, y_train)

  init_params, predict = model_def

  # Calculate privacy parameters (epsilon, delta)
  if private:
    N = np.concatenate(X[0]).shape[0]
    delta = 1 / (N ** 1.1)
    try:
      epsilon = compute_eps(iterations, noise_multiplier, N, batch_size, delta)
    except:
      epsilon = '∞'
  else:
    delta = '∞'
    epsilon = '∞'

  logging.info('Privacy: Eps = {}, Delta = {}'.format(epsilon, delta))

  # Save privacy parameters
  trial_json['epsilon'] = epsilon
  trial_json['delta'] = delta

  # Need to add additional batch dimension of size 1 if doing private training
  if private:
    X = [[np.expand_dims(X_i, 1) for X_i in slice] for slice in X]
    y = [[np.expand_dims(y_i, 1) for y_i in slice] for slice in y]

  logging.info('Training ensemble (Shards={}, Slices={})'.format(num_shards, num_slices))
  params = get_trained_sharded_and_sliced_params(rng, init_params, predict, X, y, train_fn)

  # Compute accuracy of ensemble before deletion
  ensemble_acc = sharded_and_sliced_accuracy(params, predict, X_test, y_test).item()
  logging.info('Ensemble acc (before del): {:.4f}'.format(ensemble_acc))
  trial_json['ensemble_acc_before_deletion'] = ensemble_acc

  # Compute accuracy of models before performing deletion
  accs = []
  for i in range(len(params)):
    accs.append(accuracy(params[i][-1], predict, X_test, y_test).item())
  targeted, not_targeted = accs[:cutoff], accs[cutoff:]

  trial_json['accs_before_deletion'] = accs
  try:
    logging.info('Accuracy ( t):\t{:.4f} ± {:.4f}'.format(mean(targeted), stdev(targeted)))
    logging.info('Accuracy (nt):\t{:.4f} ± {:.4f}'.format(mean(not_targeted), stdev(not_targeted)))
  except:
    logging.info('Accuracy ( t):\t{:.4f}'.format(mean(targeted)))
    logging.info('Accuracy (nt):\t{:.4f}'.format(mean(not_targeted)))

  # Identify targeted models
  preds = np.array([nn.softmax(predict(slice_params[-1], X_train)) for slice_params in params]).swapaxes(0, 1) # (num_points, num_models, num_classes): predictions of each model for each point and class
  confidences = np.take_along_axis(preds, np.argmax(y_train, 1).reshape(-1, 1, 1), -1).squeeze(-1) # (num_points, num_models): confidence of each model for correct label
  most_confident_models = np.argmax(confidences, 1) # (num_points,): most confident model index for each point

  # selected_points = (most_confident_models <= cutoff) # (num_points,): binary, whether or not example points to targeted model
  selected_points = select_points(confidences, cutoff, deletion_gap_quantile)

  # True model assignments for each point
  true_model_indices = np.zeros(X_train.shape[0])
  for i in range(len(params)):
    true_model_indices = ops.index_update(true_model_indices, ops.index[i*examples_per_shard:(i + 1)*examples_per_shard], i)

  # Compute precision and recall in predicting correct shard
  shard_preds = ops.index_update(most_confident_models, ~selected_points, -1)
  shard_prediction_precision = (shard_preds[selected_points] == true_model_indices[selected_points]).mean().item()
  num_targeted_points = examples_per_shard * cutoff
  shard_prediction_recall = (shard_preds[:num_targeted_points] == true_model_indices[:num_targeted_points]).mean().item()

  # Save and log shard prediction accuracy
  logging.info('Shard pred. precision:\t{:.4f}'.format(shard_prediction_precision))
  logging.info('Shard pred. recall:   \t{:.4f}'.format(shard_prediction_recall))
  logging.info('Expected accuracy:    \t{:.4f}'.format(1 / len(params)))
  trial_json['shard_prediction_precision'] = shard_prediction_precision
  trial_json['shard_prediction_recall'] = shard_prediction_recall

  # Delete points corresponding to targeted model
  logging.info('Deleting points and retraining ensemble...')
  idxs_to_be_deleted = [i for i, v in enumerate(selected_points) if v]
  params, X, y = delete_and_retrain_multiple(rng, idxs_to_be_deleted, params, predict, X, y, train_fn)

  # Log accuracy of ensemble after deletion
  ensemble_acc = sharded_and_sliced_accuracy(params, predict, X_test, y_test).item()
  logging.info('Ensemble acc (after del): {:.4f}'.format(ensemble_acc))
  trial_json['ensemble_acc_after_deletion'] = ensemble_acc

  # Compute accuracy and number of points per shard after deletion
  accs, lengths = [], []
  for i in range(len(params)):
    accs.append(accuracy(params[i][-1], predict, X_test, y_test).item())
    lengths.append(sum([len(piece) for piece in X[i]]))

  # Save accuracy and number of points to trial json
  trial_json['accs_after_deletion'] = accs
  trial_json['lengths_after_deletion'] = lengths

  targeted, not_targeted = accs[:cutoff], accs[cutoff:]
  targeted_lengths, not_targeted_lengths = lengths[:cutoff], lengths[cutoff:]

  # Log accuracy of models after deletion
  avg_accuracy_targeted = mean(targeted)
  avg_accuracy_not_targeted = mean(not_targeted)
  try:
    logging.info('Accuracy ( t):\t{:.4f} ± {:.4f}'.format(mean(targeted), stdev(targeted)))
    logging.info('Accuracy (nt):\t{:.4f} ± {:.4f}'.format(mean(not_targeted), stdev(not_targeted)))
  except:
    logging.info('Accuracy ( t):\t{:.4f}'.format(mean(targeted)))
    logging.info('Accuracy (nt):\t{:.4f}'.format(mean(not_targeted)))

  # Log number of points per shard after deletion
  try:
    logging.info('# points ( t):\t{:.4f} ± {:.4f}'.format(mean(targeted_lengths), stdev(targeted_lengths)))
    logging.info('# points (nt):\t{:.4f} ± {:.4f}'.format(mean(not_targeted_lengths), stdev(not_targeted_lengths)))
  except:
    logging.info('# points ( t):\t{:.4f}'.format(mean(targeted_lengths)))
    logging.info('# points (nt):\t{:.4f}'.format(mean(not_targeted_lengths)))

  # Calculate indicator statistic
  indicator = 1 if avg_accuracy_targeted < avg_accuracy_not_targeted else 0
  logging.info('Indicator: {}'.format(indicator))
  trial_json['indicator'] = indicator

  print('Done!')
  exit()
  # Save trial in relevant dictionary
  with open(trial_path, 'w') as f:
    f.write(json.dumps(trial_json, indent=2, sort_keys=True))

  return trial_json

def experiment_local(rng, trials, **kwargs):
  results = []
  for _ in tqdm(range(trials)):
    results.append(trial(rng, **kwargs))
    rng, _ = random.split(rng)
  return results

def hash_dict(exp_dict):
  dict2hash = ""
  if not isinstance(exp_dict, dict):
    raise ValueError('exp_dict is not a dict')
  for k in sorted(exp_dict.keys()):
    if '.' in k:
      raise ValueError(". has special purpose")
    elif isinstance(exp_dict[k], dict):
      v = hash_dict(exp_dict[k])
    elif isinstance(exp_dict[k], tuple):
      raise ValueError("tuples can't be hashed yet, consider converting tuples to lists")
    else:
      v = exp_dict[k]
    dict2hash += os.path.join(str(k), str(v))
  hash_id = hashlib.md5(dict2hash.encode()).hexdigest()
  return hash_id

def initialize(config, experiment_hash_offset=0):
  # Create results dict if it doesn't already exist
  if not os.path.exists('results'):
    os.makedirs('results')

  # Get trial config
  config_hash = hash_dict(config)
  config_dir = 'results/' + config_hash

  # Create config dict if it doesn't already exist
  if not os.path.exists(config_dir):
    os.makedirs(config_dir)

    config_path = config_dir + '/config.json'

    # Save config definition in relevant dictionary
    with open(config_path, 'w') as f:
      f.write(json.dumps(config, indent=2, sort_keys=True))

  # Get existing rngs in config dir
  rngs = []
  for trial_path in os.listdir(config_dir):
      try:
        with open(config_dir + '/' + trial_path, 'r') as f:
          trial = json.load(f)
          rngs.append(np.array(trial['rng']))
      except: pass

  # Create rng as hash of config, or generate new one from previous
  rng = random.PRNGKey((int(config_hash, 16) + experiment_hash_offset) % (2 ** 31 - 1))
  while any([np.array_equal(rng, rng_i).item() for rng_i in rngs]):
    rng, _ = random.split(rng)

  return rng


def main(experiment_path, experiment_hash_offset=0, parallel_execution=False, log=False):
  if log:
    logging.getLogger().setLevel(logging.INFO)

  # Load config file from path
  with open(experiment_path, 'r') as f:
    config = json.load(f)

  # Seperate out the number of trials and the actual trial configuration
  trials, config = config['trials'], config['experiment_config']

  # Create directory for this config (also computes the rng to start with)
  rng = initialize(config, experiment_hash_offset)

  result = experiment_local(rng, trials, **config)

if __name__ == '__main__':
  fire.Fire(main)
