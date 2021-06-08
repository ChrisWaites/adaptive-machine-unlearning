import jax.numpy as np
from jax import partial, grad, jit, nn, random, vmap, tree_util, ops
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, Tanh, Conv, MaxPool, Flatten

from tqdm import tqdm
from time import time


def log_time(f):
  """Utility function for printing the execution time of a function in wall-time."""
  def g(*args):
    start = time()
    ret = f(*args)
    end = time()
    print('Function {} took {:.4} seconds.'.format(f.__name__, end - start))
    return ret
  return g

def shuffle(rng, *args):
  """Shuffles a set of args, each the same way."""
  return (random.permutation(rng, arg) for arg in args)

def shard_and_slice(num_shards, num_slices, *args):
  """Shards and slices an array.

  This means, after this function, an array X becomes indexable as
  X[0<=i<num_shards][0<=j<num_slices], referring to the j'th slice of the i'th shard.
  """
  return ([np.split(shard, num_slices) for shard in np.split(arg, num_shards)] for arg in args)

def train_shard(rng, init_params, predict, X, y, train):
  """Given an individual shard, trains a model for each slice."""
  shard_params = [init_params(rng)]
  for i in range(len(X)):
    temp, rng = random.split(rng)
    X_train, y_train = np.concatenate(X[:i+1]), np.concatenate(y[:i+1])
    finetuned_params = train(temp, shard_params[-1], predict, X_train, y_train)
    shard_params.append(finetuned_params)
  return shard_params

#@log_time
def get_trained_sharded_and_sliced_params(rng, init_params, predict, X, y, train):
  """Given a sharded and sliced dataset, constructs a sharded and sliced parameter object."""
  rngs = random.split(rng, len(X))
  return [
    train_shard(rng, init_params, predict, X_shard, y_shard, train)
    for rng, X_shard, y_shard in list(zip(rngs, X, y))
  ]

def get_votes(params, predict, X):
  """Randomly samples from scores - assumes scores is already normalized."""
  preds = predict(params[0][-1], X)
  one_hot = np.eye(preds.shape[1])
  votes = one_hot[np.argmax(preds, axis=1)]
  for slice_params in params[1:]:
    preds = predict(slice_params[-1], X)
    votes += one_hot[np.argmax(preds, axis=1)]
  return votes

def randomly_sample(rng, scores):
  """Randomly samples from scores - assumes scores is already normalized."""
  rngs = random.split(rng, scores.shape[0])
  indices = np.arange(scores.shape[1])
  return np.array([random.choice(rng, indices, p=weights) for rng, weights in zip(rngs, scores)])

def max_votes(votes):
  """Deterministic aggregation mechanism, simply returns class with highest score."""
  return np.argmax(votes, axis=1)

def exponential_mechanism(rng, votes, per_example_epsilon, sensitivity=1.):
  """Exponential mechanism."""
  scores = nn.softmax(per_example_epsilon * votes / (2 * sensitivity))
  return randomly_sample(rng, scores)

def lnmax(rng, votes, per_example_epsilon):
  """LNMax: Discovering frequent patterns in sensitive data, Bhaskar et al."""
  votes = votes + (1 / (per_example_epsilon / 2)) * random.laplace(rng, votes.shape)
  return np.argmax(votes, axis=1)

def gnmax(rng, votes, sigma):
  """GNMax: Section 4 of https://arxiv.org/pdf/1802.08908.pdf."""
  votes = votes + sigma * random.normal(rng, votes.shape)
  return np.argmax(votes, axis=1)

def confident_gnmax(rng, votes, sigma_1, sigma_2, threshold):
  """GNMax: Section 4 of https://arxiv.org/pdf/1802.08908.pdf."""
  rng_1, rng_2 = random.split(rng)
  max_vote_count = np.max(votes, axis=1)
  noisy_max_vote_count = max_vote_count + sigma_1 * random.normal(rng_1, max_vote_count.shape)
  didnt_achieve_consensus = noisy_max_vote_count < threshold
  noisy_vote_counts = votes + random.normal(rng_2, votes.shape)
  votes = np.argmax(noisy_vote_counts, axis=1)
  votes = ops.index_update(votes, ops.index[didnt_achieve_consensus], -1.)
  return votes

def sharded_and_sliced_predict(params, predict, X, aggregate=max_votes):
  """Given a sharded and sliced dataset and params, defines the prediction function."""
  return aggregate(get_votes(params, predict, X))

def get_location(idx, X):
  """Retrieves the location, i.e., the shard index i, slice index j, and value index k of the idx'th element in the struct.

  For example, given a sharded and sliced data object with two shards:
    [[_, _, _], [_, _, _], [_, #, _]], [...]
  Then, the # would be the 7th element (idx), and it would be at location (shard: 0, slice: 2, value: 1).
  """
  num_examples = 0
  for i in range(len(X)):
    for j in range(len(X[i])):
      new_num_examples = num_examples + len(X[i][j])
      if idx < new_num_examples:
        return i, j, idx - num_examples
      num_examples = new_num_examples

def delete_index(idx, *args):
  """Deletes the idx'th element of each arg (assumes they all have the same shape)."""
  i, j, k = get_location(idx, args[0])
  for arg in args:
    arg[i][j] = arg[i][j][np.eye(len(arg[i][j]))[k] == 0.]
  return args

#@log_time
def delete_and_retrain(rng, idx, params, predict, X, y, train):
  """Deletes the idx'th element, and then retrains the sharded and sliced params accordingly.

  That is, if we want to delete the value at location (shard: i, slice: j, value: k), then we retrain
  all parameters of the i'th shard from slice j+1 onwards.
  """
  X, y = delete_index(idx, X, y)
  i, j, _ = get_location(idx, X)
  for s in range(j, len(X[i])):
    temp, rng = random.split(rng)
    X_train, y_train = np.concatenate(X[i][:s+1]), np.concatenate(y[i][:s+1])
    params[i][s+1] = train(temp, params[i][s], predict, X_train, y_train)
  return params, X, y

def delete_random_index_and_retrain(rng, params, predict, X, y, train):
  """Randomly samples and deletes idx'th element, and then retrains the sharded and sliced params accordingly."""
  num_examples = total_examples(X)
  idx = random.randint(rng, (), 0, num_examples).item()
  return delete_and_retrain(rng, idx, params, predict, X, y, train)

#@log_time
def delete_and_retrain_multiple(rng, idxs, params, predict, X, y, train):
  """The same as delete_and_retrain, but allows for multiple indices to be specified.

  Can be more efficient than calling delete_and_retrain multiple times in sequence,
  because if two elements fall in the same slice, you don't repeat work.
  """
  num_examples = 0
  for i in range(len(X)):
    update_occured = False
    for j in range(len(X[i])):
      new_num_examples = num_examples + len(X[i][j])
      mask = [True for i in range(len(X[i][j]))]
      while len(idxs) > 0 and idxs[0] < new_num_examples:
        update_occured = True
        idx = idxs.pop(0)
        mask[idx - num_examples] = False
      mask = np.array(mask)
      X[i][j], y[i][j] = X[i][j][mask], y[i][j][mask]
      if update_occured:
        temp, rng = random.split(rng)
        X_train, y_train = np.concatenate(X[i][:j+1]), np.concatenate(y[i][:j+1])
        params[i][j+1] = train(temp, params[i][j], predict, X_train, y_train)
      num_examples = new_num_examples
  return params, X, y

def total_examples(X):
  """Counts the total number of examples of a sharded and sliced data object X."""
  count = 0
  for i in range(len(X)):
    for j in range(len(X[i])):
      count += len(X[i][j])
  return count

def full_dataset(*args):
  return (np.concatenate([np.concatenate(arg[i]) for i in range(len(arg))]) for arg in args)

def accuracy(params, predict, X, y):
  targets = np.argmax(y, axis=1)
  predictions = np.argmax(predict(params, X), axis=1)
  return np.mean(predictions == targets)

def sharded_and_sliced_accuracy(params, predict, X, y):
  targets = np.argmax(y, axis=1)
  predictions = sharded_and_sliced_predict(params, predict, X)
  return np.mean(predictions == targets)
