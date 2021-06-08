from jax import partial, grad, jit, random, vmap, pmap
from jax.experimental import optimizers, stax
from jax.tree_util import tree_flatten, tree_unflatten
import itertools
import jax.numpy as np


def loss(params, predict, batch):
  inputs, targets = batch
  logits = predict(params, inputs)
  logits = stax.logsoftmax(logits)
  return -np.mean(np.sum(logits * targets, axis=1))

def accuracy(params, predict, X, y):
  if X.shape[1] == 1:
    X = np.squeeze(X, 1)
    y = np.squeeze(y, 1)

  targets = np.argmax(y, axis=1)
  predictions = np.argmax(predict(params, X), axis=1)
  return np.mean(predictions == targets)

def get_sampling_fn(sampling):
  if sampling == 'batch':
    def data_stream(rng, batch_size, X, y):
      # If the dataset is smaller than batch size, just continually return dataset
      if X.shape[0] < batch_size:
        while True:
          yield X, y
      # Otherwise, repeatedly go through dataset and shuffle after every epoch
      indices = np.arange(0, X.shape[0])
      while True:
        temp, rng = random.split(rng)
        indices = random.permutation(temp, indices)
        batch_start = 0
        while batch_start + batch_size < X.shape[0]:
          batch_idx = indices[batch_start:batch_start+batch_size]
          yield X[batch_idx], y[batch_idx]
          batch_start += batch_size

  elif sampling == 'uniform':
    def data_stream(rng, batch_size, X, y):
      indices = np.arange(0, X.shape[0])
      while True:
        temp, rng = random.split(rng)
        curr_indices = random.permutation(temp, indices)[:batch_size]
        yield X[curr_indices], y[curr_indices]

  elif sampling == 'poisson':
    raise Exception('Invalid sampling function: {}'.format(sampling))
  else:
    raise Exception('Invalid sampling function: {}'.format(sampling))

  return data_stream

def get_optimizer(optimizer):
  if optimizer == 'sgd':
    return optimizers.sgd
  if optimizer == 'momentum':
    return lambda step_size: optimizers.momentum(step_size, 0.9)
  elif optimizer == 'adam':
    return optimizers.adam
  else:
    raise Exception('Invalid optimizer: {}'.format(optimizer))

def l2norm(grads):
  grads, _ = tree_flatten(grads)
  return sum([np.sum(np.square(grad)) for grad in grads])

def train(rng, params, predict, X, y, optimizer, sampling, iterations, batch_size, step_size):
  """Generic train function called for each slice.

  Responsible for, given an rng key, a set of parameters to be trained, some inputs X and some outputs y,
  finetuning the params on X and y according to some internally defined training configuration.
  """
  opt_fn = get_optimizer(optimizer)
  opt_init, opt_update, get_params = opt_fn(step_size)
  opt_state = opt_init(params)

  data_stream = get_sampling_fn(sampling)
  temp, rng = random.split(rng)
  batches = data_stream(temp, batch_size, X, y)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    g = grad(loss)(params, predict, batch)
    return opt_update(i, g, opt_state)

  for i in range(iterations):
    opt_state = update(i, opt_state, next(batches))

  """
    if i % 20 == 0:
      print(accuracy(get_params(opt_state), predict, X, y))
  exit()
  """

  """
    if i % 20 == 0:
      if not prev_acc:
        prev_acc = accuracy(get_params(opt_state), predict, X, y)
      else:
        curr_acc = accuracy(get_params(opt_state), predict, X, y)
        if curr_acc - prev_acc <= 0.03:
          return get_params(opt_state)
        prev_acc = curr_acc
  exit()
  print('-'*100)
  """

  return get_params(opt_state)


def privately_train(rng, params, predict, X, y, optimizer, l2_norm_clip, noise_multiplier, sampling, iterations, batch_size, step_size, X_val, y_val):
  """Generic train function called for each slice.

  Responsible for, given an rng key, a set of parameters to be trained, some inputs X and some outputs y,
  finetuning the params on X and y according to some internally defined training configuration.
  """
  def clipped_grad(params, l2_norm_clip, single_example_batch):
    grads = grad(loss)(params, predict, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = np.linalg.norm([np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = np.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)

  def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    clipped_grads = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [np.sum(g, 0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape) for r, g in zip(rngs, aggregated_clipped_grads)]
    normalized_noised_aggregated_clipped_grads = [g / batch_size for g in noised_aggregated_clipped_grads]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

  @jit
  def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)
    g = private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, batch_size)
    return opt_update(i, g, opt_state)

  opt_fn = get_optimizer(optimizer)
  opt_init, opt_update, get_params = opt_fn(step_size)
  opt_state = opt_init(params)

  data_stream = get_sampling_fn(sampling)
  temp, rng = random.split(rng)
  batches = data_stream(temp, batch_size, X, y)

  best_opt_state, best_acc = None, None
  for i in range(iterations):
    temp, rng = random.split(rng)
    opt_state = private_update(temp, i, opt_state, next(batches))

    if i % 20 == 0:
      curr_acc = accuracy(get_params(opt_state), predict, X_val, y_val)
      #print(i, curr_acc)
      if best_opt_state == None or curr_acc >= best_acc:
        best_opt_state = opt_state
        best_acc = curr_acc
  #exit()

  return get_params(best_opt_state)
