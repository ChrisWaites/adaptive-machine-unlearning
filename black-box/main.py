import jax.numpy as np
from jax import random, nn, ops
#from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from tqdm import tqdm
from statistics import mean, stdev
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import keras
import tensorflow as tf
from utils import *
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

rng = random.PRNGKey(0)

num_classes = 2
num_shards = 3
num_trials = 200
num_points = 1000

tol = 6.5
sample = False

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.
X_test = X_test / 255.

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

train_mask = (y_train == 0) | (y_train == 1)
test_mask = (y_test == 0) | (y_test == 1)

X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

X_train = X_train[:num_points]
y_train = y_train[:num_points]

y_train = y_train + 1 # Offset labels so label == 0 is the null predction
y_test = y_test + 1 # Offset labels so label == 0 is the null predction

eps = None
#acc_thresh = 0.01
num_points_to_delete = None

y = []
x_int_no_retrain = []
x_int_retrain = []

for num_points_to_delete in [50, 100, 200, 400]:
  ModelClass = lambda: LookupTable(tol)

  before_del_test_accs = []
  before_del_accs = []

  retrain_accs = []
  no_retrain_accs = []

  rem_points = []

  for trial in tqdm(range(num_trials)):
    shuffle_rng, rng = random.split(rng)
    X_y_pieces = shuffle_and_split_into_thirds(shuffle_rng, X_train, y_train)

    ensemble = Ensemble(ModelClass, num_classes).fit(X_y_pieces)

    votes = ensemble.predict(X_test)
    temp, rng = random.split(rng)
    sampled_votes = sample_preds(temp, votes, sample, eps)
    correct = (sampled_votes == y_test)
    before_del_test_acc = np.mean(correct.astype(np.int32)).item()
    before_del_test_accs.append(before_del_test_acc)

    votes = ensemble.predict(X_train)
    temp, rng = random.split(rng)
    sampled_votes = sample_preds(temp, votes, sample, eps)
    #sampled_votes = np.argmax(votes, -1)

    correct = (sampled_votes == y_train)
    before_del_acc = np.mean(correct.astype(np.int32)).item()
    before_del_accs.append(before_del_acc)

    # print('Before del: {:.4f}'.format(acc))

    if num_points_to_delete:
      idxs = np.arange(0, correct.shape[0])[correct]
      temp, rng = random.split(rng)
      shuffled_idxs = random.permutation(temp, idxs)
      del_idxs = shuffled_idxs[:num_points_to_delete]
      deletion_mask = ~(ops.index_update(np.zeros(correct.shape[0]), ops.index[del_idxs], 1) == 1)
    else:
      deletion_mask = ~correct

    X_after_del_full = X_train[deletion_mask]
    y_after_del_full = y_train[deletion_mask]

    rem_points_after_del = (X_after_del_full.shape[0] / X_train.shape[0])
    rem_points.append(rem_points_after_del)

    # -------------- No retrain accuracy ----------------------

    sharded_deletion_mask_full = random.permutation(shuffle_rng, deletion_mask)
    split_idx = int(deletion_mask.shape[0] / num_shards)
    sharded_deletion_mask = [
      sharded_deletion_mask_full[:split_idx],
      sharded_deletion_mask_full[split_idx:2*split_idx],
      sharded_deletion_mask_full[2*split_idx:],
    ]
    X_y_pieces_after_del = [(X[mask], y[mask]) for (X, y), mask in zip(X_y_pieces, sharded_deletion_mask)]

    no_retrain_ensemble = Ensemble(ModelClass, num_classes).fit(X_y_pieces_after_del)

    votes = no_retrain_ensemble.predict(X_after_del_full)
    sampled_votes = np.argmax(votes, -1)
    no_retrain_acc = np.mean((sampled_votes == y_after_del_full).astype(np.int32)).item()
    no_retrain_accs.append(no_retrain_acc)

    # print('After del (no retrain): {:.4f}'.format(acc))

    # -------------- Retrain accuracy ------------------------

    reshuffle_rng, rng = random.split(rng)
    X_y_pieces_retrain = shuffle_and_split_into_thirds(reshuffle_rng, X_after_del_full, y_after_del_full)

    retrain_ensemble = Ensemble(ModelClass, num_classes).fit(X_y_pieces_retrain)

    votes = retrain_ensemble.predict(X_after_del_full)
    sampled_votes = np.argmax(votes, -1)
    retrain_acc = np.mean((sampled_votes == y_after_del_full).astype(np.int32)).item()
    retrain_accs.append(retrain_acc)

    # print('After del (retrain): {:.4f}'.format(acc))

  #print('Tol: {}'.format(tol))
  #print('Eps: {}'.format(eps))
  #print('Rem points after deletion: {:.2f}%'.format(mean(rem_points) * 100))
  #print('Train acc before deletion: {:.3f}'.format(mean(before_del_accs)))
  #print('Test acc before deletion: {:.3f}'.format(mean(before_del_test_accs)))

  acc_thresh = (mean(retrain_accs) + mean(no_retrain_accs)) / 2

  event_count_no_retrain = 0
  for no_retrain_acc in no_retrain_accs:
    if no_retrain_acc <= acc_thresh:
      event_count_no_retrain += 1

  event_count_retrain = 0
  for retrain_acc in retrain_accs:
    if retrain_acc <= acc_thresh:
      event_count_retrain += 1

  print('# deletions: {}'.format(num_points_to_delete))
  y.append(num_points_to_delete)

  print('Acc. after del (no retrain): {:.3f} ± {:.3f}'.format(mean(no_retrain_accs), stdev(no_retrain_accs)))
  print('Acc. after del (retrain):    {:.3f} ± {:.3f}'.format(mean(retrain_accs), stdev(retrain_accs)))

  print('Acc thresh: {}'.format(acc_thresh))

  lower_no_retrain, upper_no_retrain = proportion_confint(event_count_no_retrain, num_trials, alpha=0.025, method='binom_test')
  print('E[ind] (no retrain): [{:.3f}, {:.3f}]'.format(lower_no_retrain, upper_no_retrain))
  x_int_no_retrain.append((lower_no_retrain, upper_no_retrain))

  lower_retrain, upper_retrain = proportion_confint(event_count_retrain, num_trials, alpha=0.025, method='binom_test')
  print('E[ind] (retrain):    [{:.3f}, {:.3f}]'.format(lower_retrain, upper_retrain))
  x_int_retrain.append((lower_retrain, upper_retrain))

  print()

save_plot(x_int_no_retrain, x_int_retrain, y)

