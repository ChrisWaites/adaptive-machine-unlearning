import matplotlib.pyplot as plt
import jax.numpy as np
from keras.datasets import cifar10

def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def cifar(permute_train=True):
  train, test = cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.asarray(train_data, dtype=np.float32) / 255.
  test_data = np.asarray(test_data, dtype=np.float32) / 255.

  one_hot = np.eye(10)

  train_labels = one_hot[np.asarray(train_labels, dtype=np.int32).squeeze()]
  test_labels = one_hot[np.asarray(test_labels, dtype=np.int32).squeeze()]

  return train_data, train_labels, test_data, test_labels
