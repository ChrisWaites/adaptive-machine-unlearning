from jax.experimental import stax
import jax.numpy as np

def get_model(task):
  return {
    'mnist_linear': mnist_linear,
    'mnist_mlp': mnist_mlp,
    'mnist_conv': mnist_conv,

    'fmnist_linear': fmnist_linear,
    'fmnist_mlp': fmnist_mlp,
    'fmnist_conv': fmnist_conv,

    'cifar_linear': cifar_linear,
    'cifar_mlp': cifar_mlp,
    'cifar_conv': cifar_conv,
    'cifar_smaller_conv': cifar_smaller_conv,

    'medmnist_conv': medmnist_conv,
  }[task]()

def mnist_linear():
  init_fun, predict = stax.serial(
    stax.Flatten,
    stax.Dense(10)
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def mnist_mlp():
  init_fun, predict = stax.serial(
    stax.Flatten,
    stax.Dense(1000),
    stax.Tanh,
    stax.Dense(10)
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def mnist_conv():
  init_fun, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Tanh,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Tanh,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Tanh,
    stax.Dense(10),
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def fmnist_linear():
  init_fun, predict = stax.serial(
    stax.Flatten,
    stax.Dense(10)
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def fmnist_mlp():
  init_fun, predict = stax.serial(
    stax.Flatten,
    stax.Dense(1000),
    stax.Tanh,
    stax.Dense(10)
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def fmnist_conv():
  init_fun, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Tanh,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Tanh,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Tanh,
    stax.Dense(10),
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def CustomMean():
  def init_fun(rng, input_shape):
    return (input_shape[0], input_shape[-1]), ()
  def apply_fun(params, inputs, **kwargs):
    return np.mean(inputs, (1, 2))
  return init_fun, apply_fun

def cifar_linear():
  init_fun, predict = stax.serial(
    stax.Flatten,
    stax.Dense(10)
  )
  init_params = lambda rng: init_fun(rng, (-1, 32, 32, 3))[1]
  return init_params, predict

def cifar_mlp():
  init_fun, predict = stax.serial(
    stax.Flatten,
    stax.Dense(1000),
    stax.Tanh,
    stax.Dense(10)
  )
  init_params = lambda rng: init_fun(rng, (-1, 32, 32, 3))[1]
  return init_params, predict

def cifar_conv():
  init_fun, predict = stax.serial(
    stax.Conv(32, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.Conv(32, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),

    stax.Conv(64, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.Conv(64, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),

    stax.Conv(128, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.Conv(128, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.AvgPool((2, 2), strides=(2, 2), padding='VALID'),

    stax.Conv(256, (3, 3), padding='SAME', strides=(1, 1)),
    stax.Tanh,
    stax.Conv(10, (3, 3), padding='SAME', strides=(1, 1)),

    CustomMean(),
  )
  init_params = lambda rng: init_fun(rng, (-1, 32, 32, 3))[1]
  return init_params, predict

def cifar_smaller_conv():
  init_fun, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
  )
  init_params = lambda rng: init_fun(rng, (-1, 32, 32, 3))[1]
  return init_params, predict

def medmnist_conv():
  init_fun, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(9),
  )
  init_params = lambda rng: init_fun(rng, (-1, 28, 28, 3))[1]
  return init_params, predict
