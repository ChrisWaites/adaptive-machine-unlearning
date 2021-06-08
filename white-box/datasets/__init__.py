from .mnist import mnist
from .fmnist import fmnist
from .cifar import cifar

def get_dataset(model):
  if 'fmnist' in model:
    return fmnist()
  elif 'mnist' in model:
    return mnist()
  elif 'cifar' in model:
    return cifar()
  else:
    raise Exception('Invalid model.')

