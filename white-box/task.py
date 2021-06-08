from models import get_model
from datasets import get_dataset

def get_task(task):
  dataset = get_dataset(task)
  model = get_model(task)
  return dataset, model
