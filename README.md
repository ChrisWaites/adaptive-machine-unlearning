# Adaptive Machine Unlearning

This repository contains the implementation of the experiments shown in <em>Adaptive Machine Unlearning</em>.

## Requirements

Experiments were run using Python 3.8.10. To install dependencies:

```
pip install -r requirements.txt
```

In addition to installing the required dependencies, you will need to install [JAX](https://github.com/google/jax). For example, to install for CPU:

```
pip install --upgrade pip
pip install --upgrade jax jaxlib
```

To install for GPU, the command you run will be dependent on your device. For example, if you're running CUDA 11.0 (as used for our experiments):

```
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.67+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Evaluation and Results for Figure 1

To read the precomputed results:

```
cd white-box
python read_results.py
```

This should output the following results, and save the plots for Figure 1 under `white-box/plots`:

cifar (6)

| indicator      | acc (after)   | acc (bef)     |   noise | shard pred acc.   | hash                             |
|--------------- | ------------- | ------------- | ------- | ----------------- | ---------------------------------|
| [0.890, 0.952] | 0.507 ± 0.025 | 0.572 ± 0.011 |    0    | 0.303 ± 0.005     | 2001210bc030deaf83697e3ef79c205b |
| [0.784, 0.869] | 0.504 ± 0.020 | 0.554 ± 0.010 |    0.15 | 0.257 ± 0.004     | c308616d0d4d32be976679494164ba06 |
| [0.670, 0.772] | 0.487 ± 0.021 | 0.525 ± 0.011 |    0.22 | 0.229 ± 0.003     | c64ffa14ecfecab4e7ce8c403b4e438f |
| [0.490, 0.603] | 0.455 ± 0.025 | 0.484 ± 0.012 |    0.3  | 0.205 ± 0.003     | 2d8bc173c2bbc3ff07640a56378c3d10 |

cifar (2)

| indicator      | acc (after)   | acc (bef)     |   noise | shard pred acc.   | hash                             |
|--------------- | ------------- | ------------- | ------- | ----------------- | ---------------------------------|
| [0.947, 0.987] | 0.448 ± 0.033 | 0.521 ± 0.019 |    0    | 0.655 ± 0.012     | cca5d44d6ef698eaaebd26797fefd6d4 |
| [0.797, 0.881] | 0.433 ± 0.026 | 0.475 ± 0.015 |    0.2  | 0.587 ± 0.007     | 9727ef9007c04b81d4fdd8dc31523fbc |
| [0.638, 0.744] | 0.419 ± 0.025 | 0.452 ± 0.015 |    0.25 | 0.567 ± 0.006     | fd6e44d21d63050ee933d283291bb72d |
| [0.493, 0.607] | 0.399 ± 0.027 | 0.427 ± 0.015 |    0.3  | 0.550 ± 0.005     | e5b23c4e74a3d1e36e877f0b7dc75aea |

fmnist (6)

| indicator      | acc (after)   | acc (bef)     |   noise | shard pred acc.   | hash                             |
|--------------- | ------------- | ------------- | ------- | ----------------- | ---------------------------------|
| [0.819, 0.899] | 0.849 ± 0.011 | 0.874 ± 0.004 |    0    | 0.248 ± 0.007     | 88366775d1d313ed6523390b93f2eb64 |
| [0.662, 0.765] | 0.838 ± 0.011 | 0.854 ± 0.005 |    0.4  | 0.215 ± 0.005     | 646f56308738e45cd7c92163c655ee30 |
| [0.540, 0.652] | 0.823 ± 0.009 | 0.834 ± 0.006 |    0.6  | 0.198 ± 0.004     | e5152be9a1bb7d5d9b7e0cd4f0491ed3 |
| [0.477, 0.590] | 0.810 ± 0.012 | 0.820 ± 0.006 |    0.75 | 0.190 ± 0.004     | 74471a75cbe765e236d1e118330557c4 |

fmnist (2)

| indicator      | acc (after)   | acc (bef)     |   noise | shard pred acc.   | hash                             |
|--------------- | ------------- | ------------- | ------- | ----------------- | ---------------------------------|
| [0.976, 0.999] | 0.826 ± 0.016 | 0.863 ± 0.006 |     0   | 0.597 ± 0.014     | 5a38e3f38e604447be3f771336bacb00 |
| [0.797, 0.881] | 0.808 ± 0.013 | 0.828 ± 0.007 |     0.5 | 0.555 ± 0.010     | 262b4889a7c914d3090aeab8b0fd0bf4 |
| [0.607, 0.715] | 0.791 ± 0.016 | 0.807 ± 0.008 |     0.7 | 0.538 ± 0.007     | 25186293067603719e741adc8f7572d5 |
| [0.497, 0.610] | 0.763 ± 0.020 | 0.781 ± 0.009 |     1   | 0.523 ± 0.005     | 4acc53823282eda5ef3c416e598a7326 |

mnist (6)

| indicator      | acc (after)   | acc (bef)     |   noise | shard pred acc.   | hash                             |
|--------------- | ------------- | ------------- | ------- | ----------------- | ---------------------------------|
| [0.849, 0.922] | 0.973 ± 0.004 | 0.978 ± 0.002 |     0   | 0.201 ± 0.004     | ad013be6f8ffd0c0451cbfbb35f00ab2 |
| [0.729, 0.824] | 0.965 ± 0.005 | 0.969 ± 0.003 |     0.4 | 0.186 ± 0.003     | 3ccca21040cf98e7195bc7e8e9d6808b |
| [0.583, 0.694] | 0.940 ± 0.009 | 0.945 ± 0.004 |     0.8 | 0.178 ± 0.003     | da2b1393772670f14febb4d53bb892ad |
| [0.493, 0.607] | 0.913 ± 0.018 | 0.923 ± 0.007 |     1.1 | 0.176 ± 0.003     | 88b6ecd38063816e135611ce9bf52237 |

mnist (2)

| indicator      | acc (after)   | acc (bef)     |   noise | shard pred acc.   | hash                             |
|--------------- | ------------- | ------------- | ------- | ----------------- | ---------------------------------|
| [0.927, 0.976] | 0.962 ± 0.007 | 0.971 ± 0.003 |     0   | 0.540 ± 0.006     | a6e06d50a5fd48e6e6e32602b6565406 |
| [0.769, 0.857] | 0.959 ± 0.008 | 0.968 ± 0.003 |     0.8 | 0.534 ± 0.006     | 243ddb045d377a4f42d03ae9aed48e9a |
| [0.587, 0.697] | 0.953 ± 0.007 | 0.962 ± 0.004 |     1.3 | 0.530 ± 0.006     | d4b3eb93d07c8e2e014f59e48e1eac5d |
| [0.497, 0.610] | 0.949 ± 0.008 | 0.957 ± 0.004 |     1.6 | 0.527 ± 0.006     | 6b71a0616160fb739d89c703001548ff |

## Training for Figure 1

The following commands were run to generate the models for Figure 1:

```
python experiment.py --experiment_path configs/cifar_6/0_0.json
python experiment.py --experiment_path configs/cifar_6/0_15.json
python experiment.py --experiment_path configs/cifar_6/0_22.json
python experiment.py --experiment_path configs/cifar_6/0_3.json

python experiment.py --experiment_path configs/cifar_2/0_0.json
python experiment.py --experiment_path configs/cifar_2/0_2.json
python experiment.py --experiment_path configs/cifar_2/0_25.json
python experiment.py --experiment_path configs/cifar_2/0_3.json

python experiment.py --experiment_path configs/fmnist_6/0_0.json
python experiment.py --experiment_path configs/fmnist_6/0_4.json
python experiment.py --experiment_path configs/fmnist_6/0_6.json
python experiment.py --experiment_path configs/fmnist_6/0_75.json

python experiment.py --experiment_path configs/fmnist_2/0_0.json
python experiment.py --experiment_path configs/fmnist_2/0_5.json
python experiment.py --experiment_path configs/fmnist_2/0_7.json
python experiment.py --experiment_path configs/fmnist_2/1_0.json

python experiment.py --experiment_path configs/mnist_6/0_0.json
python experiment.py --experiment_path configs/mnist_6/0_4.json
python experiment.py --experiment_path configs/mnist_6/0_8.json
python experiment.py --experiment_path configs/mnist_6/1_1.json

python experiment.py --experiment_path configs/mnist_2/0_0.json
python experiment.py --experiment_path configs/mnist_2/0_8.json
python experiment.py --experiment_path configs/mnist_2/1_3.json
python experiment.py --experiment_path configs/mnist_2/1_6.json
```

To retrain from scratch, remove all of the result directories under `white-box/results` and run these commands. If you run as-is, it will only add new trials to the existing results, not overwrite them.

## Training for Appendix C.2

Simply execute the following and the given script will run and print the results cited:

```
cd black-box
python main.py
```

