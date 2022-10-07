# Cluster-level pseudo-labelling for source-free cross-domain facial expression recognition

This repository contains the code for the paper **Cluster-level pseudo-labelling for source-free cross-domain facial expression recognition**, accepted at BVMC 2022.

## Setup

To setup the repository, you can use poetry and run the following commands:

```sh
poetry install
```

If poetry is not available, the list of packages used in the project can be found in the `pyproject.toml` file,
under the `tool.poetry.dependencies` section.

To then enter the virtual environment, you can manually source the environment, or run:

```sh
poetry shell
```

### Configuration file

We provide a configuration file `config.yaml` that contains the default values for the parameters used in the
command-line interface. You can modify this file to change the default values.

Importantly, you should update the `raw_data_dir` parameter to point to the directory where the datasets are stored.
Also, you can modify the `data_dir` parameter to point to the directory where the preprocessed data will be stored.
Configuration attributes are overwritten by the command-line arguments.

## Usage

The repository is split into two main parts: the cli and the library. To interact with the library, use the cli.
In the `./models` directory, you can find the pretrained models for the source (`./models/source/pretrained`) and the target models (`./models/target/pretrained`).
In addition, in `./models/source/` you can find the trained models on the source datasets.

The complete list of commands can be found below. Note that you can type `python -m cli --help` to see the list of commands and arguments.

```sh
cli

Usage:
    cli classification-test <network> <dataset> [--accelerator=VAL] [--batch_size=NUM]
                      [--data_dir=PATH] [--gpus=VAL] [--num_workers=NUM] [--precision=VAL]
                      [--strategy=VAL]
    cli classification-train <network> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--batch_size=NUM] [--checkpoints] [--checkpoints_dir=PATH]
                      [--criterion_kw=VAL] [--data_dir=PATH] [--freeze=PART] [--gpus=VAL]
                      [--lr_scheduler=VAL] [--lr_scheduler_kw=VAL] [--name=VAL] [--num_workers=NUM]
                      [--optimizer=VAL] [--optimizer_kw=VAL] [--precision=VAL] [--project=VAL]
                      [--strategy=VAL] [--wandb] [--weight_norm]
    cli clup <network> <teacher> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--batch_size=NUM] [--checkpoints] [--checkpoints_dir=PATH]
                      [--criterion_kw=VAL] [--data_dir=PATH] [--freeze=PART] [--gpus=VAL]
                      [--labeled_samples_mode=VAL] [--labeled_samples_mode_kw=VAL]
                      [--labeled_samples_size=VAL] [--lr_scheduler=VAL] [--lr_scheduler_kw=VAL]
                      [--mixup_alpha=VAL] [--mixaug_alpha=VAL] [--mlp_head] [--name=VAL]
                      [--num_workers=NUM] [--strategy=VAL] [--optimizer=VAL] [--optimizer_kw=VAL]
                      [--precision=VAL] [--project=VAL] [--swa] [--swa_kw=VAL] [--wandb]
                      [--weight_norm]
    cli data-prepare <dataset> [--gpus=VAL] [--data_dir=PATH] [--raw_data_dir=PATH]
```

### Dataset preparation

To prepare the datasets, you have to manually request and download them and place them in the `raw_data_dir` directory.
After, you can run the following command to preprocess the datasets:

```sh
# Note: replace {dataset} with one in [afe, expw, fer2013, rafdb].
python -m cli prepare-data {dataset}
```

### Source training

The source models are already available in the `./models/source` directory. If you want to train them yourself, you can run the following command:

```sh
# Note: replace {dataset} with the dataset you want to train on.
python -m cli classification-train \
    ./models/source/pretrained/resnet18.pt \
    {dataset} \
    --epochs 100 \
    --lr 0.01 \
    --criterion_kw '{"label_smoothing": 0.1}' \
    --lr_scheduler exponential \
    --lr_scheduler_kw '{"gamma": 0.99}' \
    --weight_norm
```

Note that the above command is an example, and the arguments do not perfectly match those used to train the source models.

You can test the source models on the target datasets using the following command:

```sh
# Note: replace {source-model} with your custom source model, or with one available in `./models/source`.
# Note: replace {dataset} with the dataset you want to test on.
python -m cli classification-test {source-model} {dataset}
```

## Target training (CluP)

To train the target models, you can run one of the following commands:

```sh
# Train the AFE->ExpW model.
python -m cli clup \
    ./models/target/pretrained/swav-1000ep-ExpW-5000prototypes.ckpt \
    ./models/source/resnet18_afe-0.9286.pt \
    expw \
    --epochs 50 \
    --lr 0.1 \
    --batch_size 32 \
    --optimizer sgd \
    --optimizer_kw "{\"weight_decay\": 0.0005, \"momentum\": 0.9, \"nesterov\": true}" \
    --lr_scheduler cosine_annealing \
    --lr_scheduler_kw "{\"T_max\": 50}" \
    --mixup_alpha 0.2 \
    --labeled_samples_mode purity \
    --labeled_samples_mode_kw "{\"num_prototypes\": 1000, \"drop_outliers\": false, \"iter_clusters\": 1}" \
    --labeled_samples_size 0.1
```

```sh
# Train the AFE->FER2013 model.
python -m cli clup \
    ./models/target/pretrained/swav-1000ep-FER2013-5000prototypes.ckpt \
    ./models/source/resnet18_afe-0.9286.pt \
    fer2013 \
    --epochs 50 \
    --lr 0.1 \
    --batch_size 32 \
    --optimizer sgd \
    --optimizer_kw "{\"weight_decay\": 0.0005, \"momentum\": 0.9, \"nesterov\": true}" \
    --lr_scheduler cosine_annealing \
    --lr_scheduler_kw "{\"T_max\": 50}" \
    --mixup_alpha 0.2 \
    --labeled_samples_mode purity \
    --labeled_samples_mode_kw "{\"num_prototypes\": 250, \"drop_outliers\": false, \"iter_clusters\": 1}" \
    --labeled_samples_size 0.1
```

```sh
# Train the RAFDB->ExpW model.
python -m cli clup \
    ./models/target/pretrained/swav-1000ep-ExpW-5000prototypes.ckpt \
    ./models/source/resnet18_rafdb-0.839.pt \
    expw \
    --epochs 50 \
    --lr 0.1 \
    --batch_size 32 \
    --optimizer sgd \
    --optimizer_kw "{\"weight_decay\": 0.0005, \"momentum\": 0.9, \"nesterov\": true}" \
    --lr_scheduler cosine_annealing \
    --lr_scheduler_kw "{\"T_max\": 50}" \
    --mixup_alpha 0.2 \
    --labeled_samples_mode purity \
    --labeled_samples_mode_kw "{\"num_prototypes\": 250, \"drop_outliers\": false, \"iter_clusters\": 1}" \
    --labeled_samples_size 0.3
```

```sh
# Train the RAFDB->FER2013 model.
python -m cli clup \
    ./models/target/pretrained/swav-1000ep-FER2013-5000prototypes.ckpt \
    ./models/source/resnet18_rafdb-0.839.pt \
    fer2013 \
    --epochs 50 \
    --lr 0.1 \
    --batch_size 32 \
    --optimizer sgd \
    --optimizer_kw "{\"weight_decay\": 0.0005, \"momentum\": 0.9, \"nesterov\": true}" \
    --lr_scheduler cosine_annealing \
    --lr_scheduler_kw "{\"T_max\": 50}" \
    --mixup_alpha 0.2 \
    --labeled_samples_mode purity \
    --labeled_samples_mode_kw "{\"num_prototypes\": 1000, \"drop_outliers\": false, \"iter_clusters\": 1}" \
    --labeled_samples_size 0.2
```

## Common issues

### AttributeError when running CLI

If you the the error `AttributeError: module 'distutils' has no attribute 'version'`,
you can fix it by running:

```sh
pip install setuptools==59.5.0
```

### GPUs requiring cu11.3

For GPUs requiring cu11.3, overwrite the torch installation with pip

```sh
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113 --force
```
