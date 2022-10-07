"""
cli

Usage:
    cli cluster-match <network> <teacher> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--batch_size=NUM] [--checkpoints] [--checkpoints_dir=PATH]
                      [--criterion_kw=VAL] [--data_dir=PATH] [--freeze=PART] [--gpus=VAL]
                      [--labeled_samples_mode=VAL] [--labeled_samples_mode_kw=VAL]
                      [--labeled_samples_size=VAL] [--lr_scheduler=VAL] [--lr_scheduler_kw=VAL]
                      [--mixup_alpha=VAL] [--mixaug_alpha=VAL] [--mlp_head] [--name=VAL]
                      [--num_workers=NUM] [--strategy=VAL] [--optimizer=VAL] [--optimizer_kw=VAL]
                      [--precision=VAL] [--project=VAL] [--swa] [--swa_kw=VAL] [--wandb]
                      [--weight_norm]
    cli data-prepare <dataset> [--gpus=VAL] [--data_dir=PATH] [--raw_data_dir=PATH]
    cli -h | --help

Options:
    --accelerator=VAL               Accelerator to use with the trainer.
    --batch_size=NUM                Number of samples per batch.
    --checkpoints                   Save checkpoints of the run.
    --checkpoints_dir=PATH          Path to save checkpoints.
    --criterion_kw=VAL              Kwargs of the criterion.
    --data_dir=PATH                 Folder of datasets.
    --epochs=NUM                    Number of epochs.
    --freeze=PART                   Freeze a part of the network.
    --gpus=VAL                      GPUs to use.
    --labeled_samples_mode=VAL      Mode to assign classes to samples.
    --labeled_samples_mode_kw=VAL   Kwargs of the labeling sample mode.
    --labeled_samples_size=VAL      Number of pseudo-labeled samples to select.
    --lr=VAL                        Learning rate.
    --lr_scheduler=VAL              Learning rate scheduler for training.
    --lr_scheduler_kw=VAL           Kwargs of the lr scheduler.
    --mixup_alpha=VAL               Value of the mixup alpha parameter.
    --mixaug_alpha=VAL              Value of the mixaugment alpha parameter.
    --mlp_head                      Replace classification layer with an MLP.
    --name=VAL                      Name of the run.
    --num_workers=NUM               Number of cpu workers to load data.
    --optimizer=VAL                 Name of the optimizer to use.
    --optimizer_kw=VAL              Kwargs of the optimizer.
    --precision=NUM                 Floating point precision.
    --project=VAL                   Name of the wandb project.
    --raw_data_dir=PATH             Folder of raw datasets.
    --strategy=VAL                  Strategy to use with the trainer.
    --swa                           Apply Stochastic Weight Average.
    --swa_kw=VAL                    Kwargs of SWA.
    --wandb                         Log statistics on Weights & Biases.
    --weight_norm                   Apply weight normalisation on the last layer.
    -h --help                       Show this screen.
"""


import json
from typing import Dict, Type

from cli.commands.base import BaseCommand
from cli.commands.classification_test import ClassificationTest
from cli.commands.classification_train import ClassificationTrain
from cli.commands.cluster_match import ClusterMatch
from cli.commands.curriculum_labeling import CurriculumLabeling
from cli.commands.data_prepare import DataPrepare
from cli.commands.source_hypothesis_transfer import SourceHypothesisTransfer
from cli.commands.visualise_umap import VisualiseUMAP

__version__ = "0.1.0"

ARGUMENTS: Dict[str, Dict] = {
    "accelerator": {"type": str},
    "batch_size": {"type": int},
    "dataset": {"type": str},
    "checkpoints": {"type": bool},
    "checkpoints_dir": {"type": str, "default": "./models/checkpoints/"},
    "criterion_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "data_dir": {"type": str, "default": "./datasets/processed"},
    "epochs": {"type": int},
    "freeze": {"type": str},
    "gpus": {"type": list, "parse_fn": lambda x: [int(gpu) for gpu in x.split(",")]},
    "labeled_samples_mode": {"type": str, "default": "normal"},
    "labeled_samples_mode_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "labeled_samples_size": {"type": float},
    "lr": {"type": float},
    "lr_scheduler": {"type": str},
    "lr_scheduler_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "mixup_alpha": {"type": float},
    "mixaug_alpha": {"type": float, "default": 0.0},
    "mlp_head": {"type": bool},
    "name": {"type": str},
    "num_workers": {"type": int},
    "network": {"type": str},
    "optimizer": {"type": str, "default": "sgd"},
    "optimizer_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "precision": {"type": int},
    "project": {"type": str},
    "raw_data_dir": {"type": str, "default": "./datasets/raw/"},
    "strategy": {"type": str},
    "swa": {"type": bool},
    "swa_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "teacher": {"type": str},
    "wandb": {"type": bool},
    "weight_norm": {"type": bool},
    "help": {"type": bool},
    "version": {"type": bool},
}

COMMANDS: Dict[str, Type[BaseCommand]] = {
    "cluster-match": ClusterMatch,
    "data-prepare": DataPrepare,
}
