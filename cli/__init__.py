"""
cli

Usage:
    cli classification-test  <network> <dataset> [--accelerator=VAL] [--batch_size=NUM]
                      [--data_dir=PATH] [--gpus=VAL] [--num_workers=NUM] [--precision=VAL]
                      [--strategy=VAL]
    cli classification-train <network> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--auxiliary_task=VAL] [--auxiliary_warmup=VAL] [--batch_size=NUM]
                      [--checkpoints] [--checkpoints_dir=PATH] [--criterion_kw=VAL]
                      [--data_dir=PATH] [--freeze=PART] [--gpus=VAL] [--lr_scheduler=VAL]
                      [--lr_scheduler_kw=VAL] [--name=VAL] [--num_workers=NUM] [--optimizer=VAL]
                      [--optimizer_kw=VAL] [--precision=VAL] [--project=VAL] [--strategy=VAL]
                      [--wandb] [--weight_auxiliary=VAL] [--weight_norm]
    cli cluster-match <network> <teacher> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--batch_size=NUM] [--checkpoints] [--checkpoints_dir=PATH]
                      [--criterion_kw=VAL] [--data_dir=PATH] [--freeze=PART] [--gpus=VAL]
                      [--labeled_samples_mode=VAL] [--labeled_samples_mode_kw=VAL]
                      [--labeled_samples_size=VAL] [--lr_scheduler=VAL] [--lr_scheduler_kw=VAL]
                      [--mixup_alpha=VAL] [--mixaug_alpha=VAL] [--mlp_head] [--name=VAL]
                      [--num_workers=NUM] [--strategy=VAL] [--optimizer=VAL] [--optimizer_kw=VAL]
                      [--precision=VAL] [--project=VAL] [--swa] [--swa_kw=VAL] [--wandb]
                      [--weight_norm]
    cli curriculum-labeling <network> <teacher> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--batch_size=NUM] [--checkpoints] [--checkpoints_dir=PATH]
                      [--criterion_kw=VAL] [--data_dir=PATH] [--freeze=PART] [--gpus=VAL]
                      [--iter_labeled=VAL] [--iter_unlabeled=VAL] [--labeled_samples_mode=VAL]
                      [--labeled_samples_mode_kw=VAL] [--labeled_samples_size=VAL]
                      [--lr_scheduler=VAL] [--lr_scheduler_kw=VAL] [--mixup_alpha=VAL] [--name=VAL]
                      [--num_workers=NUM] [--strategy=VAL] [--optimizer=VAL] [--optimizer_kw=VAL]
                      [--precision=VAL] [--project=VAL] [--swa] [--swa_kw=VAL] [--wandb]
                      [--weight_norm]
    cli data-prepare <dataset> [--gpus=VAL] [--data_dir=PATH] [--raw_data_dir=PATH]
    cli data-visualise <network> <dataset> --split=[train | test] [--accelerator=VAL]
                      [--batch_size=NUM] [--data_dir=PATH] [--gpus=VAL] [--images_dir=PATH]
                      [--num_workers=NUM] [--precision=VAL] [--strategy=VAL]
    cli source-hypothesis-transfer <network> <dataset> --epochs=NUM --lr=VAL [--accelerator=VAL]
                      [--auxiliary_task=VAL] [--auxiliary_warmup=VAL] [--batch_size=NUM]
                      [--checkpoints] [--checkpoints_dir=PATH] [--data_dir=PATH] [--freeze=PART]
                      [--gpus=VAL] [--lr_scheduler=VAL] [--lr_scheduler_kw=VAL] [--name=VAL]
                      [--num_workers=NUM] [--optimizer=VAL] [--optimizer_kw=VAL] [--precision=VAL]
                      [--project=VAL] [--pseudo_every=NUM] [--strategy=VAL] [--wandb]
                      [--weight_auxiliary=VAL] [--weight_class=VAL] [--weight_entropy=VAL]
    cli -h | --help
    cli --version

Options:
    --accelerator=VAL               Accelerator to use with the trainer.
    --auxiliary_task=VAL            Task to perform with an auxiliary head.
    --auxiliary_warmup=VAL          Number of epochs to train only the auxiliary.
    --batch_size=NUM                Number of samples per batch.
    --checkpoints                   Save checkpoints of the run.
    --checkpoints_dir=PATH          Path to save checkpoints.
    --criterion_kw=VAL              Kwargs of the criterion.
    --data_dir=PATH                 Folder of datasets.
    --epochs=NUM                    Number of epochs.
    --freeze=PART                   Freeze a part of the network.
    --gpus=VAL                      GPUs to use.
    --images_dir=PATH               Path to save images.
    --iter_labeled=VAL              Number of labeled fit loops.
    --iter_unlabeled=VAL            Number of pseudo-labeled fit loops.
    --labeled_samples_mode=VAL      Mode to assign classes to samples.
    --labeled_samples_mode_kw=VAL   Kwargs of the labeling sample mode.
    --labeled_samples_size=VAL      Number of pseudo-labeled samples to select.
    --lr=VAL                        Learning rate.
    --lr_scheduler=VAL              Learning rate scheduler for training.
    --lr_scheduler_lw=VAL           Kwargs of the lr scheduler.
    --mixup_alpha=VAL               Value of the mixup alpha parameter.
    --mixaug_alpha=VAL              Value of the mixaugment alpha parameter.
    --mlp_head                      Replace classification layer with an MLP.
    --name=VAL                      Name of the run.
    --num_workers=NUM               Number of cpu workers to load data.
    --optimizer=VAL                 Name of the optimizer to use.
    --optimizer_kw=VAL              Kwargs of the optimizer.
    --precision=NUM                 Floating point precision.
    --project=VAL                   Name of the wandb project.
    --pseudo_every=NUM              Epochs between two evaluation of pseudo labels.
    --raw_data_dir=PATH             Folder of raw datasets.
    --split=VAL                     Train or test split.
    --strategy=VAL                  Strategy to use with the trainer.
    --swa                           Apply Stochastic Weight Average.
    --swa_kw=VAL                    Kwargs of SWA.
    --wandb                         Log statistics on Weights & Biases.
    --weight_auxiliary=VAL          Weight of the auxiliary loss.
    --weight_class=VAL              Weight of the classification loss.
    --weight_entropy=VAL            Weight of the entropy loss.
    --weight_norm                   Apply weight normalisation on the last layer.
    -h --help                       Show this screen.
    --version                       Show version.
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
    "auxiliary_task": {"type": str},
    "auxiliary_warmup": {"type": int},
    "batch_size": {"type": int},
    "dataset": {"type": str},
    "checkpoints": {"type": bool},
    "checkpoints_dir": {"type": str, "default": "./models/checkpoints/"},
    "criterion_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "data_dir": {"type": str, "default": "./datasets/processed"},
    "epochs": {"type": int},
    "freeze": {"type": str},
    "gpus": {"type": list, "parse_fn": lambda x: [int(gpu) for gpu in x.split(",")]},
    "images_dir": {"type": str, "default": "./media/images/"},
    "iter_labeled": {"type": int},
    "iter_unlabeled": {"type": int},
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
    "pseudo_every": {"type": int},
    "raw_data_dir": {"type": str, "default": "./datasets/raw/"},
    "split": {"type": str},
    "strategy": {"type": str},
    "swa": {"type": bool},
    "swa_kw": {"type": dict, "parse_fn": lambda x: json.loads(str(x))},
    "teacher": {"type": str},
    "wandb": {"type": bool},
    "weight_auxiliary": {"type": float},
    "weight_class": {"type": float},
    "weight_entropy": {"type": float},
    "weight_norm": {"type": bool},
    "help": {"type": bool},
    "version": {"type": bool},
}

COMMANDS: Dict[str, Type[BaseCommand]] = {
    "classification-test": ClassificationTest,
    "classification-train": ClassificationTrain,
    "cluster-match": ClusterMatch,
    "curriculum-labeling": CurriculumLabeling,
    "data-prepare": DataPrepare,
    "data-visualise": VisualiseUMAP,
    "source-hypothesis-transfer": SourceHypothesisTransfer,
}
