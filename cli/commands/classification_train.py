"""Implementation of a command to train classification systems."""

from typing import Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from cli import config
from cli.commands.base import BaseCommand
from src.data.image import ImageDataModule
from src.datasets import DATASETS
from src.models.utils import make_model
from src.systems import ClassificationSystem


class ClassificationTrain(BaseCommand):
    """Train a neural network for a classification task."""

    def run(self) -> None:
        model = make_model(
            self.options.get("network"),
            num_classes=len(DATASETS[self.options.get("dataset")].labels),
            weight_norm=self.options.get("weight_norm"),
        )

        # Define the logger.
        logger: Union[bool, WandbLogger] = False
        if self.options.get("wandb"):
            logger = WandbLogger(
                name=self.options.get("name"),
                project=self.options.get("project"),
                save_dir="wandb",
            )

            # Watch the model and log the options.
            logger.watch(model, log="gradients", log_freq=100)
            logger.log_hyperparams(self.options)

        # Get the trainer, the datamodule and the system.
        trainer = Trainer(
            accelerator=self.options.get("accelerator"),
            callbacks=[RichModelSummary(), RichProgressBar()],
            deterministic=config.get("deterministic"),
            enable_checkpointing=self.options.get("checkpoints"),
            gpus=self.options.get("gpus"),
            max_epochs=self.options.get("epochs"),
            logger=logger,
            precision=self.options.get("precision"),
            strategy=self.options.get("strategy"),
        )
        dm = ImageDataModule(
            self.options.get("dataset"),
            self.options.get("data_dir"),
            batch_size=self.options.get("batch_size"),
            num_workers=self.options.get("num_workers"),
        )
        criterion = torch.nn.CrossEntropyLoss(**self.options.get("criterion_kw", {}))
        system = ClassificationSystem(model, criterion=criterion, **self.options)

        # If required, freeze part of the system.
        if self.options.get("freeze"):
            system.set_requires_grad(self.options.get("freeze"), False)

        # Fit the system to the data.
        trainer.fit(system, dm)
