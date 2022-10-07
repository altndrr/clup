"""Module containing experimental callbacks."""

from copy import deepcopy
from inspect import ismethod
from typing import Any, Dict, Iterator, Optional

import pytorch_lightning as pl  # pylint: disable=unused-import
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Metric, MetricCollection, Precision, Recall

from src.data import AuxiliaryDataModule
from src.models.auxiliary import AUXILIARY_MODELS


class AuxiliaryTaskCallback(Callback):
    """Callback to add an auxiliary head to a system."""

    def __init__(
        self,
        task: str,
        alpha: float = 1.0,
        lr: Optional[str] = None,
        lr_weight: float = 1.0,
        finetune_encoder: bool = True,
        warmup: int = -1,
    ) -> None:
        """
        Add an auxiliary task to the system to train.

        :param task: name of the task
        :param alpha: factor to downscale the loss
        :param lr: learning rate to use, equal to the one of the original network if None
        :param lr_weight: lr of the head defined as a multiple of the encoder's lr
        :param finetune_encoder: while training the auxiliary, update also the encoder
        :param warmup: number of epochs where only the auxiliary head is trained
        """
        self.task = task.replace("-", "_")
        self.alpha = alpha
        self.lr = lr
        self.lr_weight = lr_weight
        self.finetune_encoder = finetune_encoder
        self.warmup = warmup

        if AUXILIARY_MODELS.get(self.task) is None:
            raise ValueError(f"auxiliary task not in {list(AUXILIARY_MODELS.keys())}")

        self.dm: AuxiliaryDataModule
        self.optimizer: torch.optim.Optimizer

        metrics: Dict[str, Metric] = {
            "accuracy": Accuracy(),
            "precision": Precision(),
            "recall": Recall(),
        }
        self.train_metrics = MetricCollection(metrics, prefix="train/auxiliary/")
        self.valid_metrics = MetricCollection(metrics, prefix="val/auxiliary/")
        self.test_metrics = MetricCollection(metrics, prefix="test/auxiliary/")

        self.train_iter: Iterator[DataLoader[Any]]
        self.val_iter: Iterator[DataLoader[Any]]
        self.test_iter: Iterator[DataLoader[Any]]

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        self.setup_datamodule(trainer, stage=stage)
        self.setup_auxiliary_head(pl_module)

    def setup_auxiliary_head(self, pl_module: "pl.LightningModule") -> None:
        """
        Setup the pl_module by adding an auxiliary head, with its forward and loss.

        :param pl_module: module to modify
        """
        if hasattr(pl_module, "auxiliary"):
            return

        # Register the auxiliary module on the pl_module.
        auxiliary_cls = AUXILIARY_MODELS.get(self.task)
        assert auxiliary_cls is not None
        assert isinstance(pl_module.classifier, Module)
        assert isinstance(pl_module.encoder, Module)
        in_features = pl_module.classifier.in_features
        assert isinstance(in_features, int)
        pl_module.auxiliary = auxiliary_cls(pl_module.encoder, in_features)

        # Register the metrics on the pl_module.
        pl_module.train_aux_metrics = self.train_metrics
        pl_module.valid_aux_metrics = self.valid_metrics
        pl_module.test_aux_metrics = self.test_metrics

    def setup_datamodule(self, trainer: "pl.Trainer", stage: Optional[str] = None) -> None:
        """
        Setup the datamodule given the trainer. This process overwrites the old datamodule
        if needed.

        :param trainer: the trainer used
        """
        assert hasattr(trainer, "datamodule")
        datamodule = getattr(trainer, "datamodule")
        trainer_dataset = datamodule.dataset.__name__

        # Re-init the datamodule only if the dataset has changed.
        if self.dm is not None:
            current_dataset = self.dm.dataset.__name__
            if trainer_dataset == current_dataset:
                return

        self.dm = AuxiliaryDataModule(
            trainer_dataset,
            self.task,
            batch_size=datamodule.batch_size,
            num_workers=datamodule.num_workers,
        )
        self.dm.setup()

    def setup_optimizers(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Setup the optimizers for the auxiliary task from a clone of the
        optimizers of the pl_module.

        :param trainer: trainer used
        :param pl_module: module containing the original optimizers
        """
        # Evaluate the lr for the auxiliary head.
        lr_encoder = self.lr if self.lr else trainer.optimizers[0].param_groups[0]["lr"]
        lr_auxiliary = lr_encoder * self.lr_weight

        # Create the param groups.
        param_groups = []
        assert isinstance(pl_module.encoder, Module)
        assert isinstance(pl_module.auxiliary, Module)
        if self.finetune_encoder:
            params = pl_module.encoder.parameters()
            param_groups.append({"params": params, "lr": lr_encoder})
        params = pl_module.auxiliary.parameters()
        param_groups.append({"params": params, "lr": lr_auxiliary})

        # Clone the optimizer and overwrite the params group.
        self.optimizer = deepcopy(trainer.optimizers[0])

        # Delete all but the first param group.
        while len(self.optimizer.param_groups) > 1:
            del self.optimizer.param_groups[-1]

        # Overwrite the param groups.
        for i, group in enumerate(param_groups):
            if i == 0:
                self.optimizer.param_groups[i]["params"] = group["params"]
                self.optimizer.param_groups[i]["lr"] = group["lr"]
            else:
                self.optimizer.add_param_group(group)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.setup_optimizers(trainer, pl_module)

        # If warmup, suspend learning on the base system.
        if self.warmup > 0:
            assert ismethod(pl_module.set_learning)
            pl_module.set_learning(False)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_iter = iter(self.dm.train_dataloader())

        # If warmup is over, restart learning on the base system.
        if self.warmup == pl_module.current_epoch:
            assert ismethod(pl_module.set_learning)
            pl_module.set_learning(True)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        batch = next(self.train_iter)
        assert len(batch) == 2
        inputs, labels = batch

        # Standardise inputs to list. This permits to work with auxiliary forwards
        # that have different numbers of arguments.
        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs = [inp.to(pl_module.device) for inp in inputs]
        labels = labels.to(pl_module.device)

        assert isinstance(pl_module.auxiliary, Module)
        assert isinstance(pl_module.auxiliary.criterion, Module)
        aux_outputs = pl_module.auxiliary(*inputs)
        loss = pl_module.auxiliary.criterion(aux_outputs, labels) * self.alpha

        # Log a copy of the inputs for the epoch and the first batch.
        if batch_idx == 0:
            assert ismethod(pl_module.log_image)
            pl_module.log_image("train/auxiliary/inputs", list(inputs[0]), once=True)

        if isinstance(pl_module.auxiliary.criterion, torch.nn.CrossEntropyLoss):
            metrics = self.train_metrics(aux_outputs, labels)
            pl_module.log_dict(metrics, on_epoch=True)
            pl_module.log("train/auxiliary/loss", loss, on_epoch=True, on_step=True)

        # Manually optimize for the auxiliary loss.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.val_iter = iter(self.dm.val_dataloader())

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = next(self.val_iter)
        assert len(batch) == 2
        inputs, labels = batch

        # Standardise inputs to list. This permits to work with auxiliary forwards
        # that have different numbers of arguments.
        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs = [inp.to(pl_module.device) for inp in inputs]
        labels = labels.to(pl_module.device)

        assert isinstance(pl_module.auxiliary, Module)
        assert isinstance(pl_module.auxiliary.criterion, Module)
        aux_outputs = pl_module.auxiliary(*inputs)
        loss = pl_module.auxiliary.criterion(aux_outputs, labels) * self.alpha

        # Log a copy of the inputs for the epoch and the first batch.
        if batch_idx == 0:
            assert ismethod(pl_module.log_image)
            pl_module.log_image("val/auxiliary/inputs", list(inputs[0]), once=True)

            # If available, save copies of the outputs and labels.
            if not isinstance(pl_module.auxiliary.criterion, torch.nn.CrossEntropyLoss):
                pl_module.log_image("val/auxiliary/outputs", list(aux_outputs), once=True)
                pl_module.log_image("val/auxiliary/labels", list(labels), once=True)

        # If needed, evaluate classification metrics.
        if isinstance(pl_module.auxiliary.criterion, torch.nn.CrossEntropyLoss):
            metrics = self.train_metrics(aux_outputs, labels)
            pl_module.log_dict(metrics, on_epoch=True)
            pl_module.log("val/auxiliary/loss", loss, on_epoch=True, on_step=True)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_iter = iter(self.dm.test_dataloader())

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = next(self.test_iter)
        assert len(batch) == 2
        inputs, labels = batch

        inputs = inputs.to(pl_module.device)
        labels = labels.to(pl_module.device)

        # Standardise inputs to list. This permits to work with auxiliary forwards
        # that have different numbers of arguments.
        if not isinstance(inputs, list):
            inputs = [inputs]

        assert isinstance(pl_module.auxiliary, Module)
        assert isinstance(pl_module.auxiliary.criterion, Module)
        aux_outputs = pl_module.auxiliary(*inputs)
        loss = pl_module.auxiliary.criterion(aux_outputs, labels) * self.alpha

        # Log a copy of the inputs for the epoch and the first batch.
        if batch_idx == 0:
            assert ismethod(pl_module.log_image)
            pl_module.log_image("test/auxiliary/inputs", list(inputs[0]), once=True)

            # If available, save copies of the outputs and labels.
            if not isinstance(pl_module.auxiliary.criterion, torch.nn.CrossEntropyLoss):
                pl_module.log_image("test/auxiliary/outputs", list(aux_outputs), once=True)
                pl_module.log_image("test/auxiliary/labels", list(labels), once=True)

        # If needed, evaluate classification metrics.
        if isinstance(pl_module.auxiliary.criterion, torch.nn.CrossEntropyLoss):
            metrics = self.train_metrics(aux_outputs, labels)
            pl_module.log_dict(metrics, on_epoch=True)
            pl_module.log("test/auxiliary/loss", loss, on_epoch=True, on_step=True)
