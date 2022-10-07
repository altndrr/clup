"""Module containing classification system."""

from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import Module
from torchmetrics import Accuracy, Metric, MetricCollection, Precision, Recall

from src.models import BaseModel
from src.systems.base import BaseSystem


class ClassificationSystem(BaseSystem):
    """Implementation of a classification system."""

    def __init__(
        self,
        model: Type[BaseModel],
        *args,
        criterion: Optional[Module] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialise a classification system.

        :param model: model to use
        :param criterion: loss to use for classification
        :param lr: learning rate of the system
        :param lr_scheduler: learning rate policy for training
        """
        super().__init__(model, *args, **kwargs)

        self.criterion: Module = torch.nn.CrossEntropyLoss()
        if criterion:
            self.criterion = criterion
        self.save_hyperparameters("lr", "weight_decay", "lr_scheduler")

        metrics: Dict[str, Metric] = {
            "accuracy": Accuracy(),
            "precision": Precision(),
            "recall": Recall(),
        }
        self.metrics["train"] = MetricCollection(metrics, prefix="train/")
        self.metrics["val"] = MetricCollection(metrics, prefix="val/")
        self.metrics["test"] = MetricCollection(metrics, prefix="test/")

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        x = args[0]

        embeddings = self.encoder(x)
        outputs = self.classifier(embeddings)
        return embeddings, outputs

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks: List[Callback] = []

        parent_callbacks = super().configure_callbacks()
        if isinstance(parent_callbacks, Callback):
            parent_callbacks = [parent_callbacks]
        callbacks.extend(parent_callbacks)

        return callbacks

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        assert len(args) == 2
        assert isinstance(args[1], int)
        batch, batch_idx = args

        indices = None
        if isinstance(batch[1], list):
            indices, batch = batch

        inputs, labels = batch

        embeddings = self.encoder(inputs)
        outputs = self.classifier(embeddings)

        training_outputs = {
            "inputs": inputs,
            "labels": labels,
            "batch_idx": batch_idx,
            "outputs": outputs,
        }

        if indices is not None:
            training_outputs["indices"] = indices

        return training_outputs

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        assert isinstance(step_output, Dict)

        inputs = step_output.get("inputs")
        labels = step_output.get("labels")
        batch_idx = step_output.get("batch_idx")
        outputs = step_output.get("outputs")

        loss = self.criterion(outputs, labels)

        # Log a copy of the inputs of the first batch only once.
        if batch_idx == 0:
            assert inputs is not None
            self.log_image("train/inputs", list(inputs), once=True)

        # Evaluate some metrics.
        self.metrics["train"] = self.metrics["train"].to(self.device)
        metrics = self.metrics["train"](outputs, labels)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        assert len(args) == 2
        assert isinstance(args[1], int)
        batch, batch_idx = args

        indices = None
        if isinstance(batch[1], list):
            indices, batch = batch

        inputs, labels = batch

        embeddings = self.encoder(inputs)
        outputs = self.classifier(embeddings)

        validation_outputs = {
            "inputs": inputs,
            "labels": labels,
            "batch_idx": batch_idx,
            "outputs": outputs,
        }

        if indices is not None:
            validation_outputs["indices"] = indices

        return validation_outputs

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        assert len(args) == 1, len(args)
        assert isinstance(args[0], Dict)
        step_output = args[0]

        inputs = step_output.get("inputs")
        labels = step_output.get("labels")
        batch_idx = step_output.get("batch_idx")
        outputs = step_output.get("outputs")

        loss = self.criterion(outputs, labels)

        # Summarise the val accuracy.
        assert self.trainer is not None
        if self.trainer.global_step == 0 and self.trainer.logger is not None:
            wandb.define_metric("val/accuracy", summary="max")

        # Log a copy of the inputs of the first batch only once.
        if batch_idx == 0:
            assert inputs is not None
            self.log_image("val/inputs", list(inputs), once=True)

        # Evaluate some metrics.
        self.metrics["val"] = self.metrics["val"].to(self.device)
        metrics = self.metrics["val"](outputs, labels)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, on_step=True)

        return loss

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        assert len(args) == 2
        assert isinstance(args[1], int)
        batch, batch_idx = args

        indices = None
        if isinstance(batch[1], list):
            indices, batch = batch

        inputs, labels = batch

        embeddings = self.encoder(inputs)
        outputs = self.classifier(embeddings)

        test_outputs = {
            "inputs": inputs,
            "labels": labels,
            "batch_idx": batch_idx,
            "outputs": outputs,
        }

        if indices is not None:
            test_outputs["indices"] = indices

        return test_outputs

    def test_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        assert len(args) == 1, len(args)
        assert isinstance(args[0], Dict)
        step_output = args[0]

        inputs = step_output.get("inputs")
        labels = step_output.get("labels")
        batch_idx = step_output.get("batch_idx")
        outputs = step_output.get("outputs")

        loss = self.criterion(outputs, labels)

        # Log a copy of the inputs of the first batch only once.
        if batch_idx == 0:
            assert inputs is not None
            self.log_image("test/inputs", list(inputs), once=True)

        # Evaluate some metrics.
        self.metrics["test"] = self.metrics["test"].to(self.device)
        metrics = self.metrics["test"](outputs, labels)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.log("test/loss", loss, on_epoch=True, on_step=True)

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        indices = None
        if isinstance(batch[1], list):
            indices, batch = batch

        inputs, labels = batch

        embeddings = self.encoder(inputs)
        outputs = self.classifier(embeddings)

        predict_outputs = {"embeddings": embeddings, "outputs": outputs, "labels": labels}

        if indices is not None:
            predict_outputs["indices"] = indices

        return predict_outputs
