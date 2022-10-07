"""Module containing the Cluster Match method."""

from typing import Dict, List, Optional, Sequence, Type, Union

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import Subset

from src.models import BaseModel
from src.systems import ClassificationSystem
from src.systems.mixins.mixup import Mixup
from src.systems.mixins.pseudo_labelling import PseudoLabelling


class ClusterMatchSystem(PseudoLabelling, Mixup, ClassificationSystem):
    """Implementation of the Cluster Match method."""

    def __init__(
        self,
        model: Type[BaseModel],
        *args,
        labels: torch.Tensor,
        labeled_samples: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, *args, pseudo_every=1, **kwargs)
        self.labels = labels
        self.labeled_samples: List[int] = labeled_samples or []
        self.train_criterion = self.criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks: List[Callback] = []

        if self.hparams.get("swa"):
            swa_kw = self.hparams.get("swa_kw", {})
            callbacks.append(StochasticWeightAveraging(**swa_kw))

        parent_callbacks = super().configure_callbacks()
        if isinstance(parent_callbacks, Callback):
            parent_callbacks = [parent_callbacks]
        callbacks.extend(parent_callbacks)

        return callbacks

    def subset_train_dataset(self):
        """
        Generate a new training dataset as a subset of the original one. The indices of the train
        samples are the combination of the labeled samples and the pseudo labeled samples.
        """
        indices = list(set(self.labeled_samples))
        train_subset = Subset(self.trainer.datamodule.predict_set, indices)

        assert self.trainer is not None
        assert hasattr(self.trainer, "datamodule")
        datamodule = getattr(self.trainer, "datamodule")
        datamodule.train_set = train_subset

    def on_fit_start(self) -> None:
        super().on_fit_start()

        assert self.trainer is not None
        assert hasattr(self.trainer, "datamodule")
        datamodule = getattr(self.trainer, "datamodule")
        train_dataset = datamodule.predict_set

        assert hasattr(train_dataset, "return_index"), "dataset must return sample index"
        assert getattr(train_dataset, "return_index") is True, "dataset must return sample index"
        assert (
            getattr(self.trainer, "reload_dataloaders_every_n_epochs") > 0
        ), "trainer must reload dataloaders"

        self.subset_train_dataset()

    def on_fit_end(self) -> None:
        super().on_fit_end()

        assert self.trainer is not None
        assert hasattr(self.trainer, "datamodule")
        datamodule = getattr(self.trainer, "datamodule")
        datamodule.train_set = datamodule.predict_set

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        assert len(args) == 2
        assert isinstance(args[1], int)
        batch, batch_idx = args

        indices, (inputs, _) = batch
        labels = self.labels[indices].to(self.device)
        training_outputs = {"inputs": inputs, "labels": labels, "batch_idx": batch_idx}

        # If needed, mixup data.
        if self.mixup_alpha >= 0:
            inputs, labels_a, labels_b = self.mixup_data(inputs, labels)
            training_outputs["inputs"] = inputs
            training_outputs["mixup"] = {
                "labels_a": labels_a,
                "labels_b": labels_b,
            }

        embeddings = self.encoder(inputs)
        outputs = self.classifier(embeddings)

        if isinstance(self.train_criterion, torch.nn.KLDivLoss):
            assert isinstance(outputs, torch.Tensor)
            outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        training_outputs["outputs"] = outputs

        return training_outputs

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        assert isinstance(step_output, Dict)

        inputs = step_output.get("inputs")
        labels = step_output.get("labels")
        batch_idx = step_output.get("batch_idx")
        outputs = step_output.get("outputs")

        mixup_outputs = step_output.get("mixup")
        if mixup_outputs is not None:
            labels_a = mixup_outputs["labels_a"]
            labels_b = mixup_outputs["labels_b"]
            loss = self.mixup_criterion(outputs, labels_a, labels_b, criterion=self.train_criterion)
            loss += self.train_criterion(outputs, labels) * self.kwargs.get("mixaug_alpha")
        else:
            loss = self.train_criterion(outputs, labels)

        # Log a copy of the inputs of the first batch only once.
        if batch_idx == 0:
            assert inputs is not None
            self.log_image("train/inputs", list(inputs), once=True)

        # Evaluate some metrics.
        self.metrics["train"] = self.metrics["train"].to(self.device)
        assert isinstance(labels, torch.Tensor)
        if not torch.is_floating_point(labels):
            metrics = self.metrics["train"](outputs, labels)
            self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, on_step=True)

        return loss
