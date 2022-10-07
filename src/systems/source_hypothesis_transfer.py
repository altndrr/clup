"""Module containing a source hypothesis transfer system."""

from typing import Dict, Optional, Type

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.losses import InfoMaxLoss
from src.models import BaseModel
from src.systems.classification import ClassificationSystem
from src.systems.mixins.pseudo_labelling import PseudoLabelling


class SourceHypothesisTransferSystem(PseudoLabelling, ClassificationSystem):
    """Implementation of a source-hypothesis transfer system."""

    def __init__(
        self,
        model: Type[BaseModel],
        *args,
        lr: float = 0.001,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        weight_entropy: float = 1.0,
        weight_class: float = 0.1,
        weight_auxiliary: float = 0.0,
        **kwargs
    ) -> None:
        """
        Initialise a source-hypothesis transfer system.

        :param model: model to use
        :param lr: learning rate of the system
        :param lr_scheduler: scheduling policy to use for training
        :param weight_entropy: weight value for the entropy loss
        :param weight_class: weight value for the class loss
        :param weight_auxiliary: weight value for the auxiliary loss
        :param pseudo_every: epochs before re-evaluating the pseudo labels
        """
        super().__init__(
            model, *args, lr=lr, weight_decay=weight_decay, lr_scheduler=lr_scheduler, **kwargs
        )
        del self.metrics["train"]

        self.weight_entropy = weight_entropy
        self.weight_class = weight_class
        self.weight_auxiliary = weight_auxiliary

        self.entropy_criterion = InfoMaxLoss()
        self.class_criterion = None

        self.save_hyperparameters(
            "weight_entropy",
            "weight_class",
            "weight_auxiliary",
        )

        if self.weight_class > 0.0:
            self.class_criterion = torch.nn.CrossEntropyLoss()

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        assert isinstance(step_output, Dict)

        inputs = step_output.get("inputs")
        batch_idx = step_output.get("batch_idx")
        outputs = step_output.get("outputs")

        loss = torch.Tensor([0.0]).to(self.device)

        # If needed, evaluate the entropy loss.
        if self.weight_entropy > 0:
            entropy_loss = self.entropy_criterion(outputs)
            entropy_loss *= self.weight_entropy
            loss += entropy_loss
            self.log("train/loss/entropy", entropy_loss, on_epoch=True, on_step=True)

        # If needed, evaluate the class loss.
        if self.weight_class > 0:
            assert self.trainer is not None
            assert self.pseudo_labels is not None
            assert self.class_criterion is not None

            batch_size = self.trainer.datamodule.batch_size  # type: ignore
            batch_start = batch_size * batch_idx
            batch_end = batch_start + batch_size
            pseudo_labels = self.pseudo_labels[batch_start:batch_end]
            pseudo_labels = pseudo_labels.to(self.device)

            class_loss = self.class_criterion(outputs, pseudo_labels)
            class_loss *= self.weight_class
            loss += class_loss
            self.log("train/loss/class", class_loss, on_epoch=True, on_step=True)

        # Log a copy of the inputs of the first batch only once.
        if batch_idx == 0:
            assert inputs is not None
            self.log_image("train/inputs", list(inputs), once=True)

        self.log("train/loss", loss, on_epoch=True, on_step=True)

        return loss

    def on_train_epoch_start(self) -> None:
        if self.weight_class > 0:
            assert self.trainer is not None
            datamodule = getattr(self.trainer, "datamodule")
            dataloader = datamodule.predict_dataloader()
            self.pseudo_labelling(dataloader)
