"""Module containing an abstract base system module."""

import os
import tempfile
import types
from abc import ABC
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from rich.progress import track
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

import wandb
from src.decorators import catch
from src.models.base import BaseModel


class BaseSystem(ABC, pl.LightningModule):
    """Implementation of a base system."""

    def __init__(self, model: Type[BaseModel], **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs
        self.save_hyperparameters(kwargs)

        self.encoder = model.encoder
        self.classifier = model.classifier

        self._learning_methods: Dict[str, Callable]
        self._logged_images: List[str] = []

        self.metrics: Dict[str, MetricCollection] = {}

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        assert isinstance(args[0], torch.Tensor)
        x = args[0]

        embeddings = self.encoder(x)
        outputs = self.classifier(embeddings)
        return embeddings, outputs

    def forward_all(self, dataloader: DataLoader) -> Any:
        """
        Apply the forward operation on a complete dataloader.

        :param dataloader: data to use for forward
        """
        embeddings = torch.Tensor()
        outputs = torch.Tensor()
        labels = torch.Tensor()
        indices = torch.Tensor()

        for _, batch in self.make_loop(dataloader, description="Forward"):
            if len(batch) == 2 and isinstance(batch[1], tuple):
                samples_idx, batch = batch
                indices = torch.cat((indices, samples_idx), dim=0)

            inputs, labels_ = batch
            embs, outs = self.forward(inputs)
            embeddings = torch.cat((embeddings, embs.detach().cpu()), dim=0)
            outputs = torch.cat((outputs, outs.detach().cpu()), dim=0)
            labels = torch.cat((labels, labels_.detach().cpu()), dim=0)

        forward_outputs = {"embeddings": embeddings, "outputs": outputs, "labels": labels}
        if len(indices) > 0:
            forward_outputs["indices"] = indices

        return forward_outputs

    def make_loop(self, dataloader: DataLoader, description: str = "Working...") -> Iterable:
        """
        Create a custom loop on a dataloader.

        :param dataloader: dataloader to iterate over
        :param description: text to print
        """
        dataset = dataloader.dataset
        return_index = hasattr(dataset, "return_index") and getattr(dataset, "return_index")

        progress_bar = track(
            enumerate(dataloader),
            transient=True,
            total=len(dataloader),
            description=description,
        )

        for batch_idx, batch in progress_bar:
            if return_index:
                indices, batch = batch
                inputs, labels = batch
                batch = indices, (inputs.to(self.device), labels.to(self.device))
            else:
                inputs, labels = batch
                batch = (inputs.to(self.device), labels.to(self.device))

            yield batch_idx, batch

    @property
    def learnable_params(self) -> List[Dict[str, Sequence[Any]]]:
        """Returns the learnable parameters in the system."""
        learnable_params = []

        for module_name in ["encoder", "classifier"]:
            module = getattr(self, module_name)
            params = [param for param in module.parameters() if param.requires_grad]
            if len(params) > 0:
                learnable_params.append({"name": module_name, "params": params})

        return learnable_params

    def on_fit_end(self) -> None:
        self._logged_images = []

    def on_test_epoch_end(self) -> None:
        self._logged_images = []

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks: List[Callback] = []

        if self.hparams.get("checkpoints"):
            name = wandb.run.id if wandb.run is not None else tempfile.mktemp(prefix="", dir="")
            checkpoints_dir = self.hparams.get("checkpoints_dir")
            assert checkpoints_dir is not None
            os.makedirs(checkpoints_dir, exist_ok=True)
            dirpath = os.path.join(checkpoints_dir, name)

            callbacks.append(
                ModelCheckpoint(
                    filename="{epoch}",
                    dirpath=dirpath,
                    monitor="val/accuracy",
                    mode="max",
                )
            )

        if self.hparams.get("wandb"):
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        return callbacks

    def configure_optimizers(self) -> Union[Type[Optimizer], Dict]:
        optimizer_cls: Optional[Callable] = None
        lr_scheduler = None

        assert isinstance(self.hparams, dict)

        optimizer_name = self.hparams.get("optimizer")
        assert optimizer_name is not None, "no optimizer specified"
        if optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        elif optimizer_name == "sgd":
            optimizer_cls = torch.optim.SGD
        else:
            raise ValueError("invald optimizer name")

        assert optimizer_cls is not None
        optimizer: Type[Optimizer] = optimizer_cls(
            self.learnable_params,
            lr=self.hparams["lr"],
            **self.hparams.get("optimizer_kw", {}),
        )

        lr_scheduler_name = self.hparams.get("lr_scheduler")
        if not lr_scheduler_name:
            return optimizer

        lr_scheduler_kw = self.hparams.get("lr_scheduler_kw", {})
        if lr_scheduler_name == "cosine_annealing":
            assert "T_max" in lr_scheduler_kw
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **lr_scheduler_kw)
        elif lr_scheduler_name == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **lr_scheduler_kw)
        elif lr_scheduler_name == "onecycle":
            assert "steps_per_epoch" in lr_scheduler_kw
            assert "max_epochs" in lr_scheduler_kw

            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams["lr"],
                **lr_scheduler_kw,
            )
        else:
            raise ValueError("invalid scheduler name")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def set_requires_grad(self, module_name: str, value: bool) -> None:
        """
        Change the gradient requirements for a module of the system.

        :param module_part: module to modify
        :param value: value of the requires_grad
        """
        if not hasattr(self, module_name):
            raise ValueError("not a valid module name")

        module = getattr(self, module_name)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{module_name} must be a torch.nn.Module")

        for param in module.parameters():
            param.requires_grad = value

    def set_learning(self, is_learning: bool) -> None:
        """
        Enable/disable automatic optimization and swap in/out the
        {training/validation}_step and {training/validation}_step_end
        to freeze learning.

        :param is_learning: enable or disable learning
        """
        self.automatic_optimization = is_learning

        # Save a copy of the original methods.
        if not self._learning_methods:
            self._learning_methods["training_step"] = self.training_step
            self._learning_methods["training_step_end"] = self.training_step_end
            self._learning_methods["validation_step"] = self.validation_step
            self._learning_methods["validation_step_end"] = self.validation_step_end

        def skip_method(self, *args, **kwargs):
            return None

        # Swap in/out the original methods.
        if is_learning:
            self.training_step = self._learning_methods["training_step"]  # type: ignore
            self.training_step_end = self._learning_methods["training_step_end"]  # type: ignore
            self.validation_step = self._learning_methods["validation_step"]  # type: ignore
            self.validation_step_end = self._learning_methods["validation_step_end"]  # type: ignore
        else:
            self.training_step = types.MethodType(skip_method, self)  # type: ignore
            self.training_step_end = types.MethodType(skip_method, self)  # type: ignore
            self.validation_step = types.MethodType(skip_method, self)  # type: ignore
            self.validation_step_end = types.MethodType(skip_method, self)  # type: ignore

    @catch(notimplemented=None)
    def log_image(
        self,
        key: str,
        images: torch.Tensor,
        step: Optional[bool] = None,
        once: bool = False,
        **kwargs,
    ) -> None:
        """
        Log an image if the trainer contains the a logger with the functionality.

        :param key: name of the logged item
        :param images: samples to log
        :param once: log the images only one time per loop.
        """
        assert self.trainer is not None

        if not self.trainer.logger:
            return

        if once and key in self._logged_images:
            return

        if hasattr(self.trainer.logger, "log_image"):
            self.trainer.logger.log_image(key, images, step, **kwargs)  # type: ignore
            self._logged_images.append(key)
