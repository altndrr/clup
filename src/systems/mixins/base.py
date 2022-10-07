"""Module containing an abstract base mixin class."""

from abc import ABC
from typing import Callable, Dict, Optional

import pytorch_lightning as pl  # pylint: disable=unused-import
import torch
from torchmetrics import MetricCollection


class BaseMixin(ABC):  # pylint: disable=too-few-public-methods
    """Implementation of an abstract mixin class."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not hasattr(self, "metrics"):
            self.metrics: Dict[str, MetricCollection]

        if not hasattr(self, "device"):
            self.device: str

        if not hasattr(self, "current_epoch"):
            self.current_epoch: int

        if not hasattr(self, "trainer"):
            self.trainer: Optional["pl.Trainer"] = None

        if not hasattr(self, "log_dict"):
            self.log_dict: Callable

        if not hasattr(self, "predict_step"):
            self.predict_step: Callable

        if not hasattr(self, "forward_all"):
            self.forward_all: Callable

        if not hasattr(self, "criterion"):
            self.criterion: torch.nn.Module
