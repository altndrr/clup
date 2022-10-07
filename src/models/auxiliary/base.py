"""Module containing an abstract class for auxiliary networks."""

from abc import ABC, abstractmethod
from typing import Any, Type

import torch
from torch.nn import Module

from src.datasets.auxiliary.base import AuxiliaryBaseDataset


class AuxiliaryBaseModel(ABC, torch.nn.Module):
    """Implementation of an abstract class for auxiliary networks."""

    criterion: Module
    dataset: Type[AuxiliaryBaseDataset]

    def __init__(self, encoder: Module, auxiliary: Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.auxiliary = auxiliary

        if self.encoder is None:
            raise ValueError("property 'encoder' must be not None")

        if self.auxiliary is None:
            raise ValueError("property 'auxiliary' must be not None")

        if self.criterion is None:
            raise ValueError("class property 'criterion' must be not None")

        if self.dataset is None:
            raise ValueError("class property 'dataset' must be not None")

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward step on the auxiliary network."""
        raise NotImplementedError("method not implemented")
