"""Implementations of auxiliary tasks related to rotation."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from src.datasets.auxiliary import (
    GridRotateImageFolder,
    RelativeRotateImageFolder,
    RotateImageFolder,
)
from src.models.auxiliary.base import AuxiliaryBaseModel


class Rotate(AuxiliaryBaseModel):
    """Implementation of an auxiliary rotation network."""

    criterion = torch.nn.CrossEntropyLoss()
    dataset = RotateImageFolder

    def __init__(self, encoder: Module, in_features: int) -> None:
        auxiliary = torch.nn.Linear(in_features, 4)
        super().__init__(encoder, auxiliary)

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        x = args[0]

        embs = self.encoder(x)
        x = self.auxiliary(embs)

        return x


class RelativeRotate(AuxiliaryBaseModel):
    """Implementation of an auxiliary rotation network with relative inputs."""

    criterion = torch.nn.CrossEntropyLoss()
    dataset = RelativeRotateImageFolder

    def __init__(self, encoder: Module, in_features: int) -> None:
        auxiliary = torch.nn.Linear(2 * in_features, 4)
        super().__init__(encoder, auxiliary)

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)
        x1, x2 = args

        embs1 = self.encoder(x1)
        embs2 = self.encoder(x2)
        x = self.auxiliary(torch.cat((embs1, embs2), 1))

        return x


class GridRotate(AuxiliaryBaseModel):
    """Implementation of an auxiliary rotation network on a single patch in a grid."""

    criterion = torch.nn.CrossEntropyLoss()
    dataset = GridRotateImageFolder

    def __init__(self, encoder: Module, in_features: int) -> None:
        auxiliary = torch.nn.Linear(in_features, 9)
        super().__init__(encoder, auxiliary)

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        x = args[0]

        embs = self.encoder(x)
        x = self.auxiliary(embs)

        return x
