"""Implementations of auxiliary tasks related to generation."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from src.datasets.auxiliary import (
    GridInpaintImageFolder,
    InpaintImageFolder,
    ReconstructImageFolder,
)
from src.losses import MSEVGGLoss
from src.models.auxiliary.base import AuxiliaryBaseModel


class Reconstruct(AuxiliaryBaseModel):
    """Implementation of an auxiliary reconstruction network."""

    criterion = MSEVGGLoss()
    dataset = ReconstructImageFolder

    def __init__(self, encoder: Module, in_features: int) -> None:
        auxiliary = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_features, 256, kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(3, 3, kernel_size=1, stride=1),
            torch.nn.Tanh(),
        )
        super().__init__(encoder, auxiliary)

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        x = args[0]

        embs = self.encoder(x)
        embs = embs.unsqueeze(2).unsqueeze(3)
        x = self.auxiliary(embs)

        return x


class Inpaint(Reconstruct):
    """Implementation of an auxiliary inpaint network."""

    dataset = InpaintImageFolder


class GridInpaint(Reconstruct):
    """Implementation of an auxiliary inpaint network on a single patch in a grid."""

    dataset = GridInpaintImageFolder
