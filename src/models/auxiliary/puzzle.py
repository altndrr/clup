"""Implementations of auxiliary tasks related to puzzles."""

from typing import Any, List

import torch
from torch import Tensor
from torch.nn import Module

from src.datasets.auxiliary import GridPuzzleImageFolder, RelativeGridPuzzleImageFolder
from src.losses import CrossEntropyListLoss
from src.models.auxiliary.base import AuxiliaryBaseModel


class GridPuzzle(AuxiliaryBaseModel):
    """Implementation of an auxiliary puzzle network on patches of a grid."""

    criterion = CrossEntropyListLoss()
    dataset = GridPuzzleImageFolder

    def __init__(self, encoder: Module, in_features: int) -> None:
        auxiliary = torch.nn.ModuleList([torch.nn.Linear(in_features, 9) for _ in range(9)])
        super().__init__(encoder, auxiliary)

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        x = args[0]

        embs = self.encoder(x)

        # Concat outputs to have (batch, preds, heads).
        assert isinstance(self.auxiliary, torch.nn.ModuleList)
        xs: List[Tensor] = [head(embs) for head in self.auxiliary]
        xs = [out.unsqueeze(2) for out in xs]
        x = torch.cat(xs, dim=2)

        return x


class RelativeGridPuzzle(AuxiliaryBaseModel):
    """
    Implementation of an auxiliary puzzle network on patches of a grid with
    relative inputs.
    """

    criterion = CrossEntropyListLoss()
    dataset = RelativeGridPuzzleImageFolder

    def __init__(self, encoder: Module, in_features: int) -> None:
        auxiliary = torch.nn.ModuleList([torch.nn.Linear(in_features * 2, 9) for _ in range(9)])
        super().__init__(encoder, auxiliary)

    def forward(self, *args, **kwargs) -> Any:
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)
        x1, x2 = args

        embs1 = self.encoder(x1)
        embs2 = self.encoder(x2)

        # Concat outputs to have (batch, preds, heads).
        assert isinstance(self.auxiliary, torch.nn.ModuleList)
        x = [head(torch.cat((embs1, embs2), 1)) for head in self.auxiliary]
        x = [out.unsqueeze(2) for out in x]
        x_cat = torch.cat(x, dim=2)

        return x_cat
