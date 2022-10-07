"""Module containing classification losses."""

import torch
from torch import Tensor


class CrossEntropyListLoss(torch.nn.CrossEntropyLoss):
    """Apply the CrossEntropyLoss on a list of Tensors."""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward implementation for the loss."""
        # pylint: disable=arguments-renamed
        loss = torch.Tensor([0.0])
        num_heads = int(x.shape[1])
        for i in range(num_heads):
            new_l = super().forward(x[:, :, i], y[:, i])
            loss += torch.div(new_l, num_heads)

        return loss
