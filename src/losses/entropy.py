"""Module containing entropy-based losses."""

import torch
from torch import Tensor


class InfoMaxLoss(torch.nn.Module):
    """Implementation of Information Maximisation loss."""

    def __init__(self, gent: bool = True) -> None:
        super().__init__()
        self.gent = gent

    def forward(self, x: Tensor) -> Tensor:
        """Forward implementation for the loss."""
        softmax_out = torch.nn.Softmax(dim=1)(x)

        entropy = -softmax_out * torch.log(softmax_out + 1e-5)
        entropy = torch.sum(entropy, dim=1)

        entropy_loss = torch.mean(entropy)

        if self.gent:
            msoftmax = softmax_out.mean(dim=0)
            entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

        return entropy_loss
