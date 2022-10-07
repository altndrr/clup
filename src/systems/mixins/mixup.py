"""Module containing a mixin implementation for mixup."""

import numpy as np
import torch

from src.systems.mixins.base import BaseMixin


class Mixup(BaseMixin):
    """Implementation of a mixin for mixup."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mixup_alpha = kwargs.get("mixup_alpha", -1)
        self.mixup_last_lambda = -1

    def mixup_data(self, x, y):
        """
        Mix inputs and targets.

        :param x: input data
        :param y: target data
        """
        assert self.mixup_alpha >= 0

        lam = 0
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        batch_size = x.size()[0]

        indices = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[indices, :]
        y_a, y_b = y, y[indices]

        self.mixup_last_lambda = lam

        return mixed_x, y_a, y_b

    def mixup_criterion(self, predictions, y_a, y_b, criterion=None):
        """Evaluate the loss with the mixed targets."""
        if criterion is None:
            criterion = self.criterion
        lam = self.mixup_last_lambda
        return lam * criterion(predictions, y_a) + (1 - lam) * criterion(predictions, y_b)
