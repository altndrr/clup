"""Module containing an abstract class for deep neural networks."""

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Module


class BaseModel(ABC, torch.nn.Module):
    """Implementation of an abstract class for neural networks."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder: Module
        self.classifier: Module

    def forward(self, x: Tensor) -> Tensor:
        """Forward step of the network."""
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    @property
    @abstractmethod
    def classifier_name(self) -> str:
        """Return the name of the last layer."""
        raise NotImplementedError("property 'classifier_name' not implemented")

    @property
    @abstractmethod
    def encoder_forward(self) -> Callable[..., Any]:
        """Returns a method that formalises the forward step applied on the encoder."""
        raise NotImplementedError("property 'encoder_forward' not implemented")

    @abstractmethod
    def load_weights(self, weights: str) -> None:
        """Load the weights on the model parts."""
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def update_classifier(self, num_classes: int, weight_norm: bool) -> None:
        """
        Update the classifier by changing the output features and apply weight normalisation.

        :param num_classes: output dimension of the classifier
        :param weight_norm: whether to apply weight norm on the last layer
        """
        raise NotImplementedError("method not implemented")
