"""Module containing MobileNet architectures."""

import types
from collections import OrderedDict
from typing import Any, Callable, Optional

import torch
import torchvision

from src.models.base import BaseModel


class MobileNetV2(BaseModel):
    """Implementation of a MobileNetV2 neural module."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        weights: Optional[str] = None,
        weight_norm: bool = False,
        pretrained: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoder = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.classifier = getattr(self.encoder, self.classifier_name)
        delattr(self.encoder, self.classifier_name)

        self.encoder.forward = types.MethodType(self.encoder_forward, self.encoder)  # type: ignore

        assert isinstance(
            self.classifier, torch.nn.Sequential
        ), "`classifier` must be of type torch.nn.Sequential"
        self.classifier.in_features = self.classifier[-1].in_features
        self.classifier.out_features = self.classifier[-1].out_features

        self.load_weights(weights)
        self.update_classifier(num_classes, weight_norm)

    @property
    def classifier_name(self) -> str:
        return "classifier"

    @property
    def encoder_forward(self) -> Callable[..., Any]:
        def _encoder_forward(self, x):
            x = self.features(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

            return x

        return _encoder_forward

    def load_weights(self, weights: Optional[str]) -> None:
        if not weights:
            return

        weights_dict = torch.load(weights, map_location="cpu")

        if "state_dict" in weights_dict:
            weights_dict = weights_dict["state_dict"]

        # If present, remove the logits layer and overwrite output size.
        if "logits.weight" in weights_dict:
            del weights_dict["logits.weight"]
            del weights_dict["logits.bias"]

        from src.models.utils import subset_state_dict  # pylint: disable=import-outside-toplevel

        encoder_weights = subset_state_dict(weights_dict, "encoder")
        classifier_weights = subset_state_dict(weights_dict, "classifier")

        # Load legacy models.
        if len(encoder_weights) == 0 and len(classifier_weights) == 0:
            encoder_weights = OrderedDict()
            classifier_weights = OrderedDict()
            for key, value in weights_dict.items():
                if self.classifier_name in key:
                    key = key.split(".")[-1]
                    classifier_weights[key] = value
                else:
                    encoder_weights[key] = value

        num_classes = list(classifier_weights.values())[-1].shape[0]

        weight_norm = f"weight_g" in classifier_weights
        self.update_classifier(num_classes, weight_norm)

        self.encoder.load_state_dict(encoder_weights, strict=True)
        self.classifier.load_state_dict(classifier_weights, strict=True)

    def update_classifier(self, num_classes: Optional[int], weight_norm: bool) -> None:
        if not self.classifier:
            return

        assert isinstance(
            self.classifier, torch.nn.Sequential
        ), "`classifier` must be of type torch.nn.Sequential"

        if num_classes and num_classes != self.classifier.out_features:
            in_features = self.classifier.in_features
            bias = hasattr(self.classifier[1], "bias")
            assert isinstance(in_features, int), "`in_features` must be of type int"
            self.classifier[1] = torch.nn.Linear(in_features, num_classes, bias)

        if weight_norm and not hasattr(self.classifier[1], "weight_g"):
            self.classifier[1] = torch.nn.utils.weight_norm(self.classifier[1])
