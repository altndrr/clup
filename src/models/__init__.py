"""Init of the nn module."""

from typing import Dict, List, Type

from src.models.base import BaseModel
from src.models.mobilenet import MobileNetV2
from src.models.resnet import ResNet18, ResNet50

__all__: List[str] = ["MobileNetV2", "ResNet18", "ResNet50"]

MODELS: Dict[str, Type[BaseModel]] = {
    "mobilenetv2": MobileNetV2,
    "resnet18": ResNet18,
    "resnet50": ResNet50,
}
