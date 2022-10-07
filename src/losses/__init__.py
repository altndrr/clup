"""Init of the losses module."""

from typing import List

from src.losses.classification import CrossEntropyListLoss
from src.losses.entropy import InfoMaxLoss
from src.losses.generation import MSELoss, MSEVGGLoss, VGGLoss

__all__: List[str] = [
    "CrossEntropyListLoss",
    "InfoMaxLoss",
    "MSELoss",
    "MSEVGGLoss",
    "VGGLoss",
]
