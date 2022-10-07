"""Init of the systems module."""

from typing import List

from src.systems.classification import ClassificationSystem
from src.systems.clup import CluPSystem

__all__: List[str] = [
    "ClassificationSystem",
    "CluPSystem",
]
