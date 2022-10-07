"""Init of the dataset module."""

from typing import Dict, List, Type

from src.datasets.base import BaseDataset
from src.datasets.emotions import AFE, FER2013, RAFDB, ExpW

__all__: List[str] = [
    "AFE",
    "FER2013",
    "RAFDB",
    "ExpW",
]

DATASETS: Dict[str, Type[BaseDataset]] = {
    "afe": AFE,
    "fer2013": FER2013,
    "rafdb": RAFDB,
    "expw": ExpW,
}
