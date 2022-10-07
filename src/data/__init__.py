"""Init of the data module."""

from typing import List

from src.data.collate import CollateDataModule
from src.data.image import AuxiliaryDataModule, ImageDataModule

__all__: List[str] = ["AuxiliaryDataModule", "CollateDataModule", "ImageDataModule"]
