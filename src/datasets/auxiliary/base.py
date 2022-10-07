"""Module containing an abstract base dataset for auxiliary tasks."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from torchvision.datasets import ImageFolder


class AuxiliaryBaseDataset(ABC, ImageFolder):
    """Implementation of an abstract auxiliary dataset."""

    def __init__(self, root: str, *args, split: str = "train", **kwargs) -> None:
        self.split = split
        transform = kwargs.get("transform")
        target_transform = kwargs.get("target_transform")
        self.args = args
        self.kwargs = kwargs
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.__transformitem__(super().__getitem__(index), index)

    @abstractmethod
    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        raise NotImplementedError("method not implemented")
