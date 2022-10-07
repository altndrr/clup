"""Module containing an abstract base dataset."""

import os
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, List, Optional, Tuple

from torchvision.datasets import ImageFolder


class BaseDataset(ABC, ImageFolder):
    """Implementation of a base dataset."""

    available_domains: List[str] = []
    available_splits: List[str] = ["train", "test"]

    def __init__(
        self,
        *args,
        domain: str = "",
        split: str = "train",
        download: bool = False,
        prepare: bool = False,
        data_dir: str = "",
        raw_data_dir: str = "",
        raw_subdir_images: str = "",
        raw_subdir_labels: str = "",
        return_index: bool = False,
        **kwargs,
    ) -> None:
        assert (
            domain in self.available_domains or domain == ""
        ), f"{domain} not in {self.available_domains}"
        assert (
            split in self.available_splits or split == ""
        ), f"{split} not in {self.available_splits}"

        self.domain = domain
        self.split = split
        self.prepare = prepare
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.return_index = return_index
        transform = kwargs.get("transform")
        target_transform = kwargs.get("target_transform")
        self.args = args
        self.kwargs = kwargs

        assert self.data_dir is not None, "`data_dir` must be not None"

        if not hasattr(self, "labels"):
            raise NotImplementedError("class property 'labels' not implemented")

        if self.raw_root:
            self.raw_images_dir = os.path.join(self.raw_root, raw_subdir_images)
            self.raw_labels_dir = os.path.join(self.raw_root, raw_subdir_labels)

        if download:
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)

            if not self.raw_root:
                os.makedirs(os.path.join(self.raw_data_dir, type(self).__name__))
            self.download_data()

        if prepare:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            if not self.processed_root:
                os.makedirs(os.path.join(self.data_dir, type(self).__name__))
            self.prepare_data()

        super().__init__(self.processed_root, transform, target_transform)

    @abstractmethod
    def download_data(self) -> None:
        """Download the raw data from the web."""
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare the raw data for training."""
        raise NotImplementedError("method not implemented")

    @property
    def raw_root(self) -> Optional[str]:
        """Returns the root to the raw data."""
        if not self.raw_data_dir:
            return None

        raw_dirs = glob(os.path.join(self.raw_data_dir, "*"))

        # Search for the raw directory matching the class name.
        for directory in raw_dirs:
            if not os.path.isdir(directory):
                continue

            dir_basename = os.path.basename(directory)
            if dir_basename.lower() == type(self).__name__.lower():
                return directory

        return None

    @property
    def processed_root(self) -> Optional[str]:
        """Returns the root to the processed data."""
        if not self.data_dir:
            return None

        data_dirs = glob(os.path.join(self.data_dir, "*"))

        # Search for the data directory matching the class name.
        data_dir = None
        for directory in data_dirs:
            if not os.path.isdir(directory):
                continue

            dir_basename = os.path.basename(directory)
            if dir_basename.lower() == type(self).__name__.lower():
                data_dir = directory

        if not data_dir:
            return None

        return os.path.join(data_dir, self.domain, self.split)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.return_index:
            return index, super().__getitem__(index)

        return super().__getitem__(index)
