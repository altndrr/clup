"""Module containing the SVHN dataset."""

import os

import numpy as np
import scipy.io as sio
import torchvision
from PIL import Image
from rich.progress import track

from src.datasets.base import BaseDataset


class SVHN(BaseDataset):
    """Implementation of the SVHN dataset."""

    labels = list(map(str, range(10)))

    def __init__(
        self,
        *args,
        split: str = "train",
        download: bool = False,
        prepare: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            domain="",
            split=split,
            download=download,
            prepare=prepare,
            raw_subdir_images="",
            raw_subdir_labels="",
            **kwargs,
        )

    def download_data(self) -> None:
        assert self.raw_root is not None, "`raw_root` must be not None"
        torchvision.datasets.SVHN(self.raw_root, download=True, split="train")
        torchvision.datasets.SVHN(self.raw_root, download=True, split="test")

    def prepare_data(self) -> None:
        assert self.raw_root is not None, "`raw_root` must be not None"
        data_file = os.path.join(self.raw_root, f"{self.split}_32x32.mat")
        loaded_mat = sio.loadmat(data_file)

        data = loaded_mat["X"]
        data = np.transpose(data, (3, 0, 1, 2))
        targets = loaded_mat["y"].astype(np.uint8).squeeze()

        # Substitute the target 10 with a 0.
        np.place(targets, targets == 10, 0)

        # Create the folders for the split and the subfolders for the labels.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for label in self.labels:
            os.makedirs(os.path.join(self.processed_root, label), exist_ok=True)

        for i in track(range(len(data))):
            img = Image.fromarray(data[i], "RGB")
            label = str(targets[i])
            save_path = os.path.join(self.processed_root, label, str(i).zfill(6) + ".jpg")
            img.save(save_path)
