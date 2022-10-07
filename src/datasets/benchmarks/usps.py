"""Module containing the USPS dataset."""

import bz2
import os

import numpy as np
import torchvision
from PIL import Image
from rich.progress import track

from src.datasets.base import BaseDataset


class USPS(BaseDataset):
    """Implementation of the USPS dataset."""

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
        torchvision.datasets.USPS(self.raw_root, download=True, train=True)
        torchvision.datasets.USPS(self.raw_root, download=True, train=False)

    def prepare_data(self) -> None:
        assert self.raw_root is not None, "`raw_root` must be not None"

        data_file: str
        if self.split == "train":
            data_file = os.path.join(self.raw_root, "usps.bz2")
        else:
            data_file = os.path.join(self.raw_root, "usps.t.bz2")

        with bz2.open(data_file) as file:
            raw_data = [line.decode().split() for line in file.readlines()]
            tmp_list = [[x.split(":")[-1] for x in raw[1:]] for raw in raw_data]
            data = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
            data = ((np.array(data) + 1) / 2 * 255).astype(dtype=np.uint8)
            targets = [int(d[0]) - 1 for d in raw_data]

        # Create the folders for the split and the subfolders for the labels.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for label in self.labels:
            os.makedirs(os.path.join(self.processed_root, label), exist_ok=True)

        for i in track(range(len(data))):
            pixels = np.array(data[i], dtype=float)
            img = Image.fromarray(pixels).convert("RGB")
            label = str(targets[i])
            save_path = os.path.join(self.processed_root, label, str(i).zfill(6) + ".jpg")
            img.save(save_path)
