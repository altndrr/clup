"""Module containing the MNIST dataset."""

import os
import shutil
from glob import glob

import numpy as np
import torchvision
from PIL import Image
from rich.progress import track
from torchvision.datasets.mnist import read_image_file, read_label_file

from src.datasets.base import BaseDataset


class MNIST(BaseDataset):
    """Implementation of the MNIST dataset."""

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
        torchvision.datasets.MNIST(self.raw_data_dir, download=True)

        # Move the downloaded files to the base of the raw root.
        assert self.raw_root is not None, "`raw_root` must be not None"
        downloaded_path = os.path.join(self.raw_root, "raw")
        downloaded_items = glob(os.path.join(downloaded_path, "*"))
        for item in downloaded_items:
            dest_name = os.path.join(self.raw_root, os.path.basename(item))
            if os.path.exists(dest_name):
                continue
            shutil.move(item, self.raw_root)

        # Remove the files not moved and the unzipped folder.
        downloaded_files = glob(os.path.join(downloaded_path, "*"))
        for file in downloaded_files:
            os.remove(file)
        os.removedirs(downloaded_path)

    def prepare_data(self) -> None:
        image_file = f"{'train' if self.split == 'train' else 't10k'}-images-idx3-ubyte"
        assert self.raw_root is not None, "`raw_root` must be not None"
        data = read_image_file(os.path.join(self.raw_root, image_file))

        label_file = f"{'train' if self.split == 'train' else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_root, label_file))

        # Create the folders for the split and the subfolders for the labels.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for label in self.labels:
            os.makedirs(os.path.join(self.processed_root, label), exist_ok=True)

        for i in track(range(len(data))):
            pixels = np.array(data[i], dtype=float)
            img = Image.fromarray(pixels).convert("RGB")
            label = str(targets[i].item())
            save_path = os.path.join(self.processed_root, label, str(i).zfill(6) + ".jpg")
            img.save(save_path)
