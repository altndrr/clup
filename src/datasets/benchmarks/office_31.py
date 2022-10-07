"""Module containing the Office-31 dataset."""

import os
import random
import shutil
import tarfile
from glob import glob
from typing import List

import gdown
from rich.progress import track

from src.datasets.base import BaseDataset


class Office31(BaseDataset):
    """Implementation of the Office-31 dataset."""

    available_domains: List[str] = ["amazon", "dslr", "webcam"]

    labels = [
        "back_pack",
        "bike",
        "bike_helmet",
        "bookcase",
        "bottle",
        "calculator",
        "desk_chair",
        "desk_lamp",
        "desktop_computer",
        "file_cabinet",
        "headphones",
        "keyboard",
        "laptop_computer",
        "letter_tray",
        "mobile_phone",
        "monitor",
        "mouse",
        "mug",
        "paper_notebook",
        "pen",
        "phone",
        "printer",
        "projector",
        "punchers",
        "ring_binder",
        "ruler",
        "scissors",
        "speaker",
        "stapler",
        "tape_dispenser",
        "trash_can",
    ]

    def __init__(
        self,
        domain: str,
        *args,
        split: str = "train",
        download: bool = False,
        prepare: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            domain=domain,
            split=split,
            download=download,
            prepare=prepare,
            raw_subdir_images=os.path.join(domain, "images"),
            raw_subdir_labels="",
            **kwargs
        )

    def download_data(self) -> None:
        assert self.raw_root is not None, "`raw_root` must be not None"
        output_path = os.path.join(self.raw_root, "office.tar.gz")
        gdown.download(id="0B4IapRTv9pJ1WGZVd1VDMmhwdlE", output=output_path)
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall(self.raw_root)

    def prepare_data(self) -> None:
        assert self.processed_root is not None, "`processed_root` must be not None"

        images = glob(os.path.join(self.raw_images_dir, "*", "*"))

        # Generate the test split.
        test_subset_size = int(len(images) * 0.2)
        test_subset = random.choices(images, k=test_subset_size)

        for _, image in track(enumerate(images), total=len(images)):
            # Skip the files of the other split.
            if (self.split == "train") == (image in test_subset):
                continue

            class_name = os.path.split(os.path.dirname(image))[1]
            basename = os.path.basename(image)
            target_name = os.path.join(self.processed_root, class_name, basename)
            dir_name = os.path.dirname(target_name)

            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=False)

            shutil.copy(image, target_name)
