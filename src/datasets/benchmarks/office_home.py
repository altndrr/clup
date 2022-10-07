"""Module containing the Office-Home dataset."""

import os
import random
import shutil
import zipfile
from glob import glob
from typing import List

import gdown
from rich.progress import track

from src.datasets.base import BaseDataset


class OfficeHome(BaseDataset):
    """Implementation of the Office-Home dataset."""

    available_domains: List[str] = ["art", "clipart", "realworld", "product"]

    labels = [
        "alarm clock",
        "backpack",
        "batteries",
        "bed",
        "bike",
        "bottle",
        "bucket",
        "calculator",
        "calendar",
        "candles",
        "chair",
        "clipboards",
        "computer",
        "couch",
        "curtains",
        "desk lamp",
        "drill",
        "eraser",
        "exit Sign",
        "fan",
        "file Cabinet",
        "flipflops",
        "flowers",
        "folder",
        "fork",
        "glasses",
        "hammer",
        "helmet",
        "kettle",
        "keyboard",
        "knives",
        "lamp Shade",
        "laptop",
        "marker",
        "monitor",
        "mop",
        "mouse",
        "mug",
        "notebook",
        "oven",
        "pan",
        "paper clip",
        "pen",
        "pencil",
        "postit notes",
        "printer",
        "push pin",
        "radio",
        "refrigerator",
        "ruler",
        "scissors",
        "screwdriver",
        "shelf",
        "sink",
        "sneakers",
        "soda",
        "speaker",
        "spoon",
        "table",
        "telephone",
        "toothbrush",
        "toys",
        "trash can",
        "tv",
        "webcam",
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
            raw_subdir_images=domain,
            raw_subdir_labels="",
            **kwargs
        )

    def download_data(self) -> None:
        assert self.raw_root is not None, "`raw_root` must be not None"
        output_path = os.path.join(self.raw_root, "OfficeHome.zip")
        gdown.download(id="0B81rNlvomiwed0V1YUxQdC1uOTg", output=output_path)
        with zipfile.ZipFile(output_path) as file:
            file.extractall(self.raw_root)

        extracted_path = os.path.join(self.raw_root, "OfficeHomeDataset_10072016")
        extracted_items = glob(os.path.join(extracted_path, "*"))
        for item in extracted_items:
            item_name = os.path.basename(item)
            if os.path.exists(os.path.join(self.raw_root, item_name)):
                continue

            formatted_name = item_name.lower().replace(" ", "")
            if os.path.exists(os.path.join(self.raw_root, formatted_name)):
                continue

            shutil.move(item, self.raw_root)

        # Rename folders to lower-case.
        raw_items = glob(os.path.join(self.raw_root, "*"))
        for item in raw_items:
            if not os.path.isdir(item):
                continue

            dirname, filename = os.path.split(item)
            new_filename = filename.lower().replace(" ", "")

            if not new_filename in self.available_domains:
                continue

            item_name = os.path.basename(item)
            if not os.path.exists(os.path.join(dirname, new_filename, item_name)):
                shutil.move(item, os.path.join(dirname, new_filename))

        # Remove recursively what was not moved from the extracted directory.
        shutil.rmtree(extracted_path, ignore_errors=True)

    def prepare_data(self) -> None:
        images = glob(os.path.join(self.raw_images_dir, "*", "*"))

        # Generate the test split.
        test_subset_size = int(len(images) * 0.2)
        test_subset = random.choices(images, k=test_subset_size)

        assert self.processed_root is not None, "`processed_root` must be not None"
        for _, image in track(enumerate(images), total=len(images)):
            # Skip the files of the other split.
            if (self.split == "train") == (image in test_subset):
                continue

            folder, filename = os.path.split(image)
            _, class_name = os.path.split(folder)
            dir_name = os.path.join(self.processed_root, class_name)
            target_name = os.path.join(dir_name, filename)

            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=False)

            shutil.copy(image, target_name)
