"""Module containing the VisDA-2017 dataset."""

import os
import shutil
import tarfile
from glob import glob

import gdown
import pandas as pd
import requests as r
from rich.progress import track

from src.datasets.base import BaseDataset


class VisDA2017(BaseDataset):
    """Implementation of the VisDA-2017 dataset."""

    available_splits = [
        "train",
        "test/trunk01",
        "test/trunk02",
        "test/trunk03",
        "test/trunk04",
        "test/trunk05",
        "test/trunk06",
        "test/trunk07",
        "test/trunk08",
        "test/trunk09",
        "test/trunk10",
        "test/trunk11",
        "test/trunk12",
        "test/trunk13",
        "test/trunk14",
        "test/trunk15",
        "test/trunk16",
        "test/trunk17",
        "test/trunk18",
        "test/trunk19",
        "test/trunk20",
        "val",
    ]

    labels = [
        "aeroplane",
        "bicycle",
        "bus",
        "car",
        "horse",
        "knife",
        "motorcycle",
        "person",
        "plant",
        "skateboard",
        "train",
        "truck",
    ]

    def __init__(
        self, *args, split: str = "train", download: bool = False, prepare: bool = False, **kwargs
    ) -> None:
        super().__init__(
            *args,
            domain="",
            split=split,
            download=download,
            prepare=prepare,
            raw_subdir_images="",
            raw_subdir_labels="",
            **kwargs
        )

    def download_data(self) -> None:
        files = {
            "train.tar": "0BwcIeDbwQ0XmdENwQ3R4TUVTMHc",
            "validation.tar": "0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA",
            "test.tar": "0BwcIeDbwQ0XmdGttZ0k2dmJYQ2c",
        }

        # Download the tar files.
        for file, web_id in files.items():
            assert self.raw_root is not None, "`raw_root` must be not None"
            output_path = os.path.join(self.raw_root, file)
            gdown.download(id=web_id, output=output_path)
            with tarfile.open(output_path, "r") as tar:
                tar.extractall(self.raw_root)

        # Download the labels of the test set.
        github_root = "https://raw.githubusercontent.com"
        entity_name = "VisionLearningGroup/taskcv-2017-public"
        file_name = "master/classification/data/image_list.txt"
        url = os.path.join(github_root, entity_name, file_name)

        assert self.raw_root is not None, "`raw_root` must be not None"
        with r.get(url, stream=True) as stream:
            with open(os.path.join(self.raw_root, "test_image_list.txt"), "wb") as f:
                for chunk in stream.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Rename the validation folder.
        os.rename(
            os.path.join(self.raw_root, "validation"),
            os.path.join(self.raw_root, "val"),
        )

    def prepare_data(self) -> None:
        assert self.raw_root is not None, "`raw_root` must be not None"
        source_dir = os.path.join(self.raw_root, self.split)

        if "test" in self.split:
            raw_test_labels = os.path.join(self.raw_root, "test_image_list.txt")
            data = pd.read_csv(raw_test_labels, sep=" ")

            for _, row in track(data.iterrows(), total=len(data)):
                orig_file, label_id = row
                label = self.labels[label_id]
                test_subsplit, filename = os.path.split(orig_file)

                # If needed, create the target folder.
                assert self.processed_root is not None, "`processed_root` must be not None"
                dir_name = os.path.join(self.processed_root, "..", test_subsplit, label)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=False)

                # Copy the image from source to target.
                source_path = os.path.join(self.raw_root, "test", orig_file)
                target_path = os.path.join(dir_name, filename)
                shutil.copy(source_path, target_path)

        else:
            images = glob(os.path.join(source_dir, "*", "*"))
            assert self.processed_root is not None, "`processed_root` must be not None"
            for _, image in track(enumerate(images), total=len(images)):
                folder, filename = os.path.split(image)
                _, class_name = os.path.split(folder)
                dir_name = os.path.join(self.processed_root, class_name)
                target_name = os.path.join(dir_name, filename)

                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=False)

                shutil.copy(image, target_name)
