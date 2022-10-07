"""Implementation of many facial emotion recognition datasets."""

import os
import random
from shutil import copy
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.progress import track

from src.datasets.base import BaseDataset


class EmotionDataset(BaseDataset):
    """Abstract base class for emotion datasets."""

    labels = [
        "Surprise",
        "Fear",
        "Disgust",
        "Happiness",
        "Sadness",
        "Anger",
        "Neutral",
    ]

    def __init__(
        self,
        *args,
        split: str = "train",
        prepare: bool = False,
        raw_subdir_images: str = "",
        raw_subdir_labels: str = "",
        emotion_labels: Iterable[Union[int, str]] = None,
        **kwargs,
    ) -> None:
        self.emotion_labels = emotion_labels
        self.kwargs = kwargs

        self.device = torch.device(0)
        gpus = kwargs.get("gpus")
        if gpus:
            self.device = torch.device(gpus[0])

        super().__init__(
            *args,
            domain="",
            split=split,
            prepare=prepare,
            raw_subdir_images=raw_subdir_images,
            raw_subdir_labels=raw_subdir_labels,
            **kwargs,
        )

    def download_data(self) -> None:
        raise ValueError("emotion datasets must be downloaded manually")


class AFE(EmotionDataset):
    """Implementation of the AFE dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="src/image/",
            raw_subdir_labels="annos/TrainTest/",
            emotion_labels=[6, 2, 1, 3, 5, 0, 4],
            **kwargs,
        )

    def prepare_data(self) -> None:
        # Read the annotations file and get the split subset.
        if self.split == "train":
            annotations = os.path.join(self.raw_labels_dir, "Train.csv")
        else:
            annotations = os.path.join(self.raw_labels_dir, "Test.csv")
        data = pd.read_csv(annotations, sep=" ", header=None)
        data.columns = ["path", "left", "top", "right", "bott", "emotion"]

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        # Copy each image to its subfolder according to the expressed emotion.
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        for i, row in track(data.iterrows(), total=len(data)):
            emotion = mapping[int(row["emotion"])]

            orig_file = os.path.join(self.raw_images_dir, row["path"])
            save_path = os.path.join(self.processed_root, emotion, str(i).zfill(6) + ".jpg")
            bbox = tuple(map(int, (row["left"], row["top"], row["right"], row["bott"])))
            img = Image.open(orig_file)
            img = img.crop(bbox).resize((160, 160))
            img.save(save_path)


class ExpW(EmotionDataset):
    """Implementation of the ExpW dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="image/origin/",
            raw_subdir_labels="label/label.lst",
            emotion_labels=[5, 2, 1, 3, 4, 0, 6],
            **kwargs,
        )

    def prepare_data(self) -> None:
        # Read the annotations file.
        lines = []
        with open(self.raw_labels_dir, "r", encoding="utf-8") as file:
            lines = [line.replace("\n", "").split(" ") for line in file.readlines()]
        data = pd.DataFrame(lines)
        data.columns = [
            "file",
            "face_id",
            "top",
            "left",
            "right",
            "bott",
            "conf",
            "emo",
        ]

        # Generate the test split.
        test_subset_size = int(len(data) * 0.2)
        test_subset = random.choices(data["file"].tolist(), k=test_subset_size)

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        # Copy each image to its subfolder according to the expressed emotion.
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        for i, row in track(data.iterrows(), total=len(data)):
            # Skip the files of the other split.
            if (self.split == "train") == (row[0] in test_subset):
                continue

            emotion = mapping[int(row["emo"])]
            orig_file = os.path.join(self.raw_images_dir, row["file"])
            save_path = os.path.join(self.processed_root, emotion, str(i).zfill(6) + ".jpg")
            bbox_values = (row["left"], row["top"], row["right"], row["bott"])
            bbox = tuple(map(int, bbox_values))
            img = Image.open(orig_file).crop(bbox)
            img.save(save_path)


class FER2013(EmotionDataset):
    """Implementation of the FER2013 dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="",
            raw_subdir_labels="fer2013.csv",
            emotion_labels=[5, 2, 1, 3, 4, 0, 6],
            **kwargs,
        )

    def prepare_data(self) -> None:
        # Read the annotations file and get the split subset.
        data = pd.read_csv(self.raw_labels_dir)
        if self.split == "train":
            data = data[data["Usage"] == "Training"]
        else:
            data = data[data["Usage"] != "Training"]

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        # Recompose each image from the pixels list and save it to its subfolder
        # according to the expressed emotion.
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        for i, row in track(data.iterrows(), total=len(data)):
            emotion = mapping[int(row["emotion"])]
            pixels = row["pixels"].split(" ")
            pixels = np.array(pixels, dtype=float).reshape(48, 48)
            img = Image.fromarray(pixels).convert("RGB")

            save_path = os.path.join(self.processed_root, emotion, str(i).zfill(6) + ".jpg")
            img.save(save_path)


class RAFDB(EmotionDataset):
    """Implementation of the RAF-DB dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="basic/Image/aligned",
            raw_subdir_labels="basic/EmoLabel/list_patition_label.txt",
            emotion_labels=[1, 2, 3, 4, 5, 6, 7],
            **kwargs,
        )

    def prepare_data(self) -> None:
        # Read the annotations file.
        lines = []
        with open(self.raw_labels_dir, "r", encoding="utf-8") as file:
            lines = [line.replace("\n", "").split(" ") for line in file.readlines()]

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        # Copy each image to its subfolder according to the expressed emotion.
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        for i, (line, emotion) in enumerate(track(lines)):
            emotion = mapping[int(emotion)]

            orig_file = line.replace(".jpg", "_aligned.jpg")

            orig_file = os.path.join(self.raw_images_dir, orig_file)
            save_path = os.path.join(self.processed_root, emotion, str(i).zfill(6) + ".jpg")
            if line.startswith(self.split):
                copy(orig_file, save_path)
