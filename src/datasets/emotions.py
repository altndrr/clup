"""Implementation of many facial emotion recognition datasets."""

import os
import random
from glob import glob
from shutil import copy
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from rich.progress import track

from src.datasets.base import BaseDataset
from src.decorators import multiprocess


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


class AffectNet(EmotionDataset):
    """Implementation of the AffectNet dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="Manually_Annotated_Images/",
            raw_subdir_labels="Manually_Annotated_file_lists/",
            emotion_labels=[3, 4, 5, 1, 2, 6, 0],
            **kwargs,
        )

    def prepare_data(self) -> None:
        # Read the annotations file and get the split subset.
        if self.split == "train":
            annotations = os.path.join(self.raw_labels_dir, "training.csv")
        else:
            annotations = os.path.join(self.raw_labels_dir, "validation.csv")
        data = pd.read_csv(annotations)

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        self.__process_images(data=data)

    @multiprocess(workers=4)
    def __process_images(self, data: pd.DataFrame) -> None:
        """
        Process all the images in the dataset.

        :param data: dataframe containing the data to process
        """
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        for i, row in track(data.iterrows(), total=len(data)):
            emotion = int(row["expression"])

            # Skip the emotion contempt and other uncertain states.
            if emotion >= 7:
                continue

            emotion_map = mapping[emotion]

            orig_file = os.path.join(self.raw_images_dir, row["subDirectory_filePath"])
            _, ext = os.path.splitext(row["subDirectory_filePath"])

            # Skip images that don't exist.
            if not os.path.exists(orig_file):
                continue

            ext = ".jpg"
            assert self.processed_root is not None, "`processed_root` must be not None"
            save_path = os.path.join(self.processed_root, emotion_map, str(i).zfill(6) + ext)
            bbox_values = (
                row["face_y"],
                row["face_x"],
                row["face_width"],
                row["face_height"],
            )
            bbox = tuple(map(int, bbox_values))
            img = Image.open(orig_file)
            img = img.crop(bbox).resize((160, 160))
            img.save(save_path)


class CKPlus(EmotionDataset):
    """Implementation of the CK+ dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="cohn-kanade-images/",
            raw_subdir_labels="Emotion/",
            emotion_labels=[7, 4, 3, 5, 6, 1, 0],
            **kwargs,
        )

    def prepare_data(self) -> None:
        # Filter and match annotations and images.
        files, images = self._match_images_labels()

        # Get random neutral faces to add the contribution of the class to the dataset.
        num_neutral_faces = 36
        sequences = glob(os.path.join(self.raw_images_dir, "*", "*"))
        neutrals = [sorted(glob(os.path.join(seq, "*.png")))[0] for seq in sequences]
        neutrals = random.choices(neutrals, k=num_neutral_faces)

        # Extend files and images with the neutral faces.
        files.extend(["" for _ in range(len(neutrals))])
        images.extend(neutrals)

        # Generate the test split.
        test_subset_size = int(len(images) * 0.2)
        test_subset = random.choices(images, k=test_subset_size)

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        # Copy each image to its subfolder according to the expressed emotion.
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        mtcnn = MTCNN(device=self.device)
        for i, data in enumerate(track(zip(files, images), total=len(files))):
            label_file, image = data

            # Skip the files of the other split.
            if (self.split == "train") == (image in test_subset):
                continue

            # Default emotion label to neutral.
            emotion_idx = 0

            # If the label_file is a file, then overwrite the emotion idx.
            if os.path.isfile(label_file):
                with open(label_file, "r", encoding="utf-8") as file:
                    emotion_idx = int(file.readline().strip()[0])

                    # Skip the emotion contempt.
                    if emotion_idx == 2:
                        continue

            emotion = mapping[emotion_idx]

            save_path = os.path.join(self.processed_root, emotion, str(i).zfill(6) + ".png")
            img = Image.open(image).convert("RGB")
            mtcnn(img, save_path=save_path)

    def _match_images_labels(self) -> Tuple[List[str], List[str]]:
        """
        Get the lists of annotations and images. Since annotations refers to the
        last image in a sequence, drop all images except the last one.

        :param annotations: path to the annotation folder
        :param image_dir: path to the image folder
        """

        # Get the annotations files.
        files = glob(os.path.join(self.raw_labels_dir, "*", "*", "*.txt"))

        # Get the path to every sequence and select the last image for each.
        sequences = glob(os.path.join(self.raw_images_dir, "*", "*"))
        images = [sorted(glob(os.path.join(seq, "*.png")))[-1] for seq in sequences]

        # Remove the images without an associated annotation file.
        basenames = [os.path.basename(file)[:17] for file in files]
        images = [image for image in images if os.path.basename(image)[:17] in basenames]

        # Sort files and images to align them.
        files = sorted(files)
        images = sorted(images)

        return files, images


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


class JAFFE(EmotionDataset):
    """Implementation of the JAFFE dataset."""

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images="images/",
            raw_subdir_labels="",
            emotion_labels=["SU", "FE", "DI", "HA", "SA", "AN", "NE"],
            **kwargs,
        )

    def prepare_data(self) -> None:
        images = glob(os.path.join(self.raw_images_dir, "*.tiff"))

        # Generate the test split.
        test_subset_size = int(len(images) * 0.2)
        test_subset = random.choices(images, k=test_subset_size)

        # Create the folders for the split and the subfolders for the emotions.
        assert self.processed_root is not None, "`processed_root` must be not None"
        os.makedirs(self.processed_root, exist_ok=True)
        for emotion in self.labels:
            os.makedirs(os.path.join(self.processed_root, emotion), exist_ok=True)

        # Copy each image to its subfolder according to the expressed emotion.
        mtcnn = MTCNN(device=self.device)
        assert self.emotion_labels is not None, "`emotion_labels` must be not None"
        mapping = dict(zip(self.emotion_labels, self.labels))
        for i, image in enumerate(track(images)):
            # Skip the files of the other split.
            if (self.split == "train") == (image in test_subset):
                continue

            emotion = mapping[os.path.basename(image).split(".")[1][:2]]
            save_path = os.path.join(self.processed_root, emotion, str(i).zfill(6) + ".tiff")
            img = Image.open(image).convert("RGB")
            mtcnn(img, save_path=save_path)


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


class SFEW(EmotionDataset):
    """Implementation of the SFEW dataset."""

    available_splits = ["train", "test"]

    def __init__(self, split: str = "train", prepare: bool = False, **kwargs) -> None:
        raw_subdir_images = "Train/Train_Aligned_Faces"
        if split == "test":
            raw_subdir_images = "Val/Val_Aligned_Faces_new/Val_Aligned_Faces"

        super().__init__(
            split=split,
            prepare=prepare,
            raw_subdir_images=raw_subdir_images,
            raw_subdir_labels="",
            emotion_labels=[1, 2, 3, 4, 5, 6, 7],
            **kwargs,
        )

    def prepare_data(self) -> None:
        assert self.processed_root is not None, "`processed_root` must be not None"

        images = glob(os.path.join(self.raw_images_dir, "*", "*"))

        for i, image in track(enumerate(images), total=len(images)):
            dir_name, filename = os.path.split(image)
            _, ext = os.path.splitext(filename)
            class_name = os.path.basename(dir_name)

            target_dir = os.path.join(self.processed_root, class_name)
            target_name = os.path.join(target_dir, str(i).zfill(6) + f".{ext}")

            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=False)

            copy(image, target_name)
