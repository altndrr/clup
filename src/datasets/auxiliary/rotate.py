"""Implementations of datasets for auxiliary tasks related to rotation."""

import random
from typing import Any, Tuple

import torch
from torchvision import transforms as T

from src.datasets.auxiliary.base import AuxiliaryBaseDataset
from src.datasets.utils import get_image_patch


class GridRotateImageFolder(AuxiliaryBaseDataset):
    """
    Dataset for rotation tasks. Rotate a patch in a grid of the input and asks
    the network understand the id of the rotated cell. The specific cell has
    a random rotation in [90, 180, 270].
    """

    def __init__(self, root: str, *args, split: str = "train", **kwargs) -> None:
        self.rotations = None
        self.cell_ids = None

        # Generate static rotations and cell ids for the validation set.
        if self.split != "train":
            self.cell_ids = torch.randint(0, 9, (len(self.samples),)).tolist()
            self.rotations = torch.randint(1, 4, (len(self.samples),)).tolist()

        super().__init__(root, *args, split=split, **kwargs)

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item

        # Generate values for the train set, or retrieve them for the validation set.
        if self.split == "train":
            cell_id = random.randint(0, 8)
            rotation = random.randint(1, 3)
        else:
            assert self.cell_ids is not None, "`cell_ids` must be not None"
            assert self.rotations is not None, "`rotations` must be not None"
            cell_id = self.cell_ids[index]
            rotation = self.rotations[index]

        # Get the patch from the cell_id with its coordinates.
        patch, coords = get_image_patch(sample, cell_id, return_coords=True)

        # Rotate only the patch.
        sample_rot = sample.clone()
        degree = rotation * 90.0
        sample_rot[coords] = T.functional.rotate(patch, degree)

        return sample_rot, cell_id


class RotateImageFolder(AuxiliaryBaseDataset):
    """
    Dataset for rotation tasks. Rotate the complete input and asks the network
    to understand its rotation. The image can present a a random rotation in
    [0, 90, 180, 270].
    """

    def __init__(self, root: str, *args, split: str = "train", **kwargs) -> None:
        self.rotations = None

        # Generate static rotations for the validation set.
        if self.split != "train":
            self.rotations = torch.randint(0, 4, (len(self.samples),)).tolist()

        super().__init__(root, *args, split=split, **kwargs)

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item

        # Generate the target for the train set, or retrieve it for the validation set.
        if self.split == "train":
            target = random.randint(0, 3)
        else:
            assert self.rotations is not None, "`rotations` must be not None"
            target = self.rotations[index]

        # Rotate image according to label.
        sample_rot = T.functional.rotate(sample, target * 90.0)

        return sample_rot, target


class RelativeRotateImageFolder(AuxiliaryBaseDataset):
    """
    Dataset for rotation tasks. Extends the RotateImageFolder to return
    the original and the rotated samples together. It asks the network
    to understand the relative rotation between the two inputs.
    """

    def __init__(self, root: str, *args, split: str = "train", **kwargs) -> None:
        self.rotations = None

        # Generate static rotations for the validation set.
        if self.split != "train":
            self.rotations = torch.randint(0, 4, (len(self.samples),)).tolist()

        super().__init__(root, *args, split=split, **kwargs)

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item

        # Generate the target for the train set, or retrieve it for the validation set.
        if self.split == "train":
            target = random.randint(0, 3)
        else:
            assert self.rotations is not None, "`rotations` must be not None"
            target = self.rotations[index]

        # Rotate image according to label.
        sample_rot = T.functional.rotate(sample, target * 90.0)

        return (sample_rot, sample), target
