"""Implementations of datasets for auxiliary tasks related to generation."""

import random
from typing import Any, Tuple

import torch

from src.datasets.auxiliary.base import AuxiliaryBaseDataset
from src.datasets.utils import get_image_patch


class GridInpaintImageFolder(AuxiliaryBaseDataset):
    """
    Dataset for inpainting tasks. Erase a patch in a grid of the input and asks
    the network to inpaint the complete input.
    """

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item

        # Get the patch from the cell_id and its coordinates.
        cell_id = random.randint(0, 8)
        patch, coords = get_image_patch(sample, cell_id, return_coords=True)

        # Erase the patch.
        sample_inpaint = sample.clone()
        sample_inpaint[coords] = torch.zeros(patch.shape)

        return sample_inpaint, sample


class InpaintImageFolder(AuxiliaryBaseDataset):
    """
    Dataset for inpainting tasks. Erase some random parts of the input and
    asks the network to inpaint the complete input.
    """

    def __init__(
        self,
        root: str,
        *args,
        split: str = "train",
        **kwargs,
    ) -> None:
        self.crop_size = 30
        self.max_num_crops = 5
        super().__init__(root, *args, split=split, **kwargs)

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item

        sample_inpaint = sample.clone()
        _, width, height = sample.shape

        for _ in range(random.randint(0, self.max_num_crops)):
            point_x1 = random.randint(0, width - self.crop_size)
            point_y1 = random.randint(0, height - self.crop_size)
            point_x2 = point_x1 + self.crop_size
            point_y2 = point_y1 + self.crop_size

            coords = [..., slice(point_x1, point_x2), slice(point_y1, point_y2)]
            patch = sample_inpaint[coords]
            sample_inpaint[coords] = torch.zeros(patch.shape)

        return sample_inpaint, sample


class ReconstructImageFolder(AuxiliaryBaseDataset):
    """
    Dataset for reconstruction tasks. Provides as input and as target
    the same image and asks the network to reconstruct the input into
    the target.
    """

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item

        return sample, sample
