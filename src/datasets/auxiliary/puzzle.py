"""Implementations of datasets for auxiliary tasks related to puzzles."""

import itertools
import random
from typing import Any, Tuple

import torch
from torch import Tensor

from src.datasets.auxiliary.base import AuxiliaryBaseDataset
from src.datasets.utils import get_image_patch


class GridPuzzleBaseDataset(AuxiliaryBaseDataset):
    """
    Implementation of an abstract class for grid puzzle datasets.
    """

    def __init__(self, root: str, *args, split: str = "train", **kwargs) -> None:
        super().__init__(root, *args, split=split, **kwargs)

        self.grid_size = 3
        self.num_cells = self.grid_size**2
        self.perm = list(itertools.permutations(range(self.num_cells)))

    def permute_sample(self, sample: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Permute the input sample by shuffling its patches in a grid.

        :param sample: item to transform.
        """
        # Get the permutation.
        perm = random.choice(self.perm)
        target = torch.Tensor(perm).to(torch.int)

        # Permute the cells in the image.
        sample_perm = sample.clone()
        for i in range(self.num_cells):
            j = perm.index(i)
            patch = get_image_patch(sample, i, grid_size=self.grid_size)
            _, new_coords = get_image_patch(sample, j, return_coords=True, grid_size=self.grid_size)
            assert isinstance(patch, Tensor), "`patch` must be of type torch.Tensor"
            sample_perm[new_coords] = patch

        return sample_perm, target


class GridPuzzleImageFolder(GridPuzzleBaseDataset):
    """
    Dataset for grid puzzle tasks. Provides as input and as target
    the same image and asks the network to reconstruct the input into
    the target.
    """

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item
        sample_perm, target = self.permute_sample(sample)

        return sample_perm, target


class RelativeGridPuzzleImageFolder(GridPuzzleBaseDataset):
    """
    Dataset for grid puzzle tasks. Extends the GridPuzzleImageFolder to return
    the original and the shuffled samples together. It asks the network
    to understand the position of the patches between the two inputs.
    """

    def __transformitem__(self, item: Tuple[Any, Any], index: int) -> Tuple[Any, Any]:
        sample, _ = item
        sample_perm, target = self.permute_sample(sample)

        return (sample_perm, sample), target
