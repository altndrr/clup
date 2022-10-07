"""Module of utility functions for datasets."""


from typing import List, Tuple, Union

import torch


def get_image_patch(
    image: torch.Tensor, cell_id: int, grid_size: int = 3, return_coords: bool = False
) -> Union[Tuple[torch.Tensor, List[object]], torch.Tensor]:
    """
    Get the `n`-th patch of an image split into `s \times s` parts.

    :param image: original image
    :param n: id of the patch to select (between 0 and `s^2`)
    :param s: size of the grid
    :param return_coords: return the points of the patch
    """
    # Get the size of a cell.
    cell_size_x = image.shape[-2] // grid_size
    cell_size_y = image.shape[-1] // grid_size

    # Evaluate the points for the patch.
    point_x1 = cell_size_x * (cell_id // grid_size)
    point_x2 = round(point_x1 + cell_size_x)
    point_x1 = round(point_x1)
    point_y1 = cell_size_y * (cell_id % grid_size)
    point_y2 = round(point_y1 + cell_size_y)
    point_y1 = round(point_y1)
    coords = [..., slice(point_x1, point_x2), slice(point_y1, point_y2)]

    # Get the patch.
    image_patch = image[coords]

    # If necessary, return the coordinates.
    if return_coords:
        return image_patch, coords

    return image_patch
