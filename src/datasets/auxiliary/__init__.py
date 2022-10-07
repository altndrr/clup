"""Init of the auxiliary dataset module."""

from typing import Dict, List, Type

from src.datasets.auxiliary.base import AuxiliaryBaseDataset
from src.datasets.auxiliary.puzzle import GridPuzzleImageFolder, RelativeGridPuzzleImageFolder
from src.datasets.auxiliary.reconstruct import (
    GridInpaintImageFolder,
    InpaintImageFolder,
    ReconstructImageFolder,
)
from src.datasets.auxiliary.rotate import (
    GridRotateImageFolder,
    RelativeRotateImageFolder,
    RotateImageFolder,
)

__all__: List[str] = [
    "GridPuzzleImageFolder",
    "RelativeGridPuzzleImageFolder",
    "GridInpaintImageFolder",
    "InpaintImageFolder",
    "ReconstructImageFolder",
    "GridRotateImageFolder",
    "RelativeRotateImageFolder",
    "RotateImageFolder",
]

AUXILIARY_DATASETS: Dict[str, Type[AuxiliaryBaseDataset]] = {
    "grid-puzzle": GridPuzzleImageFolder,
    "relative-grid-puzzle": RelativeGridPuzzleImageFolder,
    "grid-inpaint": GridInpaintImageFolder,
    "inpaint": InpaintImageFolder,
    "reconstruct": ReconstructImageFolder,
    "grid-rotate": GridRotateImageFolder,
    "relative-rotate": RelativeRotateImageFolder,
    "rotate": RotateImageFolder,
}
