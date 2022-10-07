"""Init for the auxiliary module."""

from typing import Dict, List, Type

from src.models.auxiliary.base import AuxiliaryBaseModel
from src.models.auxiliary.puzzle import GridPuzzle, RelativeGridPuzzle
from src.models.auxiliary.reconstruct import GridInpaint, Inpaint, Reconstruct
from src.models.auxiliary.rotate import GridRotate, RelativeRotate, Rotate

__all__: List[str] = [
    "GridPuzzle",
    "RelativeGridPuzzle",
    "GridInpaint",
    "Inpaint",
    "Reconstruct",
    "GridRotate",
    "RelativeRotate",
    "Rotate",
]

AUXILIARY_MODELS: Dict[str, Type[AuxiliaryBaseModel]] = {
    "grid-puzzle": GridPuzzle,
    "relative-grid-puzzle": RelativeGridPuzzle,
    "grid-inpaint": GridInpaint,
    "inpaint": Inpaint,
    "reconstruct": Reconstruct,
    "grid-rotate": GridRotate,
    "relative-rotate": RelativeRotate,
    "rotate": Rotate,
}
