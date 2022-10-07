"""Init of the systems mixin module."""


from typing import List

from src.systems.mixins.base import BaseMixin
from src.systems.mixins.pseudo_labelling import PseudoLabelling

__all__: List[str] = ["PseudoLabelling"]
