"""Init of the systems module."""

from typing import List

from src.systems.classification import ClassificationSystem
from src.systems.cluster_match import ClusterMatchSystem
from src.systems.curriculum_labeling import CurriculumLabelingSystem
from src.systems.deep_cluster import DeepClusterSystem
from src.systems.source_hypothesis_transfer import SourceHypothesisTransferSystem

__all__: List[str] = [
    "ClassificationSystem",
    "ClusterMatchSystem",
    "CurriculumLabelingSystem",
    "DeepClusterSystem",
    "SourceHypothesisTransferSystem",
]
