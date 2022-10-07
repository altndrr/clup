"""Implementation of a DeepCluster system."""

from typing import Any, Optional, Type

import torch
import torch.nn.functional as F
from torch.nn import Module

from src.models import BaseModel
from src.systems.classification import ClassificationSystem


class DeepClusterSystem(ClassificationSystem):
    """Implementation of a DeepCluster system."""

    def __init__(
        self,
        model: Type[BaseModel],
        *args,
        criterion: Optional[Module] = None,
        lr: float = 0.001,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            model,
            *args,
            criterion=criterion,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            **kwargs
        )

        assert hasattr(model, "projector"), "model must have the projector module"
        assert hasattr(model, "prototypes"), "model must have the prototypes module"
        self.projector = getattr(model, "projector")
        self.prototypes = getattr(model, "prototypes")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        indices = None
        if isinstance(batch[1], list):
            indices, batch = batch

        inputs, labels = batch

        embeddings = self.encoder(inputs)
        z = F.normalize(self.projector(embeddings))
        p = torch.stack([p(z) for p in self.prototypes])

        predict_outputs = {
            "embeddings": embeddings,
            "labels": labels,
            "projections": z,
            "prototypes": p,
        }

        if indices is not None:
            predict_outputs["indices"] = indices

        return predict_outputs
