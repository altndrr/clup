"""Module containing a mixin implementation for pseudo labelling."""

from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Metric, MetricCollection, Precision, Recall
from torchmetrics.functional import accuracy

from src.systems.mixins.base import BaseMixin
from src.systems.utils import predict_classes


class PseudoLabelling(BaseMixin):
    """Implementation of a mixin for pseudo labelling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pseudo_labels: Tensor
        self.pseudo_labels_confidence: Tensor
        self.pseudo_every = kwargs.get("pseudo_every", 10)
        self.pseudo_labeling_mode = kwargs.get("pseudo_labeling_mode", "centroid")
        self.pseudo_labeling_mode_kw = kwargs.get("pseudo_labeling_mode_kw", {"ssl_cycles": 0})

        metrics: Dict[str, Metric] = {
            "accuracy": Accuracy(),
            "precision": Precision(),
            "recall": Recall(),
        }
        self.metrics["pl/classifier"] = MetricCollection(
            metrics,
            prefix="pseudo_labels/classifier/",
        )
        self.metrics["pl/clustering"] = MetricCollection(
            metrics,
            prefix="pseudo_labels/clustering/",
        )

    def pseudo_labelling(self, dataloader: DataLoader):
        """
        Perform pseudo labelling on the outputs of a forward step.

        :param dataloader: dataloader to use for labeling
        """
        if self.current_epoch % self.pseudo_every != 0:
            return

        results = self.forward_all(dataloader)

        # Sort results by indices.
        assert isinstance(results, Dict)
        sorted_idx = results["indices"].argsort()
        results = {key: results[key][sorted_idx] for key in results}

        results = predict_classes(
            results,
            mode=self.pseudo_labeling_mode,
            **self.pseudo_labeling_mode_kw,
        )
        self.pseudo_labels = results["pseudo_labels"]
        self.pseudo_labels_confidence = results["pseudo_labels_confidences"]
        self.pseudo_labelling_end(results)

    def pseudo_labelling_end(self, results: STEP_OUTPUT) -> None:
        """
        This method performs additional operations not directly related with the
        evaluation of the pseudo labels.

        :param results: outputs of a forward epoch (embedding, outputs, labels)
        """
        assert isinstance(results, dict)

        outputs = results["outputs"]
        labels = results["labels"]

        # Evaluate some information for logging.
        k = outputs.shape[1]
        assert self.trainer is not None
        dataset_labels = self.trainer.datamodule.dataset.labels  # type: ignore
        labels = labels.to(int)
        _, predictions = torch.max(outputs, 1)
        subset_acc_kwargs: Dict[str, Any] = {
            "average": None,
            "num_classes": k,
            "subset_accuracy": True,
        }

        # Evaluate statistics for the classifier and the clusters.
        assert self.pseudo_labels is not None
        loop: List[Tuple[MetricCollection, torch.Tensor]] = [
            (self.metrics["pl/classifier"], predictions),
            (self.metrics["pl/clustering"], self.pseudo_labels),
        ]
        for metrics, values in loop:
            assert isinstance(metrics.prefix, str)
            subset_acc_name = metrics.prefix + "accuracy/"

            # ! We evaluate these metrics on the cpu to avoid a runtime error.
            # ! "RuntimeError: t == DeviceType::CUDAINTERNAL ASSERT FAILED"
            metrics = metrics.to("cpu")
            values = values.to("cpu")
            metrics = metrics(values, labels)

            # Evaluate and store per-class accuracy.
            subset_accuracy = accuracy(values, labels, **subset_acc_kwargs)
            for key, value in zip(dataset_labels, subset_accuracy):
                key = subset_acc_name + str(key).lower()
                metrics[key] = value
            self.log_dict(metrics, on_epoch=True)
