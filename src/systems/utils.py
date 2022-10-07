"""Module containing utils functions about systems."""

from typing import Any, Dict, List
from warnings import warn

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def collate_batches(batches: List[Any]) -> Dict[str, torch.Tensor]:
    """
    Collate the outputs of a list of batches into a single dictionary.
    Useful to aggregate the outputs of a trainer.predict cycle.

    :param batches: list of batch outputs
    """
    keys = list(batches[0].keys())
    outputs = {key: torch.Tensor() for key in keys}

    for batch in batches:
        for key in keys:
            dim = len(batch[key].shape) - 2
            outputs[key] = torch.cat((outputs[key], batch[key]), dim=dim)

    return outputs


def predict_classes(
    out: Dict[str, torch.Tensor], mode="normal", **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Evaluate the labels assigned from a system to a batch of samples.
    The batch of samples is the output of a trainer.predict step or a
    system.forward.

    The mode defines how labels are assigned to samples.
     - normal: assing classes according to confidence
     - centroid: assign classes according to centroids. Similar to KMeans,
       but follows the implementation of SHOT. Supports also self-supervised
       cycles to refine predictions with argument `ssl_cycle`
     - purity: cluster embeddings and assign them to classes. Cluster purity
       defines the confidence in the prediction. Use `num_classes` and
       `num_prototypes` to define the cluster parameters.

    :param out: dictionary of prediction outputs
    :param mode: modality to use for pseudo labeling
    """

    assert mode in ["normal", "centroid", "purity", "soft-purity"]

    if mode == "normal":
        softmax_out = torch.nn.functional.softmax(out["outputs"], dim=1)
        confidences, predictions = torch.max(softmax_out, 1)
        out["pseudo_labels"] = predictions
        out["pseudo_labels_confidences"] = confidences
    elif mode == "centroid":
        out = _predict_classes_centroids(out, **kwargs)
    elif mode in ["purity", "soft-purity"]:
        out = _predict_classes_purity(out, **kwargs)
    return out


def _predict_classes_centroids(out: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    ssl_cycles = kwargs.get("ssl_cycles", 0)

    embeddings = out["embeddings"]
    embeddings = torch.cat((embeddings, torch.ones(len(embeddings), 1)), 1)
    embeddings = (embeddings.t() / embeddings.norm(p=2, dim=1)).t()

    eps = 1e-8

    # Attain the class centroids.
    softmax_out = torch.nn.functional.softmax(out["outputs"], dim=1)
    centroids = torch.matmul(softmax_out.t(), embeddings)
    centroids /= eps + softmax_out.sum(dim=0)[:, None]

    # Get pseudo labels via nearest centroid classifier.
    distances = cdist(embeddings, centroids, "cosine")
    pseudo_labels = distances.argmin(axis=1)
    confidences = torch.nn.functional.softmax(torch.from_numpy(-distances), dim=1).max(dim=1)[0]

    # If needed, refine pseudo labels.
    for _ in range(ssl_cycles):
        # Compute the class centroids from the pseudo labels.
        aff = np.eye(out["outputs"].shape[1])[pseudo_labels]
        centroids = aff.transpose().dot(embeddings) / (eps + aff.sum(axis=0)[:, None])

        # Get pseudo labels via nearest centroid classifier.
        distances = cdist(embeddings, centroids, "cosine")
        pseudo_labels = distances.argmin(axis=1)

    out["pseudo_labels"] = torch.from_numpy(pseudo_labels)
    out["pseudo_labels_confidences"] = confidences

    return out


def _predict_classes_purity(out: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    num_classes = kwargs.get("num_classes", 7)
    num_prototypes = kwargs.get("num_prototypes", 1000)

    prototypes = out.get("prototypes")
    if prototypes is not None:
        prototype_labels = prototypes[0].argmax(dim=1).to(torch.int32)
    else:
        embeddings = out.get("projections")
        if embeddings is None:
            embeddings = out["embeddings"]
            embeddings = torch.cat((embeddings, torch.ones(len(embeddings), 1)), 1)
            embeddings = (embeddings.t() / embeddings.norm(p=2, dim=1)).t()
        kmeans = KMeans(n_clusters=num_prototypes).fit(embeddings)
        prototype_labels = torch.from_numpy(kmeans.labels_)

    softmax_out = torch.nn.functional.softmax(out["outputs"], dim=1)
    _, predictions = torch.max(softmax_out, dim=1)
    matches, scores, soft_scores = _match_predictions(
        prototype_labels, predictions, num_prototypes=num_prototypes, num_classes=num_classes
    )

    assignment_labels: torch.Tensor = prototype_labels + 10000
    assignment_soft_labels = torch.clone(softmax_out)
    sample_scores = torch.zeros_like(prototype_labels, dtype=torch.float)
    for prototype_idx, class_idx in matches:
        is_match = assignment_labels == (prototype_idx + 10000)
        assignment_labels[is_match] = class_idx
        sample_scores[is_match] = scores[prototype_idx]
        assignment_soft_labels[is_match] = soft_scores[prototype_idx]

    out["pseudo_labels"] = assignment_labels.to(torch.long)
    out["soft_pseudo_labels"] = assignment_soft_labels
    out["pseudo_labels_confidences"] = sample_scores

    # Set the purity to zero for outliers.
    drop_outliers = kwargs.get("drop_outliers", False)
    if drop_outliers:
        is_outlier = assignment_labels != predictions
        warn(f"zeroing purity of {is_outlier.sum().item()} outliers")
        sample_scores[is_outlier] = 0.0

    return out


def _match_predictions(predictions, labels, num_classes=7, num_prototypes=1000):
    """
    Associate predictions and labels. The method considers `n` predicted
    classes and `k` ground truth classes. It solves the matching in a `n to k`
    fashion.

    :param predictions: output predictions of the model
    :param labels: ground truth labels (or pseudo labels)
    """
    assert num_prototypes > num_classes

    match = {p: -1 for p in range(num_prototypes)}
    match_scores = {p: 0.0 for p in range(num_prototypes)}
    match_soft_scores = {p: ([0.0] * num_classes) for p in range(num_prototypes)}
    for prototype_idx in range(num_prototypes):
        for class_idx in range(num_classes):
            prototype_samples = predictions == prototype_idx
            class_samples = labels == class_idx

            score = ((prototype_samples * class_samples).sum()) / prototype_samples.sum()
            score = score.item()

            match_soft_scores[prototype_idx][class_idx] = score
            if score > match_scores.get(prototype_idx, -1):
                match[prototype_idx] = class_idx
                match_scores[prototype_idx] = score

    matches = list(match.items())
    scores = torch.Tensor(list(match_scores.values()))
    soft_scores = torch.Tensor(list(match_soft_scores.values()))

    return matches, scores, soft_scores
