"""Module containing utils functions about models."""

import os
import types
from copy import copy
from typing import Dict

import torch
import torchvision

from src.models import MODELS, ResNet18


def make_model(network, **kwargs):
    """
    Load a neural network by name or weights.

    :param network: name of the network or path to its weights.
    """
    name = None if os.path.isfile(network) else network
    weights = network if os.path.isfile(network) else None

    if name:
        model = MODELS.get(name)
        return model(weights=None, pretrained=True, **kwargs)

    for name in MODELS:
        try:
            model = MODELS.get(name)
            return model(weights=weights, **kwargs)
        except IndexError:
            pass
        except RuntimeError:
            pass

    # Try also self-supervised models.
    try:
        model = make_ssl_model(weights, **kwargs)
        return model
    except RuntimeError:
        pass

    return None


def make_ssl_model(
    weights: str,
    *args,
    num_classes: int = 7,
    features_dim=512,
    proj_hidden_dim=2048,
    proj_output_dim=128,
    **kwargs,
):
    """
    Load a DeepClusterV2 neural network from its weights.
    Currently works only for ResNet18.

    :param weights: path to the weights of the DeepClusterV2 model
    :param num_classes: number of classes
    """
    encoder = torchvision.models.resnet18()
    encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=2, bias=False)
    encoder.maxpool = torch.nn.Identity()
    encoder.fc = torch.nn.Identity()

    model = ResNet18()
    model.encoder = encoder
    model.classifier = torch.nn.Linear(512, num_classes, bias=True)
    if kwargs.get("mlp_head"):
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 128, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, num_classes, bias=True),
        )
    delattr(model.encoder, "fc")
    model.encoder.forward = types.MethodType(model.encoder_forward, model.encoder)  # type: ignore

    model.projector = torch.nn.Sequential(
        torch.nn.Linear(features_dim, proj_hidden_dim),
        torch.nn.BatchNorm1d(proj_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(proj_hidden_dim, proj_output_dim),
    )

    state_dict = torch.load(weights)["state_dict"]
    prototypes_state_dict = subset_state_dict(state_dict, "prototypes")
    if "weight_g" in prototypes_state_dict.keys():
        num_prototypes = prototypes_state_dict["weight_g"].shape[0]
        model.prototypes = torch.nn.utils.weight_norm(
            torch.nn.Linear(proj_output_dim, num_prototypes, bias=False)
        )
    else:
        num_prototypes = [p.shape[0] for p in prototypes_state_dict.values()]
        model.prototypes = torch.nn.ModuleList(
            [torch.nn.Linear(proj_output_dim, np, bias=False) for np in num_prototypes]
        )
        for proto in model.prototypes:
            for params in proto.parameters():
                params.requires_grad = False
            proto.weight.copy_(torch.nn.functional.normalize(proto.weight.data.clone(), dim=-1))

    model.encoder.load_state_dict(subset_state_dict(state_dict, "backbone"), strict=True)
    model.projector.load_state_dict(subset_state_dict(state_dict, "projector"), strict=True)
    model.prototypes.load_state_dict(prototypes_state_dict, strict=True)

    return model


def subset_state_dict(state_dict: Dict, key: str):
    """
    Get a subset of a state dictionary.

    :param state_dict: collection of weights of a neural network
    :param key: name of the subset to extract
    """
    state = copy(state_dict)
    for k in list(state.keys()):
        if key in k:
            state[k.replace(f"{key}.", "")] = state[k]
        del state[k]
    return state
