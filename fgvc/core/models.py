import io
import warnings
from collections import OrderedDict
from typing import Union

import timm
import torch
import torch.nn as nn


def get_model(
    architecture_name: str,
    target_size: int = None,
    pretrained: bool = False,
    *,
    checkpoint_path: Union[str, io.BytesIO] = None,
    strict: bool = True,
) -> nn.Module:
    """Get a `timm` model.

    Parameters
    ----------
    architecture_name
        Name of the network architecture from `timm` library.
    target_size
        Output feature size of the new prediction head.
    pretrained
        If true load pretrained weights from `timm` library.
    checkpoint_path
        Path (or IO Buffer) with checkpoint weights to load after the model is initialized.
    strict
        Whether to strictly enforce the keys in state_dict to match
        between the model and checkpoint weights from file.
        Used when argument checkpoint_path is specified.

    Returns
    -------
    model
        PyTorch model from `timm` library.
    """
    model = timm.create_model(architecture_name, pretrained=pretrained and checkpoint_path is None)

    # load custom weights
    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location="cpu")

        # remove prefix "module." created by nn.DataParallel wrapper
        if all([k.startswith("module.") for k in weights.keys()]):
            weights = OrderedDict({k[7:]: v for k, v in weights.items()})

        # identify target size in the weights
        cls_name = model.default_cfg["classifier"]
        weights_target_size = len(weights[f"{cls_name}.bias"])
        model_target_size = model.default_cfg["num_classes"]
        if weights_target_size != model_target_size:
            # set different target size based on the checkpoint weights
            model = set_prediction_head(model, weights_target_size)

        # load checkpoint weights
        model.load_state_dict(weights, strict=strict)

    # set classification head
    if target_size is not None:
        model = set_prediction_head(model, target_size)

    return model


def set_prediction_head(model: nn.Module, target_size: int):
    """Replace prediction head of a `timm` model.

    Parameters
    ----------
    model
        PyTorch model from `timm` library.
    target_size
        Output feature size of the new prediction head.

    Returns
    -------
    model
        The input `timm` model with new prediction head.
    """
    assert hasattr(model, "default_cfg")
    cls_name = model.default_cfg["classifier"]
    # iterate through nested modules
    parts = cls_name.split(".")
    module = model
    for i, part_name in enumerate(parts):
        if i == len(parts) - 1:
            last_layer = getattr(module, part_name)
            setattr(module, part_name, nn.Linear(last_layer.in_features, target_size))
        else:
            module = getattr(module, part_name)
    return model


def get_model_target_size(model: nn.Module) -> int:
    """Get target size (number of output classes) of a `timm` model.

    Parameters
    ----------
    model
        PyTorch model from `timm` library.

    Returns
    -------
    target_size
        Output feature size of a prediction head.
    """
    target_size = None
    if isinstance(model, nn.DataParallel):
        model = model.module  # unwrap model from data parallel wrapper for multi-gpu training
    elif hasattr(model, "model"):
        model = model.model
    cls_name = model.default_cfg["classifier"]

    # iterate through nested modules
    parts = cls_name.split(".")
    module = model
    for i, part_name in enumerate(parts):
        if i == len(parts) - 1:
            target_size = getattr(module, part_name).out_features
        else:
            module = getattr(module, part_name)

    if target_size is None:
        warnings.warn(
            "Could not find target size (number of classes) "
            f"of the model {model.__class__.__name__}."
        )

    return target_size
