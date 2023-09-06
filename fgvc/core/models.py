import io
import logging
import warnings
from collections import OrderedDict
from typing import Optional, Union

import timm
import torch
import torch.nn as nn

logger = logging.getLogger("fgvc")


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
    pretrained = pretrained and checkpoint_path is None
    model = timm.create_model(architecture_name, pretrained=pretrained)

    # load model with classification head if missing
    # models like ViT trained with DINO do not have a classification head by default
    # classification head is missing to get `in_features` value in the method `set_prediction_head`
    if model.default_cfg["num_classes"] == 0 and target_size is not None:
        model = timm.create_model(architecture_name, pretrained=pretrained, num_classes=1000)

    # load custom weights
    if checkpoint_path is not None:
        logger.debug("Loading pre-trained checkpoint.")
        weights = torch.load(checkpoint_path, map_location="cpu")

        # remove prefix "module." created by nn.DataParallel wrapper
        if all([k.startswith("module.") for k in weights.keys()]):
            weights = OrderedDict({k[7:]: v for k, v in weights.items()})

        # identify target size in the weights
        cls_name = model.default_cfg["classifier"]
        if f"{cls_name}.bias" in weights and f"{cls_name}.weight" in weights:
            weights_target_size = weights[f"{cls_name}.bias"].shape[0]
            model_target_size = model.default_cfg["num_classes"]
            if weights_target_size != model_target_size:
                # set different target size based on the checkpoint weights
                in_features = weights[f"{cls_name}.weight"].shape[1]
                model = set_prediction_head(model, weights_target_size, in_features=in_features)

        # load checkpoint weights
        model.load_state_dict(weights, strict=strict)

    # set classification head
    if target_size is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_target_size = get_model_target_size(model)
        if target_size != model_target_size:
            logger.debug("Setting new prediction head with random initial weights.")
            model = set_prediction_head(model, target_size)

    return model


def set_prediction_head(model: nn.Module, target_size: int, *, in_features: int = None):
    """Replace prediction head of a `timm` model.

    Parameters
    ----------
    model
        PyTorch model from `timm` library.
    target_size
        Output feature size of the new prediction head.
    in_features
        Number of input features for the prediction head.
        The parameter is needed in special cases,
        e.g., when the current prediction head is `nn.Identity`.

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
            in_features = in_features or last_layer.in_features
            setattr(module, part_name, nn.Linear(in_features, target_size))
        else:
            module = getattr(module, part_name)
    return model


def get_model_target_size(model: nn.Module) -> Optional[int]:
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
        module = getattr(module, part_name)
    # set target size of the last nested module if it is a linear layer
    # other layers like nn.Identity are ignored
    if hasattr(module, "out_features"):
        target_size = module.out_features

    if target_size is None:
        warnings.warn(
            "Could not find target size (number of classes) "
            f"of the model {model.__class__.__name__}."
        )

    return target_size
