import warnings

import timm
import torch
import torch.nn as nn


def get_model(
    architecture_name: str,
    target_size: int = None,
    pretrained: bool = False,
    *,
    checkpoint_path: str = None,
    strict: bool = True,
) -> nn.Module:
    """Get a `timm` model.

    Parameters
    ----------
    architecture_name
        Name of the network architecture from `timm` library.
    target_size
        If set the model's classification head is replaced
        with a classification head with output feature size = `target_size`.
    pretrained
        If true load pretrained ImageNet-1k weights.
    checkpoint_path
        Path of checkpoint to load after the model is initialized.
    strict
        Whether to strictly enforce the keys in state_dict to match
        between the model and checkpoint weights from file.
        Used when argument checkpoint_path is specified.

    Returns
    -------
    model
        PyTorch `nn.Module` instance.
    """
    model = timm.create_model(architecture_name, pretrained=pretrained and checkpoint_path is None)

    # set classification head
    if target_size is not None:
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

    # load custom weights
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=strict)

    return model


def get_model_target_size(model: nn.Module) -> int:
    """Get target size (number of output classes) of `timm` model.

    Parameters
    ----------
    model
        PyTorch `nn.Module` instance.

    Returns
    -------
    target_size
        Output feature size of a classification head.
    """
    target_size = None
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
