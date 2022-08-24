import timm
import torch
import torch.nn as nn


def get_model(
    architecture_name: str,
    target_size: int,
    pretrained: bool = False,
    checkpoint_path: str = None,
) -> nn.Module:
    model = timm.create_model(
        architecture_name, pretrained=pretrained and checkpoint_path is None
    )

    # set classification head
    model_cfg = model.default_cfg
    cls_name = model_cfg["classifier"]
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
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    return model
