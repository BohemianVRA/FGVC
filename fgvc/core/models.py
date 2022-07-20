import timm
import torch
import torch.nn as nn


def get_model(architecture_name: str, target_size: int, pretrained: bool = False, checkpoint_path: str = None) -> nn.Module:
    net = timm.create_model(architecture_name, pretrained=pretrained and checkpoint_path is None)

    # set classification head
    net_cfg = net.default_cfg
    last_layer = net_cfg["classifier"]
    in_features = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(in_features, target_size))

    # load custom weights
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    return net
