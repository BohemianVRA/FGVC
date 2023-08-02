import io

import pytest
import timm
import torch
import torch.nn as nn

from fgvc.core.models import get_model, get_model_target_size


@pytest.mark.parametrize(
    argnames=["architecture_name"],
    argvalues=[
        ("vit_small_patch16_224",),
        ("swin_tiny_patch4_window7_224",),
        ("resnet18",),
        ("convnext_tiny",),
    ],
)
def test_get_model_1(architecture_name: str):
    """Test getting `timm` model and setting custom prediction head.

    The test case tests method `get_model` as well as `set_prediction_head`.
    """
    model = get_model(
        architecture_name=architecture_name,
        pretrained=False,
        target_size=600,
    )
    assert hasattr(model, "default_cfg")
    if architecture_name == "vit_small_patch16_224":
        assert model.head.out_features == 600
    elif architecture_name == "swin_tiny_patch4_window7_224":
        assert model.head.fc.out_features == 600
    elif architecture_name == "resnet18":
        assert model.fc.out_features == 600
    elif architecture_name == "convnext_tiny":
        assert model.head.fc.out_features == 600
    else:
        raise ValueError()


def test_get_model_2():
    """Test loading `timm` model with checkpoint saved in `nn.DataParallel` wrapper."""
    # create checkpoint for testing
    model = timm.create_model(model_name="vit_small_patch16_224", pretrained=False)
    target_state_dict = model.state_dict()
    model = nn.DataParallel(model)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    # test function
    model = get_model(
        architecture_name="vit_small_patch16_224",
        pretrained=False,
        checkpoint_path=buffer,
    )
    state_dict = model.state_dict()
    assert target_state_dict.keys() == state_dict.keys()
    for k in target_state_dict.keys():
        torch.testing.assert_close(target_state_dict[k], state_dict[k])


@pytest.mark.parametrize(
    argnames=["architecture_name"],
    argvalues=[
        ("vit_small_patch16_224",),
        ("swin_tiny_patch4_window7_224",),
        ("resnet18",),
        ("convnext_tiny",),
    ],
)
def test_get_model_3(architecture_name: str):
    """Test loading `timm` model with checkpoint with different number of classes.

    In the test case, the pre-trained model has 1000 output neurons,
    the "saved" checkpoint has 88 output neurons, and the new model should have 600 output neurons.
    """
    # create checkpoint for testing
    model = timm.create_model(model_name=architecture_name, pretrained=False, num_classes=88)
    target_state_dict = model.state_dict()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    # test function
    model = get_model(
        architecture_name=architecture_name,
        pretrained=False,
        target_size=600,
        checkpoint_path=buffer,
    )
    state_dict = model.state_dict()
    assert target_state_dict.keys() == state_dict.keys()


@pytest.mark.parametrize(
    argnames=["architecture_name"],
    argvalues=[
        ("vit_small_patch16_224",),
        ("swin_tiny_patch4_window7_224",),
        ("resnet18",),
        ("convnext_tiny",),
    ],
)
def test_get_model_target_size(architecture_name: str):
    """Test method for identifying target size (number of output classes) of a `timm` model."""
    model = timm.create_model(model_name=architecture_name, pretrained=False, num_classes=422)
    target_size = get_model_target_size(model)
    assert target_size == 422
