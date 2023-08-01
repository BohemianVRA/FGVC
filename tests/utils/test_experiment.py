from typing import Optional
from unittest.mock import MagicMock, mock_open, patch

import pytest

from fgvc.utils.experiment import load_config, load_train_args


@pytest.mark.parametrize(
    argnames=[
        "target_cuda_devices",
        "target_wandb_entity",
        "target_wandb_project",
        "target_resume_exp_name",
        "target_root_path",
        "target_architecture",
    ],
    argvalues=[
        ("0", "zcu", "test", "exp1", "/data/", "vit_base"),
        ("0", None, None, "exp1", "/data/", "vit_base"),
        ("0", "zcu", "test", "exp1", "/data/", None),
        ("0", "zcu", "test", "exp1", None, None),
        (None, None, None, None, None, None),
    ],
)
def test_load_train_args(
    target_cuda_devices: Optional[str],
    target_wandb_entity: Optional[str],
    target_wandb_project: Optional[str],
    target_resume_exp_name: Optional[str],
    target_root_path: Optional[str],
    target_architecture: Optional[str],
):
    """Test loading training arguments."""
    target_config_path = "./configs/some_config.yaml"

    input_args = ["--config-path", target_config_path]
    if target_cuda_devices is not None:
        input_args.append("--cuda-devices")
        input_args.append(target_cuda_devices)
    if target_wandb_entity is not None:
        input_args.append("--wandb-entity")
        input_args.append(target_wandb_entity)
    if target_wandb_project is not None:
        input_args.append("--wandb-project")
        input_args.append(target_wandb_project)
    if target_resume_exp_name is not None:
        input_args.append("--resume-exp-name")
        input_args.append(target_resume_exp_name)
    if target_root_path is not None:
        input_args.append("--root-path")
        input_args.append(target_root_path)
    if target_architecture is not None:
        input_args.append("--architecture")
        input_args.append(target_architecture)
    args, extra_args = load_train_args(input_args)
    assert args.config_path == target_config_path
    if target_cuda_devices is not None:
        assert args.cuda_devices == target_cuda_devices
    if target_wandb_entity is not None:
        assert args.wandb_entity == target_wandb_entity
    if target_wandb_project is not None:
        assert args.wandb_project == target_wandb_project
    if target_resume_exp_name is not None:
        assert args.resume_exp_name == target_resume_exp_name
    if target_root_path is not None:
        assert extra_args.get("root-path") == target_root_path
    if target_architecture is not None:
        assert extra_args.get("architecture") == target_architecture


@pytest.fixture(scope="module")
def input_config():
    return {
        # data
        "augmentations": "light",
        "image_size": [224, 224],  # [height, width]
        "dataset": "DF20-Mini",
        # model
        "architecture": "vit_base_patch32_224",
        # training
        "loss": "CrossEntropyLoss",
        "optimizer": "SGD",
        "scheduler": "cosine",
        "epochs": 40,
        "learning_rate": 0.001,
        "batch_size": 64,
        "accumulation_steps": 1,
        # other
        "random_seed": 777,
        "workers": 4,
        "multigpu": False,
        "tags": ["baseline"],  # W&B Run tags
    }


@patch("builtins.open", new_callable=mock_open, read_data=None)
@patch("fgvc.utils.experiment.yaml.safe_load")
@patch("os.makedirs")
def test_load_config_1(
    makedirs: MagicMock, safe_load: MagicMock, open_: MagicMock, input_config: dict
):
    """Test loading configuration YAML file and setting up experiment path."""
    target_exp_path = "./runs/vit_base_patch32_224-CrossEntropyLoss-light/exp1"
    safe_load.return_value = input_config
    config = load_config(
        config_path="dummy_config.yaml", run_name_fmt="architecture-loss-augmentations"
    )
    makedirs.assert_called_once_with(target_exp_path, exist_ok=False)
    for k in input_config.keys():
        assert input_config[k] == config[k], k
    assert "exp_name" in config
    assert config["exp_name"] == "exp1"
    assert "exp_path" in config
    assert config["exp_path"] == target_exp_path
    assert "run_name" in config
    assert config["run_name"] == "vit_base_patch32_224-CrossEntropyLoss-light"


@patch("builtins.open", new_callable=mock_open, read_data=None)
@patch("fgvc.utils.experiment.yaml.safe_load")
@patch("os.makedirs")
def test_load_config_2(
    makedirs: MagicMock, safe_load: MagicMock, open_: MagicMock, input_config: dict
):
    """Test loading configuration YAML file and setting up experiment path.

    Test `run_name_fmt` method argument.
    """
    target_exp_path = "./runs/DF20-Mini-40-0.001/exp1"
    safe_load.return_value = input_config
    config = load_config(
        config_path="dummy_config.yaml", run_name_fmt="dataset-epochs-learning_rate"
    )
    makedirs.assert_called_once_with(target_exp_path, exist_ok=False)
    for k in input_config.keys():
        assert input_config[k] == config[k], k
    assert "exp_path" in config
    assert config["exp_path"] == target_exp_path
    assert "run_name" in config
    assert config["run_name"] == "DF20-Mini-40-0.001"


@patch("builtins.open", new_callable=mock_open, read_data=None)
@patch("fgvc.utils.experiment.yaml.safe_load")
@patch("os.makedirs")
def test_load_config_3(
    makedirs: MagicMock, safe_load: MagicMock, open_: MagicMock, input_config: dict
):
    """Test loading configuration YAML file and setting up experiment path.

    Test `root_path` config parameter.
    """
    # test root_path in config
    input_config_ = {**input_config, "root_path": "/data/project1/"}
    target_exp_path = "/data/project1/runs/vit_base_patch32_224-CrossEntropyLoss-light/exp1"
    safe_load.return_value = input_config_
    config = load_config(
        config_path="dummy_config.yaml", run_name_fmt="architecture-loss-augmentations"
    )
    makedirs.assert_called_once_with(target_exp_path, exist_ok=False)
    for k in input_config_.keys():
        assert input_config_[k] == config[k], k
    assert "exp_path" in config
    assert config["exp_path"] == target_exp_path

    # test root_path in method argument
    target_exp_path = "/data/project1/runs/vit_base_patch32_224-CrossEntropyLoss-light/exp1"
    safe_load.return_value = input_config
    config = load_config(
        config_path="dummy_config.yaml",
        run_name_fmt="architecture-loss-augmentations",
        extra_args={"root_path": "/data/project1/"},
    )
    for k in input_config_.keys():
        assert input_config_[k] == config[k], k
    assert "exp_path" in config
    assert config["exp_path"] == target_exp_path
