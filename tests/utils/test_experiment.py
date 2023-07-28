from typing import Optional

import pytest

from fgvc.utils.experiment import load_train_args


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
