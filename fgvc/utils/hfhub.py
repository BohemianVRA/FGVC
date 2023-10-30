import json
import logging
import os
import os.path as osp
import warnings
from functools import wraps

import torch
import yaml

try:
    import huggingface_hub

    # verify package import not local dir
    assert hasattr(huggingface_hub, "__version__")
    HuggingFaceAPI = huggingface_hub.HfApi
    HFHubCreateRepo = huggingface_hub.create_repo
except (ImportError, AssertionError):
    huggingface_hub = None
    HuggingFaceAPI = None
    HFHubCreateRepo = None

logger = logging.getLogger("fgvc")


def if_huggingface_hub_is_installed(func: callable):
    """A decorator function that checks if the HuggingFaceHub library is installed."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if huggingface_hub is not None:
            return func(*args, **kwargs)
        else:
            warnings.warn("Library huggingface_hub is not installed.")

    return decorator


# Used to match the saved model names
SAVED_MODEL_NAMES = {
    "accuracy": "best_accuracy",
    "f1": "best_f1",
    "loss": "best_loss",
    "recall": "best_recall@3",
    "last_epoch": "epoch",
}


@if_huggingface_hub_is_installed
def export_to_huggingface_hub_from_checkpoint(
    *, exp_path: str = None, repo_owner: str = None, saved_model: str = None
) -> str:
    """Exports a saved model to the HuggingFace Hub.

    Creates a new model repo if it does not exist. If it does exist,
    the pytorch_model.bin and config.json files will be overwritten!
    Can be run from CLI with 'python hfhub.py --exp-path <exp_path>
     --repo-owner <repo_owner> (optionally --saved-model <saved_model>)'

    Parameters
    ----------
    exp_path
        A dictionary with experiment configuration. Must contain config.yaml and
        the appropriate model.pth file. Config must have "architecture", "image_size", and
         "number_of_classes" to create the timm config.json.
    repo_owner
        The "shortcut" of the HuggingFace repository owner name (owner_name/repository_name).
    saved_model
        (optional) String key to select the saved model to export (accuracy, f1, loss, recall, last_epoch).
        best_accuracy.pth is the default.

    Returns
    -------
    repo_name
        The whole HuggingFace repository name suitable to download the model through timm.
    """
    api = HuggingFaceAPI()

    saved_model_type = SAVED_MODEL_NAMES.get(saved_model, "best_accuracy")

    # create new experiment directory or use existing one
    file_names = os.listdir(exp_path)

    model_path = None
    for file_name in file_names:
        if file_name.endswith(".pth") and saved_model_type in file_name:
            model_path = osp.join(exp_path, file_name)
            break
    assert osp.exists(model_path), f"Model path {model_path} does not exist."

    # Save selected model saves as bin
    model = torch.load(model_path)
    model_bin_path = f'{model_path.removesuffix(".pth")}.bin'

    logging.info(f"Saving model to {model_bin_path}")

    torch.save(model, model_bin_path)

    # timm supports specific config.json file
    config_path_yaml = osp.join(exp_path, "config.yaml")
    assert osp.exists(config_path_yaml), f"Config path {config_path_yaml} does not exist."
    with open(config_path_yaml, "r") as fp:
        config_data = yaml.safe_load(fp)

    config_path_json = osp.join(exp_path, "config.json")
    _create_timm_config(config_data, config_path_json)

    run_name_exp_path = osp.normpath(exp_path).split(os.sep)[-2:]
    repo_name = f"{repo_owner}/FGVC-{'-'.join(run_name_exp_path)}"

    logging.info(f"Creating new repository: {repo_name}")

    try:
        HFHubCreateRepo(repo_id=repo_name, repo_type="model", exist_ok=True)
        # Upload model
        api.upload_file(
            path_or_fileobj=model_bin_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_name,
            repo_type="model",
        )
        # Upload config
        api.upload_file(
            path_or_fileobj=config_path_json,
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
        )

    except Exception as exp:
        logging.warning(f"Error while uploading files to HuggingFace Hub:\n{exp}")

    return repo_name


def _create_timm_config(config, config_path_json):
    """Create timm config.json file."""
    timm_config = {
        "architecture": config["architecture"],
        "input_size": [3, *config["image_size"]],  # assumes 3 color channels
        "num_classes": config["number_of_classes"],
    }

    with open(config_path_json, "w") as fp:
        json.dump(timm_config, fp, indent=4)
