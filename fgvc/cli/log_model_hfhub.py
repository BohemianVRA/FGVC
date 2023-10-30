import argparse
import logging

from fgvc.utils.hfhub import export_to_huggingface_hub_from_checkpoint

logger = logging.getLogger("script")


def hfhub_load_args():
    """Load script arguments via `argparse` library."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-path",
        help="Path to a exp directory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--repo-owner",
        help="Name of the HuggingFace repository owner (shortcut).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--saved-model",
        help="Specify to select a specific model to export (accuracy, f1, loss, "
        "recall, last_epoch). The 'best_accuracy.pth' is the default.",
        type=str,
        required=False,
    )
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def log_model_hfhub(
    *, exp_path: str = None, repo_owner: str = None, saved_model: str = None
) -> str:
    """Exports a saved model to the HuggingFace Hub."""
    if exp_path is None or repo_owner is None:
        args, extra_args = hfhub_load_args()
        exp_path = args.exp_path
        repo_owner = args.repo_owner
        saved_model = args.saved_model

    model_repo_name = export_to_huggingface_hub_from_checkpoint(
        exp_path=exp_path, repo_owner=repo_owner, saved_model=saved_model
    )
    logger.info(f"Uploaded model to HuggingFace Hub repo: {model_repo_name}")
    return model_repo_name


if __name__ == "__main__":
    log_model_hfhub()
