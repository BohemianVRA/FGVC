import argparse
import logging

from fgvc.utils.hfhub import export_model_to_huggingface_hub_from_checkpoint

logger = logging.getLogger("script")


def hfhub_load_args() -> tuple[argparse.Namespace, list[str]]:
    """ Load script arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-path",
        help="Path to a exp directory with a valid config.yaml file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--repo-owner",
        help="Name of the HuggingFace repository owner (shortcut).",
        type=str,
        default=True,
    )
    parser.add_argument(
        "--saved-model",
        help="Specify to select a specific model to export (accuracy, f1, loss, "
             "recall, last_epoch).",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model-card",
        help="Contents of the model card file.",
        type=str,
        required=False,
    )
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def export_to_hfhub(
        *,
        exp_path: str = None,
        repo_owner: str = None,
        saved_model: str = None,
        model_card: str = None,
) -> str:
    """Wraps the export_to_huggingface_hub_from_checkpoint() with a CLI interface.
    Can be run from CLI with 'python hfhub.py --exp-path <exp_path> --repo-owner <repo_owner>
    (optionally --saved-model <saved_model> --model-card <model_card>)'
    '"""
    if exp_path is None or repo_owner is None:
        args, extra_args = hfhub_load_args()
        config = {"exp_path": args.exp_path}
        repo_owner = args.repo_owner
        saved_model = args.saved_model
        model_card = args.model_card
    else:
        config = {"exp_path": exp_path}

    model_repo_name = export_model_to_huggingface_hub_from_checkpoint(
        config=config,
        repo_owner=repo_owner,
        saved_model=saved_model,
        model_card=model_card,
    )
    logger.info(f"Uploaded model to HuggingFace Hub repo: {model_repo_name}")
    return model_repo_name


if __name__ == "__main__":
    export_to_hfhub()
