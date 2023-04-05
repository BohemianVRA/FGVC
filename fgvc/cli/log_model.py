import argparse
import logging
import os

from fgvc.utils.wandb import wandb

logger = logging.getLogger("script")


def load_args(args: str = None) -> argparse.Namespace:
    """Load script arguments using `argparse` library."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-run-id",
        help="Run id for logging experiment to wandb.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wandb-project",
        help="Project name for logging experiment to wandb.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wandb-entity",
        help="Entity name for logging experiment to wandb.",
        type=str,
        required=True,
    )
    args = parser.parse_args(args)
    return args


def log_model(
    *,
    wandb_entity: str = None,
    wandb_project: str = None,
    wandb_run_id: str = None,
    **kwargs,
):
    """Log model from W&B experiment run as a W&B artifact."""
    if wandb is None:
        raise ImportError("Package wandb is not installed.")

    if wandb_entity is None or wandb_project is None or wandb_run_id is None:
        # load script args
        args = load_args()
        wandb_entity = args.wandb_entity
        wandb_project = args.wandb_project
        wandb_run_id = args.wandb_run_id

    # resume wandb run
    wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
    logger.info(f"Logging model in W&B from experiment run: {wandb_run_path}")
    with wandb.init(
        id=wandb_run_id,
        project=wandb_project,
        entity=wandb_entity,
        save_code=False,
        resume="must",
    ) as run:
        model_weights = os.path.join(run.config["exp_path"], "best_loss.pth")
        if not os.path.isfile(model_weights):
            raise ValueError(f"Model checkpoint '{model_weights}' not found.")
        logger.info(f"Using model checkpoint from the file: {model_weights}")
        arch_name = run.config["architecture"]

        # log artifact
        artifact_name = f"{run.name}-{arch_name}"
        artifact = wandb.Artifact(artifact_name, type="model", metadata=None)
        artifact.add_file(local_path=model_weights)
        run.log_artifact(artifact)
        logger.info(f"Created W&B artifact: {artifact_name}")


if __name__ == "__main__":
    log_model()
