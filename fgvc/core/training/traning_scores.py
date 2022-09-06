import numpy as np

from fgvc.core.metrics import classification_scores
from fgvc.utils.wandb import log_clf_progress


class TrainingScores:
    """Class for evaluating scores and preparing them to log.

    Parameters
    ----------
    elapsed_epoch_time
        Number of seconds past during training and validation of one epoch.
    train_preds
        Numpy array with training predictions.
    train_targs
        Numpy array with training ground-truth targets.
    train_loss
        Average training loss.
    valid_preds
        Numpy array with validation predictions.
    valid_targs
        Numpy array with validation ground-truth targets.
    valid_loss
        Average validation loss.
    """

    def __init__(
        self,
        elapsed_epoch_time: float,
        train_preds: np.ndarray,
        train_targs: np.ndarray,
        train_loss: float,
        valid_preds: np.ndarray = None,
        valid_targs: np.ndarray = None,
        valid_loss: float = None,
    ):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.elapsed_epoch_time = elapsed_epoch_time

        # evaluate metrics
        self.train_acc, _, self.train_f1 = classification_scores(
            train_preds, train_targs, top_k=None
        )
        self.valid_acc, self.valid_acc3, self.valid_f1 = None, None, None
        if valid_preds is not None and valid_targs is not None:
            self.valid_acc, self.valid_acc3, self.valid_f1 = classification_scores(
                valid_preds, valid_targs, top_k=3
            )

    def log_wandb(self, epoch: int, lr: float):
        """Log evaluated scores to WandB.

        Parameters
        ----------
        epoch
            Epoch number.
        lr
            Current learning rate used by optimizer.
        """
        log_clf_progress(
            epoch,
            train_loss=self.train_loss,
            valid_loss=self.valid_loss,
            train_acc=self.train_acc,
            train_f1=self.train_f1,
            valid_acc=self.valid_acc,
            valid_acc3=self.valid_acc3,
            valid_f1=self.valid_f1,
            lr=lr,
        )

    def to_str(self):
        """Convert evaluated scores to string for logging."""
        scores = {
            "avg_train_loss": str(np.round(self.train_loss, 4)),
            "avg_val_loss": str(np.round(self.valid_loss, 4)),
            "F1": str(np.round(self.valid_f1 * 100, 2)),
            "Acc": str(np.round(self.valid_acc * 100, 2)),
            "Recall@3": str(np.round(self.valid_acc3 * 100, 2)),
            "time": f"{self.elapsed_epoch_time:.0f}s",
        }
        scores_str = "\t".join([f"{k}: {v}" for k, v in scores.items()])
        return scores_str

    def get_checkpoint_metrics(self) -> dict:
        """Get dictionary with metrics to use for saving checkpoints (besides loss).

        E.g. save checkpoints during training with the best accuracy or f1 scores.
        """
        return {"accuracy": self.valid_acc, "f1": self.valid_f1}
