from typing import Callable, Union

import numpy as np


class ScoresMonitor:
    """Helper class for monitoring scores during training.

    Parameters
    ----------
    scores_fn
        Callable function for evaluating training scores.
        The function should accept `preds` and `targs` and return a dictionary with scores.
    num_samples
        Number of samples in the dataset.
    eval_batches
        If true the method evaluates scores on each mini-batch during training.
        Otherwise, it stores predictions and targets (`preds`, `targs`)
        and evaluates scores on full dataset.
        Set `eval_batches=False` in cases where all data points are needed to compute a score,
        e.g. F1 score in classification.
    store_preds_targs
        If true the method stores predictions and targets (`preds`, `targs`) for later use.
    """

    def __init__(
        self,
        scores_fn: Callable,
        num_samples: int,
        *,
        eval_batches: bool = True,
        store_preds_targs: bool = False,
    ):
        self.metrics_fc = scores_fn
        self.num_samples = num_samples
        self.eval_batches = eval_batches
        self.store_preds_targs = store_preds_targs

        # initialize score variables used for eager evaluation
        self._avg_scores = None

        # initialize (preds, targs) variables used for lazy evaluation
        self._i = 0
        self._bs = None
        self._preds_all = None
        self._targs_all = None

    def reset(self):
        """Reset internal variables including average scores and stored predictions and targets."""
        self._avg_scores = None
        self._i = 0
        self._bs = None
        self._preds_all = None
        self._targs_all = None

    def _update_scores(self, preds: np.ndarray, targs: np.ndarray):
        batch_scores = self.metrics_fc(preds, targs)
        batch_scores = {k: v / self.num_samples for k, v in batch_scores.items()}
        if self._avg_scores is None:
            self._avg_scores = batch_scores
        else:
            for k in self._avg_scores.keys():
                self._avg_scores[k] += batch_scores[k]

    def _store_preds_targs(self, preds: np.ndarray, targs: np.ndarray):
        if self._preds_all is None and self._targs_all is None:
            # initialize empty array
            self._preds_all = np.zeros((self.num_samples, *preds.shape[1:]), dtype=preds.dtype)
            self._targs_all = np.zeros((self.num_samples, *targs.shape[1:]), dtype=targs.dtype)
            self._bs = preds.shape[0]
            self._i = 0

        start_index = self._i * self._bs
        end_index = (self._i + 1) * self._bs
        self._preds_all[start_index:end_index] = preds
        self._targs_all[start_index:end_index] = targs
        self._i += 1

    def update(self, preds: np.ndarray, targs: np.ndarray):
        """Evaluate scores based on the given predictions and targets and update average scores.

        Parameters
        ----------
        preds
            Numpy array with predictions.
        targs
            Numpy array with ground-truth targets.
        """
        assert len(preds) == len(targs)
        if self.eval_batches:
            self._update_scores(preds, targs)

        if not self.eval_batches or self.store_preds_targs:
            self._store_preds_targs(preds, targs)

    @property
    def avg_scores(self) -> dict:
        """Get average scores."""
        if self.eval_batches:
            scores = self._avg_scores
        else:
            scores = self.metrics_fc(self._preds_all, self._targs_all)
        return scores

    @property
    def preds_all(self) -> np.ndarray:
        """Get stored predictions from the full dataset."""
        return self._preds_all

    @property
    def targs_all(self) -> np.ndarray:
        """Get stored predictions from the full dataset."""
        return self._targs_all


class LossMonitor:
    """Helper class for monitoring loss(es) during training.

    Parameters
    ----------
    num_batches
        Number of batches in dataloader.
    """

    def __init__(self, num_batches: int):
        self.num_batches = num_batches
        self._avg_loss = None

    def reset(self):
        """Reset internal variable average loss."""
        self._avg_loss = None

    def update(self, loss: Union[float, dict]):
        """Update average loss."""
        if isinstance(loss, float):
            if self._avg_loss is None:
                self._avg_loss = 0.0  # initialize average loss
            assert isinstance(self._avg_loss, float)
            self._avg_loss += loss / self.num_batches
        elif isinstance(loss, dict):
            if self._avg_loss is None:
                self._avg_loss = {k: 0.0 for k in loss.keys()}  # initialize average loss
            assert isinstance(self._avg_loss, dict)
            for k, v in loss.items():
                self._avg_loss[k] += v / self.num_batches
        else:
            raise ValueError()

    @property
    def avg_loss(self) -> float:
        """Get average loss."""
        if isinstance(self._avg_loss, float):
            avg_loss = self._avg_loss
        elif isinstance(self._avg_loss, dict):
            assert "loss" in self._avg_loss
            avg_loss = self._avg_loss["loss"]
        else:
            raise ValueError()
        return avg_loss

    @property
    def other_avg_losses(self) -> dict:
        """Get other average losses."""
        if isinstance(self._avg_loss, float):
            other_avg_losses = {}
        elif isinstance(self._avg_loss, dict):
            other_avg_losses = {k: v for k, v in self._avg_loss.items() if k != "loss"}
        else:
            raise ValueError()
        return other_avg_losses
