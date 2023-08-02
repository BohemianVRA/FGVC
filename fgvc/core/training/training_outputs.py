from typing import NamedTuple, Optional, Union

import numpy as np


class BatchOutput(NamedTuple):
    """Tuple returned from `train_batch` and `predict_batch` trainer methods."""

    preds: np.ndarray
    targs: np.ndarray
    loss: Union[float, dict]


class TrainEpochOutput(NamedTuple):
    """Tuple returned from `train_epoch` trainer method."""

    avg_loss: float
    avg_scores: Optional[dict] = {}
    max_grad_norm: Optional[float] = None
    other_avg_losses: Optional[dict] = {}


class PredictOutput(NamedTuple):
    """Tuple returned from `predict` trainer method."""

    preds: Optional[np.ndarray] = None
    targs: Optional[np.ndarray] = None
    avg_loss: Optional[float] = np.nan
    avg_scores: Optional[dict] = {}
