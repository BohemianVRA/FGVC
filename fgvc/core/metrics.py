from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def classification_scores(
    preds: np.ndarray, targs: np.ndarray, *, top_k: Optional[int] = 3
) -> Tuple[float, float, float]:
    """Compute top-1 and top-k accuracy and f1 score.

    Parameters
    ----------
    preds
        Numpy array with predictions.
    targs
        Numpy array with ground-truth targets.
    top_k
        Value of k to compute top k accuracy.

    Returns
    -------
    acc
        Accuracy score.
    topk
        Top k accuracy score.
    f1
        F1 score.
    """
    preds_argmax = preds.argmax(1)
    labels = np.arange(preds.shape[1])
    acc = accuracy_score(targs, preds_argmax)
    acc_k = None
    if top_k is not None and preds.shape[1] > top_k:
        acc_k = top_k_accuracy_score(targs, preds, k=top_k, labels=labels)
    f1 = f1_score(targs, preds_argmax, labels=labels, average="macro")
    return acc, acc_k, f1
