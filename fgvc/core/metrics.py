from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def classification_scores(
    preds: np.ndarray, targs: np.ndarray, *, top_k: Optional[int] = 3, prefix: str = ""
) -> dict:
    """Compute top-1 and top-k accuracy and f1 score.

    Parameters
    ----------
    preds
    targs
    top_k

    Returns
    -------
    acc
    topk
    f1
    """
    preds_argmax = preds.argmax(1)
    labels = np.arange(preds.shape[1])
    if len(prefix) > 0 and prefix[-1] != " ":
        prefix += " "
    scores = {}
    scores[f"{prefix}Accuracy"] = accuracy_score(targs, preds_argmax)
    if top_k is not None and preds.shape[1] > top_k:
        scores[f"{prefix}Recall@{top_k}"] = top_k_accuracy_score(
            targs, preds, k=top_k, labels=labels
        )
    scores[f"{prefix}F1"] = f1_score(
        targs, preds_argmax, labels=labels, average="macro"
    )
    return scores
