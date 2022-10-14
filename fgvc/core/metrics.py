from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
    top_k_accuracy_score,
)


def classification_scores(
    preds: np.ndarray, targs: np.ndarray, *, top_k: Optional[int] = 3, return_dict: bool = False
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
    return_dict
        If True, the method returns dictionary with metrics.

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

    if return_dict:
        out = {"Acc": acc, f"Recall@{top_k}": acc_k, "F1": f1}
    else:
        out = acc, acc_k, f1

    return out


def binary_segmentation_tp_fp_fn_tn(
    preds: np.ndarray, targs: np.ndarray, pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute values of confusion matrix for binary segmentation.

    Parameters
    ----------
    preds
        Array [b, (2), h, w] with predicted values.
    targs
        Array [b, h, w] with target values.
    pos_label
        Class to report in the binary classification.

    Returns
    -------
    tp
        Array [b,] with True Positives for each sample.
    fp
        Array [b,] with False Positives for each sample.
    fn
        Array [b,] with False Negatives for each sample.
    tn
        Array [b,] with True Negatives for each sample.
    """
    assert len(targs.shape) == 3
    assert pos_label in (0, 1)
    if len(preds.shape) == 4:
        preds = preds.argmax(1)
    assert preds.shape == targs.shape
    assert np.all(np.isin(preds, [0, 1])), "The method supports only binary classification"
    assert np.all(np.isin(targs, [0, 1])), "The method supports only binary classification"

    # reshape samples
    num_samples = preds.shape[0]
    _preds = preds.reshape(num_samples, -1)
    _targs = targs.reshape(num_samples, -1)

    # compute confusion matrices per each sample (sklearn)
    mcm = multilabel_confusion_matrix(_targs, _preds, samplewise=True)
    ((tn, fn), (fp, tp)) = mcm.T

    # compute confusion matrices per each sample (custom faster but memory intensive implementation)
    # binary implementation
    # targs_bin = (_targs == pos_label).astype(targs.dtype)
    # preds_bin = (_preds == pos_label).astype(preds.dtype)
    # tp = (targs_bin * preds_bin).sum(axis=1)  # true positive
    # fp = ((1 - targs_bin) * preds_bin).sum(axis=1)  # false positive
    # fn = (targs_bin * (1 - preds_bin)).sum(axis=1)  # false negative
    # tn = ((1 - targs_bin) * (1 - preds_bin)).sum(axis=1)  # true negative

    return tp, fp, fn, tn


def binary_segmentation_scores(preds: np.ndarray, targs: np.ndarray, reduction: str = "mean") -> dict:
    """Compute segmentation scores Balanced Accuracy, Precision, Recall, F1, and IoU.

    Parameters
    ----------
    preds
        Array with predicted values.
    targs
        Array with target values.
    reduction
        Whether to aggregate scores using mean, sum, or not aggregate at all.

    Returns
    -------
    scores
        A dictionary with segmentation scores.
    """

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    assert reduction in ("mean", "sum", None)

    # compute batch-wise confusion matrices
    tp, fp, fn, tn = binary_segmentation_tp_fp_fn_tn(preds, targs, pos_label=1)

    # precision, recall, specificity
    p = divide(tp, (tp + fp))
    r = divide(tp, (tp + fn))  # sensitivity
    spec = divide(tn, (tn + fp))

    # compute scores
    scores = {
        # "acc": (tp + tn) / (tp + tn + fp + fn)
        "balanced_acc": (r + spec) / 2,
        "recall": r,
        "precision": p,
        # "specificity": spec,
        "f1": divide((2 * p * r), (p + r)),
        "iou": divide(tp, (tp + fp + fn)),
    }
    if reduction == "mean":
        scores = {k: v.mean() for k, v in scores.items()}
    elif reduction == "sum":
        scores = {k: v.sum() for k, v in scores.items()}

    return scores
