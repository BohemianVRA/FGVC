from typing import Union

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

from fgvc.core.metrics import classification_scores


def confidence_threshold_report(
    preds: np.ndarray,
    targs: np.ndarray,
) -> pd.DataFrame:
    """Evaluates classification scores for different confidence thresholds.

    Parameters
    ----------
    preds
        Numpy array with predictions.
    targs
        Numpy array with ground-truth targets.

    Returns
    -------
    scores_per_th
        A DataFrame with scores for each confidence threshold.
    """
    assert len(preds.shape) == 2
    assert len(targs.shape) == 1
    probs = softmax(preds, 1)

    scores_per_th = {}
    for th in np.arange(0, 1, 0.02):
        # use predictions with probability above threshold
        cond = (probs >= th).any(1)

        # evaluate metrics
        scores_sample = classification_scores(probs[cond], targs[cond], return_dict=True)
        scores_per_th[th] = {
            "Num Preds Made": cond.sum(),
            **scores_sample,
        }
    scores_per_th = pd.DataFrame.from_dict(scores_per_th, orient="index")

    return scores_per_th


def estimate_optimal_confidence_thresholds(
    preds: np.ndarray,
    targs: np.ndarray,
    error_rate: float = 0.05,
    *,
    return_df: bool = False,
) -> Union[dict, pd.DataFrame]:
    """Computes the optimal confidence threshold for each class.

    The threshold is based on the specified maximum misclassification error rate.

    Parameters
    ----------
    preds
        Numpy array with predictions.
    targs
        Numpy array with ground-truth targets.
    error_rate
        Expected misclassification rate used for optimizing confidence threshold for each class.
    return_df
        If true returns a DataFrame with additional information (accuracy).

    Returns
    -------
    confidence_thresholds
        If `return_df` is false returns a dictionary with class labels as keys
        and evaluated optimal thresholds as values.
        If `return_df` is true returns a DataFrame with class labels as index
        and evaluated optimal thresholds (`opt_th`) and corresponding accuracy (`acc`).
    """
    assert len(preds.shape) == 2
    assert len(targs.shape) == 1
    argmax_preds = preds.argmax(1)
    probs = softmax(preds, 1)
    confs = probs.max(1)

    confidence_thresholds = {}
    for label in np.unique(targs):
        # calculate  accuracies per class for different confidence thresholds
        # and use first confidence theshold where accuracy is higher than `(1 - error_rate)`
        opt_th, min_opt_acc = None, None
        for th in np.arange(0.01, 1, 0.01).round(2):
            # select records with predicted label and confidence higher than threshold
            cond = (argmax_preds == label) & (confs >= th)

            # compute accuracy and break if it is higher than `(1 - error_rate)`
            acc = accuracy_score(targs[cond], argmax_preds[cond]) if cond.sum() > 0 else 0
            if acc >= (1 - error_rate):
                opt_th, min_opt_acc = th, acc
                break

        # store results
        confidence_thresholds[label] = {"opt_th": opt_th, "acc": min_opt_acc}

    if return_df:
        confidence_thresholds = pd.DataFrame.from_dict(confidence_thresholds, orient="index")
    else:
        confidence_thresholds = {k: v["opt_th"] for k, v in confidence_thresholds.items()}

    return confidence_thresholds


def class_wise_confidence_threshold_report(
    preds: np.ndarray,
    targs: np.ndarray,
    confidence_thresholds: dict,
    *,
    target_names: list = None,
) -> pd.DataFrame:
    """Evaluates classification scores based on the optimal confidence threshold for each class.

    Parameters
    ----------
    preds
        Numpy array with predictions.
    targs
        Numpy array with ground-truth targets.
    confidence_thresholds
        A dictionary with class labels as keys and evaluated optimal thresholds as values.
        It is computed using `get_optimal_confidence_thresholds` method.
    target_names
        Class names to use in the DataFrame report instead of class ids.

    Returns
    -------
    scores_per_class
        A DataFrame with scores for each class and macro averaged total scores.
    """
    assert len(preds.shape) == 2
    assert len(targs.shape) == 1
    argmax_preds = preds.argmax(1)
    probs = softmax(preds, 1)
    confs = probs.max(1)
    labels = np.arange(preds.shape[1])

    # compute classification scores based on the confidence_thresholds for each class
    scores_per_class = {}
    for label in np.unique(targs):
        th = confidence_thresholds[label]
        num_records = (argmax_preds == label).sum()
        num_preds_made, freq_preds_made, label_scores = None, None, {}
        if th is not None:
            cond = (argmax_preds == label) & (confs >= th)
            num_preds_made = cond.sum()
            if num_preds_made > 0:
                freq_preds_made = num_preds_made / num_records
                label_scores = classification_scores(
                    preds[cond], targs[cond], top_k=None, return_dict=True
                )
                # compute F1 score for the current class instead of macro averaged F1 score
                label_scores["F1"] = f1_score(
                    targs[cond], argmax_preds[cond], labels=labels, average=None, zero_division=0
                )[label]

        label_str = label
        if target_names is not None and isinstance(label, (int, np.integer)):
            label_str = target_names[label]
        scores_per_class[label_str] = {
            "Num Records": num_records,
            "Num Preds Made": num_preds_made,
            "Frac Preds Made": freq_preds_made,
            **label_scores,
        }
    scores_per_class = pd.DataFrame.from_dict(scores_per_class, orient="index")

    # compute total classification scores based on the confidence_thresholds for each class
    confs_th = np.array([confidence_thresholds[x] or np.inf for x in argmax_preds])
    cond = confs >= confs_th
    total_scores = classification_scores(preds[cond], targs[cond], top_k=None, return_dict=True)
    scores_per_class.loc["macro avg"] = {
        "Num Records": len(cond),
        "Num Preds Made": cond.sum(),
        "Frac Preds Made": cond.sum() / len(cond),
        **total_scores,
    }

    return scores_per_class
