import math

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from fgvc.core.training.scores_monitor import ScoresMonitor


def _classification_scores(preds, targs):
    if len(preds.shape) == 2:
        preds = preds.argmax(-1)
    return dict(
        acc=accuracy_score(targs, preds),
        f1=f1_score(targs, preds, average="macro"),
    )


@pytest.mark.parametrize(
    argnames=["batch_size"],
    argvalues=[(4,), (5,), (8,)],
)
@pytest.mark.parametrize(
    argnames=["store_preds_targs"],
    argvalues=[(False,), (True,)],
)
def test_scores_monitor_1(batch_size: int, store_preds_targs: bool):
    """Test `ScoresMonitor` class that monitors scores during training.

    Test evaluation of all records after training (F1 score for classification).
    """
    preds_all = np.array([1, 2, 3, 2, 2, 3, 1, 2, 3, 0])
    targs_all = np.array([3, 2, 1, 2, 1, 2, 3, 2, 0, 0])
    preds_all_one_hot = OneHotEncoder(sparse_output=False).fit_transform(preds_all[..., None])
    target_scores = _classification_scores(preds_all, targs_all)
    num_batches = math.ceil(len(preds_all) / batch_size)
    preds_list = [
        preds_all_one_hot[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]
    targs_list = [targs_all[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    scores_monitor = ScoresMonitor(
        scores_fn=_classification_scores,
        num_samples=len(preds_all),
        eval_batches=False,
        store_preds_targs=store_preds_targs,
    )
    for preds, targs in zip(preds_list, targs_list):
        scores_monitor.update(preds, targs)
    scores = scores_monitor.avg_scores
    assert np.isclose(scores["acc"], target_scores["acc"])
    assert np.isclose(scores["f1"], target_scores["f1"])
    assert np.allclose(scores_monitor.preds_all, preds_all_one_hot)
    assert np.allclose(scores_monitor.targs_all, targs_all)
    assert scores_monitor._i == num_batches
    assert scores_monitor._bs == batch_size


def _segmentation_scores(preds, targs, mean_reduction=False):
    acc_scores, f1_scores = [], []
    for pred, targ in zip(preds, targs):
        pred, targ = pred.reshape(-1), targ.reshape(-1)
        acc_scores.append(accuracy_score(pred, targ))
        f1_scores.append(f1_score(pred, targ, average="macro"))
    return dict(
        acc=np.mean(acc_scores) if mean_reduction else np.sum(acc_scores),
        f1=np.mean(f1_scores) if mean_reduction else np.sum(f1_scores),
    )


@pytest.mark.parametrize(
    argnames=["store_preds_targs"],
    argvalues=[(False,), (True,)],
)
def test_scores_monitor_2(store_preds_targs: bool):
    """Test `ScoresMonitor` class that monitors scores during training.

    Test evaluation in batches (F1 score on individual images for segmentation).
    """
    batch_size = 3
    preds_all = np.random.randint(0, 4, size=(8, 1, 24, 24))
    targs_all = np.random.randint(0, 4, size=(8, 1, 24, 24))
    target_scores = _segmentation_scores(preds_all, targs_all, mean_reduction=True)
    num_batches = math.ceil(len(preds_all) / batch_size)
    preds_list = [preds_all[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    targs_list = [targs_all[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    scores_monitor = ScoresMonitor(
        scores_fn=_segmentation_scores,
        num_samples=len(preds_all),
        eval_batches=True,
        store_preds_targs=store_preds_targs,
    )
    for preds, targs in zip(preds_list, targs_list):
        scores_monitor.update(preds, targs)
    scores = scores_monitor.avg_scores
    assert np.isclose(scores["acc"], target_scores["acc"])
    assert np.isclose(scores["f1"], target_scores["f1"])
    if store_preds_targs:
        assert np.allclose(scores_monitor.preds_all, preds_all)
        assert np.allclose(scores_monitor.targs_all, targs_all)
        assert scores_monitor._i == num_batches
        assert scores_monitor._bs == batch_size
    else:
        assert scores_monitor.preds_all is None
        assert scores_monitor.targs_all is None


@pytest.mark.parametrize(
    argnames=["batch_size"],
    argvalues=[(4,), (5,), (8,)],
)
def test_scores_monitor_3(batch_size: int):
    """Test `ScoresMonitor` class that monitors scores during training.

    Test having predictions as dictionary of arrays.
    """

    def _classification_scores_2(preds, targs):
        return _classification_scores(preds["p1"], targs)

    preds_all = dict(
        p1=np.array([1, 2, 3, 2, 2, 3, 1, 2, 3, 0]),
        p2=np.random.randn(10, 72),
    )
    targs_all = np.array([3, 2, 1, 2, 1, 2, 3, 2, 0, 0])
    preds_all_one_hot = dict(
        p1=OneHotEncoder(sparse_output=False).fit_transform(preds_all["p1"][..., None]),
        p2=preds_all["p2"],
    )
    target_scores = _classification_scores_2(preds_all, targs_all)
    num_batches = math.ceil(len(preds_all["p1"]) / batch_size)
    preds_list = [
        dict(
            p1=preds_all_one_hot["p1"][i * batch_size : (i + 1) * batch_size],
            p2=preds_all_one_hot["p2"][i * batch_size : (i + 1) * batch_size],
        )
        for i in range(num_batches)
    ]
    targs_list = [targs_all[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    scores_monitor = ScoresMonitor(
        scores_fn=_classification_scores_2,
        num_samples=len(preds_all["p1"]),
        eval_batches=False,
    )
    for preds, targs in zip(preds_list, targs_list):
        scores_monitor.update(preds, targs)
    scores = scores_monitor.avg_scores
    assert np.isclose(scores["acc"], target_scores["acc"])
    assert np.isclose(scores["f1"], target_scores["f1"])
    assert np.allclose(scores_monitor.preds_all["p1"], preds_all_one_hot["p1"])
    assert np.allclose(scores_monitor.preds_all["p2"], preds_all_one_hot["p2"])
    assert np.allclose(scores_monitor.targs_all, targs_all)
    assert scores_monitor._i == num_batches
    assert scores_monitor._bs == batch_size


def test_scores_monitor_4():
    """Test `ScoresMonitor` class that monitors scores during training.

    Test raising exceptions when elements with different batch sizes are passed.
    """
    scores_monitor = ScoresMonitor(
        scores_fn=_classification_scores,
        num_samples=10,
        eval_batches=False,
    )
    with pytest.raises(ValueError):
        scores_monitor.update(np.array([1, 2, 3]), np.array([3, 2]))
    with pytest.raises(ValueError):
        scores_monitor.update(
            dict(p1=np.array([1, 2, 3]), p2=np.array([4, 2])), np.array([3, 2, 1])
        )
    with pytest.raises(ValueError):
        scores_monitor.update(
            dict(p1=np.array([1, 2, 3]), p2=np.array([4, 2, 1])), np.array([3, 2])
        )
    with pytest.raises(ValueError):
        scores_monitor.update(
            np.array([3, 2, 1]),
            dict(p1=np.array([1, 2, 3]), p2=np.array([4, 2])),
        )
    with pytest.raises(ValueError):
        scores_monitor.update(
            np.array([3, 2]),
            dict(p1=np.array([1, 2, 3]), p2=np.array([4, 2, 1])),
        )
