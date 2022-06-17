import tqdm
import torch

import numpy as np

from scipy import stats
from scipy.special import softmax
from sklearn.metrics import accuracy_score


def max_logits_performance(test_metadata):

    test_metadata["max_logits"] = [np.max(row) for row in test_metadata["logits"]]
    test_metadata["observation_max"] = None

    ObservationIds = test_metadata.ObservationId.unique()

    for obs_id in ObservationIds:
        obs_images = test_metadata[test_metadata["ObservationId"] == obs_id]
        max_index = obs_images.index[np.argmax(np.array(obs_images["max_logits"]))]
        for index, pred in obs_images.iterrows():
            test_metadata.at[index, "observation_max"] = test_metadata["preds"][
                max_index
            ]

    test_metadata_obs = test_metadata.drop_duplicates(subset=["ObservationId"])
    max_logits_accuracy = accuracy_score(
        test_metadata_obs["class_id"],
        test_metadata_obs["observation_max"].astype("int32"),
    )

    return np.round(max_logits_accuracy * 100, 2)


def mean_logits_performance(test_metadata):

    test_metadata["observation_mean"] = None

    ObservationIds = test_metadata.ObservationId.unique()

    for obs_id in ObservationIds:
        obs_images = test_metadata[test_metadata["ObservationId"] == obs_id]

        max_index = np.argmax(sum(obs_images["logits"]))
        for index, pred in obs_images.iterrows():
            test_metadata.at[index, "observation_mean"] = max_index

    test_metadata_obs = test_metadata.drop_duplicates(subset=["ObservationId"])
    mean_logits_accuracy = accuracy_score(
        test_metadata_obs["class_id"],
        test_metadata_obs["observation_mean"].astype("int32"),
    )

    return np.round(mean_logits_accuracy * 100, 2)


def max_softmax_performance(test_metadata):

    test_metadata["max_softmax"] = [np.max(row) for row in test_metadata["softmax"]]
    test_metadata["observation_max"] = None

    ObservationIds = test_metadata.ObservationId.unique()

    for obs_id in ObservationIds:
        obs_images = test_metadata[test_metadata["ObservationId"] == obs_id]
        max_index = obs_images.index[np.argmax(np.array(obs_images["max_softmax"]))]
        for index, pred in obs_images.iterrows():
            test_metadata.at[index, "observation_max"] = test_metadata["preds"][
                max_index
            ]

    test_metadata_obs = test_metadata.drop_duplicates(subset=["ObservationId"])
    max_logits_accuracy = accuracy_score(
        test_metadata_obs["class_id"],
        test_metadata_obs["observation_max"].astype("int32"),
    )

    return np.round(max_logits_accuracy * 100, 2)


def mean_softmax_performance(test_metadata):

    test_metadata["observation_mean"] = None

    ObservationIds = test_metadata.ObservationId.unique()

    for obs_id in ObservationIds:
        obs_images = test_metadata[test_metadata["ObservationId"] == obs_id]

        max_index = np.argmax(sum(obs_images["softmax"]))
        for index, pred in obs_images.iterrows():
            test_metadata.at[index, "observation_mean"] = max_index

    test_metadata_obs = test_metadata.drop_duplicates(subset=["ObservationId"])
    mean_logits_accuracy = accuracy_score(
        test_metadata_obs["class_id"],
        test_metadata_obs["observation_mean"].astype("int32"),
    )

    return np.round(mean_logits_accuracy * 100, 2)


def observation_performance(test_metadata):

    mean_logits_accuracy = mean_logits_performance(test_metadata)

    accuracy = np.round(
        accuracy_score(test_metadata["class_id"], test_metadata["preds"]) * 100, 2
    )

    return {"acc": accuracy, "mean_logits_acc": mean_logits_accuracy}


def observation_performance_all(test_metadata):

    max_logits_accuracy = max_logits_performance(test_metadata)
    mean_logits_accuracy = mean_logits_performance(test_metadata)

    max_softmax_accuracy = max_softmax_performance(test_metadata)
    mean_softmax_accuracy = mean_softmax_performance(test_metadata)

    accuracy = np.round(
        accuracy_score(test_metadata["class_id"], test_metadata["preds"]) * 100, 2
    )

    return {
        "acc": accuracy,
        "max_logits_acc": max_logits_accuracy,
        "mean_logits_acc": mean_logits_accuracy,
        "max_softmax_acc": max_softmax_accuracy,
        "mean_softmax_acc": mean_softmax_accuracy,
    }


def test_loop(
    test_metadata, test_loader, model, device, batch_size, disable_tqdm=False
):

    preds = np.zeros((len(test_metadata)))
    preds_raw = []

    for i, (images, _, _) in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), disable=disable_tqdm
    ):

        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        preds[i * batch_size : min((i + 1) * batch_size, len(test_metadata))] = (
            y_preds.argmax(1).to("cpu").numpy()
        )
        preds_raw.extend(y_preds.to("cpu").numpy())

    test_metadata["logits"] = preds_raw
    test_metadata["preds"] = preds
    test_metadata["softmax"] = [softmax(row) for row in test_metadata["logits"]]

    return test_metadata


def test_loop_performance(
    test_metadata, test_loader, model, device, batch_size, disable_tqdm=False
):

    preds = np.zeros((len(test_metadata)))
    preds_raw = []

    for i, (images, _, _) in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), disable=disable_tqdm
    ):

        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        preds[i * batch_size : min((i + 1) * batch_size, len(test_metadata))] = (
            y_preds.argmax(1).to("cpu").numpy()
        )
        preds_raw.extend(y_preds.to("cpu").numpy())

    test_metadata["logits"] = preds_raw
    test_metadata["preds"] = preds
    test_metadata["softmax"] = [softmax(row) for row in test_metadata["logits"]]

    performance = observation_performance(test_metadata)

    return performance


def test_loop_full_performance(
    test_metadata, test_loader, model, device, batch_size, disable_tqdm=False
):

    preds = np.zeros((len(test_metadata)))
    preds_raw = []

    for i, (images, _, _) in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), disable=disable_tqdm
    ):

        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        preds[i * batch_size : min((i + 1) * batch_size, len(test_metadata))] = (
            y_preds.argmax(1).to("cpu").numpy()
        )
        preds_raw.extend(y_preds.to("cpu").numpy())

    test_metadata["logits"] = preds_raw
    test_metadata["preds"] = preds
    test_metadata["softmax"] = [softmax(row) for row in test_metadata["logits"]]

    performance = observation_performance_all(test_metadata)

    return performance


def test_loop_insights(
    test_metadata, test_loader, model, device, batch_size, disable_tqdm=False
):

    preds = np.zeros((len(test_metadata)))
    preds_raw = []

    for i, (images, _, _) in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), disable=disable_tqdm
    ):

        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        preds[i * batch_size : min((i + 1) * batch_size, len(test_metadata))] = (
            y_preds.argmax(1).to("cpu").numpy()
        )
        preds_raw.extend(y_preds.to("cpu").numpy())

    test_metadata["logits"] = preds_raw
    test_metadata["preds"] = preds
    test_metadata["softmax"] = [softmax(row) for row in test_metadata["logits"]]

    performance = observation_performance(test_metadata)

    return performance, preds, preds_raw


def test_loop_priors(
    test_metadata,
    test_loader,
    model,
    device,
    class_priors,
    batch_size,
    disable_tqdm=False,
):

    preds = np.zeros((len(test_metadata)))
    preds_raw = []

    for i, (images, _, _) in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), disable=disable_tqdm
    ):

        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        preds[i * batch_size : min((i + 1) * batch_size, len(test_metadata))] = (
            y_preds.argmax(1).to("cpu").numpy()
        )
        preds_raw.extend(y_preds.to("cpu").numpy())

    test_metadata["logits"] = preds_raw
    test_metadata["preds"] = preds
    test_metadata["softmax"] = [
        softmax(row) / class_priors for row in test_metadata["logits"]
    ]

    performance = observation_performance_all(test_metadata)

    return performance, preds, preds_raw
