import tqdm
import torch

import numpy as np

from scipy import stats
from sklearn.metrics import accuracy_score


def max_logits_performance(test_metadata):
    
    test_metadata['max_logits'] = [np.max(row) for row in test_metadata['logits']]
    test_metadata['observation_max'] = None

    ObservationIds = test_metadata.ObservationId.unique()

    for obs_id in ObservationIds:
        obs_images = test_metadata[test_metadata['ObservationId'] == obs_id]
        max_index =  obs_images.index[np.argmax(np.array(obs_images['max_logits']))]
        for index, pred in obs_images.iterrows():
            test_metadata.at[index, 'observation_max'] = test_metadata['preds'][max_index]
    
    test_metadata_obs = test_metadata.drop_duplicates(subset=['ObservationId'])
    max_logits_accuracy = accuracy_score(test_metadata_obs['class_id'], test_metadata_obs['observation_max'].astype('int32'))
    
    return max_logits_accuracy
    
    
def mean_softmax_performance(test_metadata):
    
    test_metadata['observation_mean'] = None

    ObservationIds = test_metadata.ObservationId.unique()

    for obs_id in ObservationIds:
        obs_images = test_metadata[test_metadata['ObservationId'] == obs_id]

        max_index =  np.argmax(sum(obs_images['logits']))
        for index, pred in obs_images.iterrows():
            test_metadata.at[index, 'observation_mean'] = max_index
    
    test_metadata_obs = test_metadata.drop_duplicates(subset=['ObservationId'])
    mean_softmax_accuracy = accuracy_score(test_metadata_obs['class_id'], test_metadata_obs['observation_mean'].astype('int32'))
    
    return mean_softmax_accuracy
    
    
def observation_performance(test_metadata):
    
    max_logits_accuracy = max_logits_performance(test_metadata)
    mean_softmax_accuracy = mean_softmax_performance(test_metadata)
    
    return max_logits_accuracy, mean_softmax_accuracy


def test_loop(test_metadata, test_loader, model, device):
    
    preds = np.zeros((len(test_metadata)))
    preds_raw = []
    wrong_paths = []

    for i, (images, _, _) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):

        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        preds[i * len(images): (i+1) * len(images)] = y_preds.argmax(1).to('cpu').numpy()
        preds_raw.extend(y_preds.to('cpu').numpy())

    
    test_metadata['logits'] = preds_raw
    test_metadata['preds'] = preds

    
    accuracy = accuracy_score(test_metadata['class_id'], test_metadata['preds'])
    
    max_logit_obs_acc, mean_softmax_obs_acc = observation_performance(test_metadata)
    
    return accuracy, max_logit_obs_acc, mean_softmax_obs_acc