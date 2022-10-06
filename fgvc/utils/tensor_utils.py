from typing import List, Optional, Union

import numpy as np
import torch


def to_device(*tensors: List[Union[torch.Tensor, dict]], device: torch.device) -> List[Union[torch.Tensor, dict]]:
    """Converts pytorch tensors to device.

    Parameters
    ----------
    tensors
        (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.
    device
        Device to use (CPU,CUDA,CUDA:0,...).

    Returns
    -------
    (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.
    """
    out = []
    for tensor in tensors:
        if isinstance(tensor, dict):
            tensor = {k: v.to(device) for k, v in tensor.items()}
        else:
            tensor = tensor.to(device)
        out.append(tensor)
    return out if len(out) > 1 else out[0]


def to_numpy(*tensors: List[Union[torch.Tensor, dict]]) -> List[Union[np.ndarray, dict]]:
    """Converts pytorch tensors to numpy arrays.

    Parameters
    ----------
    tensors
        (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """
    out = []
    for tensor in tensors:
        if isinstance(tensor, dict):
            tensor = {k: v.detach().cpu().numpy() for k, v in tensor.items()}
        else:
            tensor = tensor.detach().cpu().numpy()
        out.append(tensor)
    return out if len(out) > 1 else out[0]


def concat_arrays(*lists: List[List[Union[np.ndarray, dict]]]) -> List[Optional[List[Union[np.ndarray, dict]]]]:
    """Concatenates lists of numpy arrays with predictions and targets to numpy arrays.

    Parameters
    ----------
    lists
        (One or multiple items) List of numpy arrays or dictionary of lists.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """
    out = []
    for array_list in lists:
        concatenated = None
        if len(array_list) > 0:
            if isinstance(array_list[0], dict):
                # concatenate list of dicts of numpy arrays to a dict of numpy arrays
                concatenated = {}
                for k in array_list[0].keys():
                    concatenated[k] = np.concatenate([x[k] for x in array_list])
            else:
                # concatenate list of numpy arrays to a numpy array
                concatenated = np.concatenate(array_list, axis=0)
        out.append(concatenated)
    return out if len(out) > 1 else out[0]
