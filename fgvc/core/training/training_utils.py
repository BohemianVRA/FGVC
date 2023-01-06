from typing import Iterable, List, Optional, Union

import numpy as np
import torch


def to_device(
    *tensors: List[Union[torch.Tensor, dict]], device: torch.device
) -> List[Union[torch.Tensor, dict]]:
    """Convert pytorch tensors to device.

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
    """Convert pytorch tensors to numpy arrays.

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


def concat_arrays(
    *lists: List[List[Union[np.ndarray, dict]]]
) -> List[Optional[List[Union[np.ndarray, dict]]]]:
    """Concatenate lists of numpy arrays with predictions and targets to numpy arrays.

    Parameters
    ----------
    lists
        (One or multiple items) List of numpy arrays or dictionary of lists.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """

    def _safer_concat(array_list):
        num_elems = sum([len(x) for x in array_list])
        out_array = np.zeros((num_elems, *array_list[0].shape[1:]), dtype=array_list[0].dtype)
        np.concatenate(array_list, axis=0, out=out_array)
        return out_array

    out = []
    for array_list in lists:
        concatenated = None
        if len(array_list) > 0:
            if isinstance(array_list[0], dict):
                # concatenate list of dicts of numpy arrays to a dict of numpy arrays
                concatenated = {}
                for k in array_list[0].keys():
                    concatenated[k] = _safer_concat([x[k] for x in array_list])
            else:
                # concatenate list of numpy arrays to a numpy array
                concatenated = _safer_concat(array_list)
        out.append(concatenated)
    return out if len(out) > 1 else out[0]


def get_gradient_norm(
    model_params: Union[torch.Tensor, Iterable[torch.Tensor]], norm_type: float = 2.0
) -> float:
    """Compute norm of model parameter gradients.

    Parameters
    ----------
    model_params
        Model parameters.
    norm_type
        The order of norm.

    Returns
    -------
    Norm of model parameter gradients.
    """
    grads = [p.grad for p in model_params if p.grad is not None]
    if len(grads) == 0:
        total_norm = 0.0
    else:
        norms = torch.stack([torch.norm(g.detach(), norm_type) for g in grads])
        total_norm = torch.norm(norms, norm_type).item()
    return total_norm
