from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def _numpy_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float) -> Union[np.ndarray, float]:
    """Compute cosine similarity in numpy."""
    # a [emb_size]
    # b [emb_size] or b [num_items, emb_size]
    a = a / np.maximum(np.linalg.norm(a), eps)
    if len(b.shape) == 1:
        b = b / np.maximum(np.linalg.norm(b), eps)
        out = a @ b
    else:
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), eps)
        out = a @ b.T
    return out


def cosine_similarity(
    a: Union[torch.Tensor, np.ndarray],
    b: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-08,
) -> Union[torch.Tensor, np.ndarray, float]:
    """Compute cosine similarity between a vector and a vector or matrix.

    Parameters
    ----------
    a
        A vector represented as numpy array or pytorch tensor.
    b
        A vector or a matrix represented as numpy array or pytorch tensor.
    eps
        Epsilon small constant to prevent division by 0.

    Returns
    -------
    cos_sim
        Cosine similarity number or vector.
    """
    assert len(a.shape) == 1, "Input parameter (a) must be array."
    assert len(b.shape) in [1, 2], "Input parameter (b) must be array or 2d matrix."
    assert a.shape[-1] == b.shape[-1], f"Embedding size does not match: a={a.shape}, b={b.shape}."
    if isinstance(a, torch.Tensor):
        cos_sim = F.cosine_similarity(a, b, dim=-1)
    else:
        cos_sim = _numpy_cosine_similarity(a, b, eps)
    return cos_sim


def pairwise_cosine_similarity(
    mat: Union[torch.Tensor, np.ndarray], eps: float = 1e-08
) -> Union[torch.Tensor, np.ndarray]:
    """Compute pair-wise cosine similarity between all elements in the input matrix.

    Parameters
    ----------
    mat
        A matrix represented as numpy array or pytorch tensor.
    eps
        Epsilon small constant to prevent division by 0.

    Returns
    -------
    sim_mat
        A similarity matrix.
    """
    assert len(mat.shape) == 2, "Input parameter (mat) must be a 2d matrix."
    if isinstance(mat, torch.Tensor):
        sim_mat = torch.zeros((len(mat), len(mat)), dtype=mat.dtype, device=mat.device)
        for i in range(len(mat)):
            sim_mat[i, i:] = sim_mat[i:, i] = F.cosine_similarity(mat[i], mat[i:], dim=-1, eps=eps)
    else:
        sim_mat = np.zeros((len(mat), len(mat)), dtype=mat.dtype)
        for i in range(len(mat)):
            sim_mat[i, i:] = sim_mat[i:, i] = _numpy_cosine_similarity(mat[i], mat[i:], eps)
    return sim_mat


def _numpy_batch_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """Compute batch-wise cosine similarity in numpy."""
    # a [bs, emb_size]
    # b [bs, emb_size] or b [bs, num_items, emb_size]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), eps)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), eps)
    if len(b.shape) == 2:
        b = np.expand_dims(b, 1)  # b [bs, emb_size] -> [bs, 1, emb_size]
    return np.einsum("Bi,Bji ->Bj", a, b)


def _torch_batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute batch-wise cosine similarity in pytorch."""
    # a [bs, emb_size]
    # b [bs, emb_size] or b [bs, num_items, emb_size]
    eps = torch.tensor(eps, dtype=a.dtype, device=a.device)
    a = a / torch.max(torch.norm(a, dim=-1, keepdim=True), eps)
    b = b / torch.max(torch.norm(b, dim=-1, keepdim=True), eps)
    if len(b.shape) == 2:
        b = b.unsqueeze(1)  # b [bs, emb_size] -> [bs, 1, emb_size]
    return torch.einsum("Bi,Bji ->Bj", a, b)


def batch_cosine_similarity(
    a: Union[torch.Tensor, np.ndarray],
    b: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-08,
) -> Union[torch.Tensor, np.ndarray]:
    """Compute batch-wise cosine similarity between a vector and a vector or matrix.

    Parameters
    ----------
    a
        A batch of vectors represented as numpy array or pytorch tensor.
    b
        A batch of vectors or matrices represented as numpy array or pytorch tensor.
    eps
        Epsilon small constant to prevent division by 0.

    Returns
    -------
    cos_sim
        Cosine similarity vector or matrix.
    """
    assert len(a.shape) == 2, "Input parameter (a) must be 2d matrix."
    assert len(b.shape) in [2, 3], "Input parameter (b) must be 2d or 3d matrix."
    assert a.shape[-1] == b.shape[-1], f"Embedding size does not match: a={a.shape}, b={b.shape}."
    if isinstance(a, torch.Tensor):
        cos_sim = _torch_batch_cosine_similarity(a, b, eps)
    else:
        cos_sim = _numpy_batch_cosine_similarity(a, b, eps)
    return cos_sim
