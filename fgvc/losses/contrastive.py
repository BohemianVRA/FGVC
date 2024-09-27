from typing import Tuple

import torch
import torch.nn as nn


def sigmoid(tensor: torch.Tensor, temperature: float = 1.0):
    """Implementation of a sigmoid function."""
    exponent = -tensor / temperature
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class RecallatKSurrogate(nn.Module):
    """An implementation of Recall@k Surrogate Loss.
    Based on https://github.com/yash0307/RecallatK_surrogate.

    The loss was proposed in a paper for CVPR2022.
    For more detail, please refer https://arxiv.org/abs/2108.11179.
    Default values are taken from the paper.

    Parameters
    ----------
    sigmoid_temperature
        Scaling factor of sigmoid function, according to the paper
    batch_size
        Number of samples in a single batch
    samples_per_class
    k_values
        Sample sizes to calculate recall from
    k_temperatures
        Temperatures for training recall@k vals
    """

    def __init__(
        self,
        batch_size: int,
        samples_per_class: int = 4,
        sigmoid_temperature: float = 1.0,
        k_values: Tuple[int] = (1, 2, 4, 8),
        k_temperatures: Tuple[int] = (1, 2, 4, 8),
    ):
        super(RecallatKSurrogate, self).__init__()
        self.sigmoid_temperature = sigmoid_temperature
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_id = int(batch_size / samples_per_class)

        self.k_values = [min(batch_size, k) for k in k_values]
        self.k_temperatures = k_temperatures

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> float:
        """Evaluate RecallatK loss."""
        assert (
            self.batch_size == logits.shape[0]
        ), f"Input must have batch size of {self.batch_size}"
        # Test batch ordering
        _targs = targs.view(-1, self.samples_per_class).float()
        _targs = (_targs - _targs.mean(dim=1, keepdim=True).int()).sum()
        assert (
            _targs == 0
        ), "Batch has a bad structure. Must have 'samples_per_class' consecutive samples"

        batch_size = logits.shape[0]
        num_id = self.num_id
        k_temperatures = self.k_temperatures
        samples_per_class = self.samples_per_class
        normalization_values = torch.Tensor(
            [min(k, (samples_per_class - 1)) for k in self.k_values]
        ).to(logits.device)

        loss = 0.0

        for query_id in range(0, logits.shape[0]):
            group_num = int(query_id / samples_per_class)

            similarity_all = (logits[query_id] * logits).sum(1)
            similarity_all_grouped = similarity_all.view(num_id, samples_per_class)
            similarity_diff_all = similarity_all.unsqueeze(-1) - similarity_all_grouped[
                group_num, :
            ].unsqueeze(0).repeat(batch_size, 1)
            similarity_sigmoid = sigmoid(similarity_diff_all, temperature=self.sigmoid_temperature)
            for i in range(samples_per_class):
                similarity_sigmoid[group_num * samples_per_class + i, i] = 0.0
            sim_all_rk = (1.0 + torch.sum(similarity_sigmoid, dim=0)).unsqueeze(dim=0)

            sim_all_rk[:, query_id % samples_per_class] = 0.0
            sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1, 1, len(self.k_values))
            _k_values = torch.Tensor(self.k_values).to(logits.device)
            _k_values = _k_values.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, samples_per_class, 1)
            sim_all_rk = _k_values - sim_all_rk
            for given_k in range(0, len(self.k_values)):
                sim_all_rk[:, :, given_k] = sigmoid(
                    sim_all_rk[:, :, given_k], temperature=float(k_temperatures[given_k])
                )

            sim_all_rk[:, query_id % samples_per_class, :] = 0.0
            k_vals_loss = torch.Tensor(self.k_values).to(logits.device)
            k_vals_loss = k_vals_loss.unsqueeze(dim=0)
            recall = torch.sum(sim_all_rk, dim=1)
            recall = torch.minimum(recall, k_vals_loss)
            recall = torch.sum(recall, dim=0)
            recall = torch.div(recall, normalization_values)
            recall = torch.sum(recall) / len(self.k_values)
            loss += (1.0 - recall) / batch_size

        return loss
