from typing import Tuple

import torch
import torch.nn as nn
from collections import defaultdict


def sigmoid(tensor: torch.Tensor, temperature: float = 1.0):
    """Implementation of a sigmoid function."""
    exponent = -tensor / temperature
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class RecallatKSurrogate(nn.Module):
    """Implementation of Recall@k Surrogate Loss.

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
        self.cls_per_batch_count = int(batch_size / samples_per_class)

        self.k_values = [min(batch_size, k) for k in k_values]
        self.k_temperatures = k_temperatures

    def forward_orig(self, logits: torch.Tensor, targs: torch.Tensor) -> float:
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

        cls_to_logits = defaultdict(list)
        for cls_id in targs.unique():
            cls_to_logits[cls_id] = logits[targs == cls_id]

        normalization_values = torch.Tensor(
            [min(k, (self.samples_per_class - 1)) for k in self.k_values]
        )
        normalization_values.to(logits.device)

        loss = 0.0
        for query_id in range(0, logits.shape[0]):
            group_num = int(query_id / self.samples_per_class)

            similarity_all = (logits[query_id] * logits).sum(1)
            similarity_all_grouped = similarity_all.view(self.cls_per_batch_count, self.samples_per_class)
            similarity_diff_all = similarity_all.unsqueeze(-1) - similarity_all_grouped[
                group_num, :
            ].unsqueeze(0).repeat(self.batch_size, 1)
            similarity_sigmoid = sigmoid(similarity_diff_all, temperature=self.sigmoid_temperature)
            for i in range(self.samples_per_class):
                similarity_sigmoid[group_num * self.samples_per_class + i, i] = 0.0

            sim_all_rk = (1.0 + torch.sum(similarity_sigmoid, dim=0)).unsqueeze(dim=0)

            sim_all_rk[:, query_id % self.samples_per_class] = 0.0
            sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1, 1, len(self.k_values))

            _k_values = torch.Tensor(self.k_values).to(logits.device)
            _k_values = _k_values.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, self.samples_per_class, 1)

            sim_all_rk = _k_values - sim_all_rk
            for given_k in range(0, len(self.k_values)):
                sim_all_rk[:, :, given_k] = sigmoid(
                    sim_all_rk[:, :, given_k], temperature=float(self.k_temperatures[given_k])
                )
            sim_all_rk[:, query_id % self.samples_per_class, :] = 0.0

            k_vals_loss = torch.Tensor(self.k_values).to(logits.device)
            k_vals_loss = k_vals_loss.unsqueeze(dim=0)
            recall = torch.sum(sim_all_rk, dim=1)
            recall = torch.minimum(recall, k_vals_loss)
            recall = torch.sum(recall, dim=0)
            recall = torch.div(recall, self.normalization_values)
            recall = torch.sum(recall) / len(self.k_values)
            loss += (1.0 - recall) / self.batch_size

        return loss


    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> float:
        """Evaluate RecallatK loss."""
        # assert logits.shape[0] == targs.shape[0], "Logits and targets must have the same batch size"

        cls_targs, feature_targs = targs, None
        if isinstance(targs, tuple):
            cls_targs, feature_targs = targs

        # Reorder logits based on sorted targets
        sorted_targs, indices = torch.sort(cls_targs)
        sorted_logits = logits[indices]

        # Create class-to-logits mapping
        cls_to_logits = defaultdict(list)
        for idx, cls_id in enumerate(sorted_targs):
            cls_to_logits[cls_id.item()].append(sorted_logits[idx])

        # Convert lists to tensors
        for cls_id in cls_to_logits:
            cls_to_logits[cls_id] = torch.stack(cls_to_logits[cls_id])

        # Compute loss
        cls_loss = self.feature_loss(logits, cls_targs, cls_to_logits)
        if feature_targs is None:
            return cls_loss

        sorted_targs, indices = torch.sort(feature_targs)
        sorted_logits = logits[indices]
        feature_to_logits = defaultdict(list)
        for idx, feature_id in enumerate(sorted_targs):
            feature_to_logits[feature_id.item()].append(sorted_logits[idx])
        for feature_id in feature_to_logits:
            feature_to_logits[feature_id] = torch.stack(feature_to_logits[feature_id])
        feature_loss = self.feature_loss(logits, feature_targs, feature_to_logits)

        return cls_loss / 2 + feature_loss / 2



    def feature_loss(self, logits: torch.Tensor, targs, feature_to_logits) -> float:
        loss = 0.0
        for feature_id, feature_logits in feature_to_logits.items():
            normalization_values = torch.Tensor(
                [min(k, (len(feature_to_logits[feature_id]) - 1)) for k in self.k_values]
            ).to(logits.device)
            same_group_mask = targs == feature_id

            num_samples = feature_logits.shape[0]
            for query_idx in range(num_samples):
                query_logit = feature_logits[query_idx]
                similarity_all = (query_logit * logits).sum(1)
                similarity_in_cls = similarity_all[same_group_mask].unsqueeze(0)
                similarity_diff = similarity_all.unsqueeze(-1) - similarity_in_cls.repeat(logits.shape[0], 1)
                similarity_sigmoid = sigmoid(similarity_diff, temperature=self.sigmoid_temperature)

                # Zero Out Self-Similarities
                diag_mask = torch.eye(num_samples, device=logits.device, dtype=torch.bool)
                similarity_sigmoid[same_group_mask, :] *= ~diag_mask

                sim_all_rk = (1.0 + torch.sum(similarity_sigmoid, dim=0)).unsqueeze(dim=0)

                # Zero Out Query's Own Rank Contribution
                sim_all_rk[:, query_idx] = 0.0
                sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1, 1, len(self.k_values))

                _k_values = torch.Tensor(self.k_values).to(logits.device)
                _k_values = _k_values.unsqueeze(0).repeat(num_samples, 1)

                sim_all_rk = _k_values - sim_all_rk
                for i, k_value in enumerate(self.k_values):
                    sim_all_rk[:, :, i] = sigmoid(
                        sim_all_rk[:, :, i], temperature=float(self.k_temperatures[i])
                    )

                if sim_all_rk.shape[1] == 1:  # Leads to nan recall TODO
                    continue
                sim_all_rk[:, query_idx, :] = 0.0

                k_vals_loss = torch.Tensor(self.k_values).to(logits.device).unsqueeze(dim=0)
                recall = torch.sum(sim_all_rk, dim=1)
                recall = torch.minimum(recall, k_vals_loss)
                recall = torch.sum(recall, dim=0)
                recall = torch.div(recall, normalization_values)
                recall = torch.sum(recall) / len(self.k_values)
                loss += (1.0 - recall) / logits.shape[0]
        return loss
