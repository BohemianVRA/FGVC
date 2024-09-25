import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """A wapper class for `torch.nn.BCEWithLogitsLoss` that aligns shapes and dtypes of inputs."""

    def __init__(
        self, weight: torch.Tensor = None, reduction: str = "mean", pos_weight: torch.Tensor = None
    ):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            weight=weight, reduction=reduction, pos_weight=pos_weight
        )

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Evaluate Binary Cross Entropy Loss."""
        if len(logits.shape) == len(targs.shape) + 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        targs = targs.to(logits.dtype)
        assert logits.shape == targs.shape
        return self.criterion(logits, targs)


class FocalLossWithLogits(nn.Module):
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.5):
        super().__init__()
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight
        self.gamma = gamma
        self.reduction = "mean"

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Evaluate Focal Loss."""
        ce_loss = F.cross_entropy(logits, targs, reduction="none", weight=self.weight)
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()  # apply mean reduction
        return focal_loss


class SeesawLossWithLogits(nn.Module):
    """An unofficial implementation for Seesaw loss.

    The loss was proposed in the technical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.

    Parameters
    ----------
    class_counts
        The list which has number of samples for each class. Should have same length as num_classes.
    p
        Scale parameter which adjust the strength of punishment.
        Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts, p: float = 0.8):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Evaluate Seesaw Loss."""
        targs = F.one_hot(targs, self.num_labels)
        self.s = self.s.to(targs.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targs)[:, None, :] * self.s[None, :, :] * torch.exp(logits)[:, None, :]
        ).sum(axis=-1) + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (-targs * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class DiceLossWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targs: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Evaluate Dice Loss across images in the batch."""
        # flatten label and prediction tensors
        logits = logits.view(-1)
        targs = targs.view(-1)

        intersection = (logits * targs).sum()
        dice = (2.0 * intersection + smooth) / (logits.sum() + targs.sum() + smooth)

        return 1 - dice


def sigmoid(tensor: torch.Tensor, temperature: float = 1.0):
    exponent = -tensor / temperature
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class RecallatK(nn.Module):
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
            k_values: tuple[int] = (1, 2, 4, 8),
            k_temperatures: tuple[int] = (1, 2, 4, 8),
    ):
        super(RecallatK, self).__init__()
        self.sigmoid_temperature = sigmoid_temperature
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_id = int(batch_size / samples_per_class)

        self.k_values = [min(batch_size, k) for k in k_values]
        self.k_temperatures = k_temperatures

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> float:
        assert self.batch_size == logits.shape[0], f"Input must have batch size of {self.batch_size}"
        # Test batch ordering
        _targs = targs.view(-1, self.samples_per_class).float()
        _targs = (_targs - _targs.mean(dim=1, keepdim=True).int()).sum()
        assert _targs == 0, "Batch has a bad structure. Must have 'samples_per_class' consecutive samples"

        batch_size = logits.shape[0]
        num_id = self.num_id
        k_temperatures = self.k_temperatures
        samples_per_class = self.samples_per_class
        normalization_values = torch.Tensor([min(k, (samples_per_class - 1)) for k in self.k_values]).to(logits.device)

        loss = 0.

        for query_id in range(0, logits.shape[0]):
            group_num = int(query_id / samples_per_class)
            # q_id_ = group_num * samples_per_class

            similarity_all = (logits[query_id] * logits).sum(1)
            sim_all_g = similarity_all.view(num_id, int(batch_size / num_id))
            sim_diff_all = similarity_all.unsqueeze(-1) - sim_all_g[group_num, :].unsqueeze(0).repeat(batch_size, 1)
            sim_sg = sigmoid(sim_diff_all, temperature=self.sigmoid_temperature)
            for i in range(samples_per_class):
                sim_sg[group_num * samples_per_class + i, i] = 0.
            sim_all_rk = (1.0 + torch.sum(sim_sg, dim=0)).unsqueeze(dim=0)

            sim_all_rk[:, query_id % samples_per_class] = 0.
            sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1, 1, len(self.k_values))
            _k_values = torch.Tensor(self.k_values).to(logits.device)
            _k_values = _k_values.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, samples_per_class, 1)
            sim_all_rk = _k_values - sim_all_rk
            for given_k in range(0, len(self.k_values)):
                sim_all_rk[:, :, given_k] = sigmoid(sim_all_rk[:, :, given_k], temperature=float(k_temperatures[given_k]))

            sim_all_rk[:, query_id % samples_per_class, :] = 0.
            k_vals_loss = torch.Tensor(self.k_values).to(logits.device)
            k_vals_loss = k_vals_loss.unsqueeze(dim=0)
            recall = torch.sum(sim_all_rk, dim=1)
            recall = torch.minimum(recall, k_vals_loss)
            recall = torch.sum(recall, dim=0)
            recall = torch.div(recall, normalization_values)
            recall = torch.sum(recall) / len(self.k_values)
            loss += (1. - recall) / batch_size

        return loss
