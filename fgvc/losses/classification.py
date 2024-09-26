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
