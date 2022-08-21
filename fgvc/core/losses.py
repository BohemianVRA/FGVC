import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    def __init__(self, weight=None, gamma=2.5):
        super().__init__()
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight
        self.gamma = gamma
        # self.reduction = "mean"

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()  # apply mean reduction
        return focal_loss


class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the technical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class.
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of punishment.
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

    def forward(self, logits, targets):
        targets = F.one_hot(targets, self.num_labels)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]
        ).sum(axis=-1) + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (-targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class DiceLossWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
