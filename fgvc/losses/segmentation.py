import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_dice_loss(probs: torch.Tensor, targs: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Compute binary Dice Loss."""
    assert probs.shape == targs.shape
    a = (probs * targs).sum(1)
    b = (probs ** 2).sum(1) + eps
    c = (targs ** 2).sum(1) + eps
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss.

        Thanks to: https://arxiv.org/abs/1606.04797
        And: https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

        Parameters
        ----------
        logits
            Tensor with predicted values.
        targs
            Tensor with target values.

        Returns
        -------
        loss
        """
        probs = F.sigmoid(logits)
        bg_loss = binary_dice_loss(1 - probs[:, 0], 1 - targs, eps=self.eps)  # background class
        fg_loss = binary_dice_loss(probs[:, 0], targs, eps=self.eps)  # foreground class
        loss = bg_loss + fg_loss
        return loss


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss.

        Thanks to: https://arxiv.org/abs/1606.04797
        And: https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

        Parameters
        ----------
        logits
            Tensor with predicted values.
        targs
            Tensor with target values.

        Returns
        -------
        loss
        """
        probs = F.softmax(logits, 1)
        bg_loss = binary_dice_loss(probs[:, 0], 1 - targs, eps=self.eps)  # background class
        fg_loss = binary_dice_loss(probs[:, 1], targs, eps=self.eps)  # foreground class
        loss = bg_loss + fg_loss
        return loss
