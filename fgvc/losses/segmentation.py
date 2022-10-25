import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def binary_dice_loss(self, probs: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Compute binary Dice Loss."""
        assert probs.shape == targs.shape
        a = (probs * targs).sum(1)
        b = (probs**2).sum(1) + self.eps
        c = (targs**2).sum(1) + self.eps
        d = (2 * a) / (b + c)
        loss = 1 - d
        return loss.mean()

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss.

        Thanks to: https://arxiv.org/abs/1606.04797
        And: https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

        Parameters
        ----------
        preds
            Tensor with predicted values.
        targs
            Tensor with target values.

        Returns
        -------
        loss
        """
        probs = F.softmax(logits, 1)
        bg_loss = self.binary_dice_loss(probs[:, 0], 1 - targs)  # background class
        fg_loss = self.binary_dice_loss(probs[:, 1], targs)  # foreground class
        loss = bg_loss + fg_loss
        return loss
