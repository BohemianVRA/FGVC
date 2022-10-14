import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def binary_dice_loss(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Compute binary Dice Loss."""
        assert preds.shape == targs.shape
        a = (preds * targs).sum(1)
        b = (preds**2).sum(1) + self.eps
        c = (targs**2).sum(1) + self.eps
        d = (2 * a) / (b + c)
        loss = 1 - d
        return loss.mean()

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
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
        preds = F.softmax(preds, 1)
        loss = self.binary_dice_loss(preds[:, 0], 1 - targs) + self.binary_dice_loss(  # background class
            preds[:, 1], targs
        )  # foreground class
        return loss
