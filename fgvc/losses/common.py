from typing import List

import numpy as np
import torch
import torch.nn as nn


class ComposeLoss(nn.Module):
    def __init__(self, criterions: List[nn.Module], weights: List[float] = None):
        super().__init__()
        if weights is None:
            weights = np.ones(len(criterions))
        assert len(criterions) > 1
        assert len(criterions) == len(weights), "Each loss should have a corresponding weights."
        self.criterions = criterions
        self.weights = weights

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        """Compute individual losses and combine them together."""
        loss = 0
        for criterion, weight in zip(self.criterions, self.weights):
            loss += weight * criterion(logits, targs)
        return loss
