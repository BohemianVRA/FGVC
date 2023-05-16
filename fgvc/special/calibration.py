import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

logger = logging.getLogger("fgvc")


class ModelWithTemperature(nn.Module):
    """A model wrapper that adds classifier calibration - temperature scaling method.

    Paper: https://arxiv.org/abs/1706.04599
    GitHub repository: https://github.com/gpleiss/temperature_scaling
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply forward pass on the model and scale output logits."""
        logits = self.model(x, *args, **kwargs)
        scaled_logits = logits / self.temperature
        return scaled_logits


def get_temperature(logits: np.ndarray, targs: np.ndarray, device: torch.device = None) -> float:
    """Tune the temperature parameter after training the model.

    Paper: https://arxiv.org/abs/1706.04599
    GitHub repository: https://github.com/gpleiss/temperature_scaling

    The authors tune the temperature using the validation set.
    Tuning temperature is done using Cross Entropy loss (Softmax + NLL).

    Parameters
    ----------
    logits
        Numpy array with classifier raw predictions (before softmax).
    targs
        Numpy array with ground-truth targets.
    device
        Device to use (cpu,0,1,2,...).

    Returns
    -------
    temperature
        A scalar value for scaling model logits.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.from_numpy(logits).to(device)
    targs = torch.from_numpy(targs).to(device)

    # define loss functions
    ce_criterion = nn.CrossEntropyLoss()  # loss to optimize
    ece_criterion = _ECELoss()  # loss as an interesting metric

    # calculate Cross Entropy and ECE before temperature scaling
    init_ce = ce_criterion(logits, targs).item()
    init_ece = ece_criterion(logits, targs).item()
    logger.info(f"Loss values before temperature scaling: NLL={init_ce:.3f}; ECE={init_ece:.3f}.")

    # optimize the temperature w.r.t. Cross Entropy
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def _eval():
        optimizer.zero_grad()
        loss = ce_criterion(logits / temperature, targs)
        loss.backward()
        return loss

    optimizer.step(_eval)

    # calculate Cross Entropy and ECE after temperature scaling
    optimized_ce = ce_criterion(logits / temperature, targs).item()
    optimized_ece = ece_criterion(logits / temperature, targs).item()
    temperature = temperature.item()
    logger.info(f"Optimal temperature: {temperature:.4f}")
    logger.info(
        f"Loss values after temperature scaling: NLL={optimized_ce:.3f}; ECE={optimized_ece:.3f}."
    )

    return temperature


class _ECELoss(nn.Module):
    """Calculates the Expected Calibration Error of a model.

    Thanks to https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py.

    This isn't necessary for temperature scaling, just a cool metric.

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.

    Parameters
    ----------
    num_bins
        Number of confidence interval bins.
    """

    def __init__(self, num_bins: int = 15):
        super().__init__()
        self.bin_boundaries = np.linspace(0, 1, num_bins + 1)

    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(targs)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_boundaries[:-1], self.bin_boundaries[1:]):
            # calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
