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


def loss_select(loss, opt, to_optim):
    if loss == 'recallatk':
        loss_params = {'anneal': opt.sigmoid_temperature, 'batch_size': opt.bs,
                       "samples_per_class": opt.samples_per_class, 'feat_dims': opt.embed_dim,
                       'k_vals': opt.k_vals_train, 'k_temperatures': opt.k_temperatures, 'mixup': opt.mixup}
        criterion = RecallatK(**loss_params)
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim


def sigmoid(tensor: torch.Tensor, temperature: float = 1.0):
    exponent = -tensor / temperature
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class RecallatK(torch.nn.Module):
    def __init__(
            self,
            sigmoid_temperature: float,
            batch_size: int,
            samples_per_class: int,
            num_id: int,  # batch size / samples per class - number of samples in one class drawn before choosing the next class
            # feat_dims,
            k_values: list[int],  # selection of k values
            k_temperatures: list[int],  # Temperature for training recall@k vals
            use_mixup: bool  # not used here
    ):
        super(RecallatK, self).__init__()
        assert (batch_size % num_id == 0)
        self.sigmoid_temperature = sigmoid_temperature
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.num_id = int(batch_size / samples_per_class)

        # self.feat_dims = feat_dims
        self.k_values = [min(batch_size, k) for k in k_values]
        self.k_temperatures = k_temperatures
        self.use_mixup = use_mixup

    def forward(self, preds: torch.Tensor, query_id):
        batch_size = preds.shape[0]
        num_id = self.num_id
        # anneal = self.sigmoid_temperature
        # feat_dims = self.feat_dims
        k_values = self.k_values
        k_temperatures = self.k_temperatures
        samples_per_class = self.samples_per_class

        normalization_values = torch.Tensor([min(k, (samples_per_class - 1)) for k in k_values]).cuda()
        group_num = int(query_id / samples_per_class)
        # q_id_ = group_num * samples_per_class

        sim_all = (preds[query_id] * preds).sum(1)
        sim_all_g = sim_all.view(num_id, int(batch_size / num_id))
        sim_diff_all = sim_all.unsqueeze(-1) - sim_all_g[group_num, :].unsqueeze(0).repeat(batch_size, 1)
        sim_sg = sigmoid(sim_diff_all, temperature=self.sigmoid_temperature)
        for i in range(samples_per_class):
            sim_sg[group_num * samples_per_class + i, i] = 0.
        sim_all_rk = (1.0 + torch.sum(sim_sg, dim=0)).unsqueeze(dim=0)

        sim_all_rk[:, query_id % samples_per_class] = 0.
        sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1, 1, len(k_values))
        k_values = torch.Tensor(k_values).cuda()
        k_values = k_values.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, samples_per_class, 1)
        sim_all_rk = k_values - sim_all_rk
        for given_k in range(0, len(self.k_values)):
            sim_all_rk[:, :, given_k] = sigmoid(sim_all_rk[:, :, given_k], temperature=float(k_temperatures[given_k]))

        sim_all_rk[:, query_id % samples_per_class, :] = 0.
        k_vals_loss = torch.Tensor(self.k_values).cuda()
        k_vals_loss = k_vals_loss.unsqueeze(dim=0)
        recall = torch.sum(sim_all_rk, dim=1)
        recall = torch.minimum(recall, k_vals_loss)
        recall = torch.sum(recall, dim=0)
        recall = torch.div(recall, normalization_values)
        recall = torch.sum(recall) / len(self.k_values)
        return (1. - recall) / batch_size
