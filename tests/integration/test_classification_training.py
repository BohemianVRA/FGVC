import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from fgvc.core.optimizers import cosine, sgd
from fgvc.core.training import ClassificationTrainer


class DummyShapesDataset(Dataset):
    """Generated dataset with small images that contain 1 - 3 circles."""

    def __init__(self, num_samples: int = 250, **kwargs):
        num_classes = 3  # = number of circles in the image
        self.num_samples = num_samples
        self.num_classes = num_classes

        # generate images and class_ids
        np.random.seed(42)
        self.images = np.zeros((num_samples, 24, 24, 1)).astype(np.float32)
        self.class_ids = np.random.uniform(0, num_classes, size=num_samples).astype(np.int64)

        # add circles to the images
        for image, class_id in zip(self.images, self.class_ids):
            image = image[..., 0]

            # create circle object represented in numpy array
            r = 2 + class_id * 3
            xx, yy = np.mgrid[-r : r + 1, -r : r + 1]
            circle: np.ndarray = xx**2 + yy**2 <= r**2

            coordinates = np.random.randint(4, 20, size=(1, 2))
            for x, y in coordinates:
                # valid indices of the array
                i = slice(max(x - r, 0), min(x + r + 1, image.shape[0]))
                j = slice(max(y - r, 0), min(y + r + 1, image.shape[1]))

                # visible slice of the circle
                ci = slice(
                    abs(min(x - r, 0)), circle.shape[0] - abs(min(image.shape[0] - (x + r + 1), 0))
                )
                cj = slice(
                    abs(min(y - r, 0)), circle.shape[1] - abs(min(image.shape[1] - (y + r + 1), 0))
                )

                image[i, j] += circle[ci, cj]

        # create simple augmentation
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.transform(self.images[idx])
        class_id = self.class_ids[idx]
        return image, class_id


@pytest.fixture(scope="module")
def model() -> nn.Module:
    """Fixture that returns `timm` model."""
    return timm.create_model(
        model_name="mobilenetv3_small_050", pretrained=False, num_classes=80, in_chans=1
    )


@pytest.fixture(scope="module")
def trainloader() -> DataLoader:
    """Fixture that returns dataloader with dummy training data (random small images)."""
    trainset = DummyShapesDataset(num_samples=250, num_classes=80)
    return DataLoader(trainset, batch_size=32, num_workers=0, shuffle=True)


@pytest.fixture(scope="module")
def validloader() -> DataLoader:
    """Fixture that returns dataloader with dummy validation data (random small images)."""
    valset = DummyShapesDataset(num_samples=150, num_classes=80)
    return DataLoader(valset, batch_size=32, num_workers=0, shuffle=False)


def parse_losses(log: str) -> Tuple[int, float, float]:
    """Helper method for parsing training and validation losses from the training log."""
    split_1 = log.split(" - avg_train_loss: ")
    assert len(split_1) == 2
    epoch = int(split_1[0][6:])
    split_2 = split_1[1].split(" avg_val_loss: ")
    assert len(split_2) == 2
    train_loss = float(split_2[0].strip())
    val_loss = float(split_2[1])
    return epoch, train_loss, val_loss


def test_classification_trainer_1(
    tmp_path: Path,
    model: nn.Module,
    trainloader: DataLoader,
    validloader: DataLoader,
):
    """Test training model on dummy data using Classification Trainer for few epochs."""
    tmp_path = str(tmp_path)
    num_epochs = 3
    optimizer = sgd(model.parameters(), lr=0.01)
    scheduler = cosine(optimizer, epochs=num_epochs)
    trainer = ClassificationTrainer(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
    )
    trainer.train(
        num_epochs=num_epochs,
        seed=777,
        path=tmp_path,
        resume=False,
    )
    assert os.path.isfile(os.path.join(tmp_path, "training.log"))
    assert os.path.isfile(os.path.join(tmp_path, "best_loss.pth"))
    assert os.path.isfile(os.path.join(tmp_path, "best_f1.pth"))
    assert os.path.isfile(os.path.join(tmp_path, "best_accuracy.pth"))
    assert os.path.isfile(os.path.join(tmp_path, f"epoch_{num_epochs}.pth"))
    assert not os.path.isfile(os.path.join(tmp_path, "checkpoint.pth.tar"))
    with open(os.path.join(tmp_path, "training.log")) as f:
        training_log = f.read()

    # parse training log
    loss_logs = re.findall(
        r"Epoch [0-9]+ - avg_train_loss: [0-9.]+ +avg_val_loss: [0-9.]+", training_log
    )
    train_losses, val_losses = [], []
    for log in loss_logs:
        epoch, train_loss, val_loss = parse_losses(log)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    # test that training loss decreases
    for prev, curr in zip(train_losses[:-1], train_losses[1:]):
        assert prev > curr
    # test that validation loss decreases
    for prev, curr in zip(val_losses[:-1], val_losses[1:]):
        assert prev > curr


def test_classification_trainer_2(
    tmp_path: Path,
    model: nn.Module,
    trainloader: DataLoader,
):
    """Test training model on dummy data using Classification Trainer for few epochs.

    Test training without validation loader.
    """
    tmp_path = str(tmp_path)
    num_epochs = 3
    optimizer = sgd(model.parameters(), lr=0.01)
    scheduler = cosine(optimizer, epochs=num_epochs)
    trainer = ClassificationTrainer(
        model=model,
        trainloader=trainloader,
        validloader=None,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
    )
    trainer.train(
        num_epochs=num_epochs,
        seed=777,
        path=tmp_path,
        resume=False,
    )
    assert os.path.isfile(os.path.join(tmp_path, "training.log"))
    assert not os.path.isfile(os.path.join(tmp_path, "best_loss.pth"))
    assert not os.path.isfile(os.path.join(tmp_path, "best_f1.pth"))
    assert not os.path.isfile(os.path.join(tmp_path, "best_accuracy.pth"))
    assert os.path.isfile(os.path.join(tmp_path, f"epoch_{num_epochs}.pth"))
    assert not os.path.isfile(os.path.join(tmp_path, "checkpoint.pth.tar"))
    with open(os.path.join(tmp_path, "training.log")) as f:
        training_log = f.read()

    # parse training log
    loss_logs = re.findall(
        r"Epoch [0-9]+ - avg_train_loss: [0-9.]+ +avg_val_loss: [0-9.]+", training_log
    )
    train_losses, val_losses = [], []
    for log in loss_logs:
        epoch, train_loss, val_loss = parse_losses(log)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    # test that training loss decreases
    for prev, curr in zip(train_losses[:-1], train_losses[1:]):
        assert prev > curr
