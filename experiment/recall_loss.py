import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from collections import defaultdict
import copy
import random
from typing import Tuple, Callable
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sigmoid(tensor, temp=1.0):
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class RecallAtKLoss(torch.nn.Module):
    def __init__(self, anneal, batch_size, num_id, k_vals, k_temperatures, mixup):
        super(RecallAtKLoss, self).__init__()
        assert(batch_size%num_id==0)
        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.k_vals = [min(batch_size, k) for k in k_vals]
        self.k_temperatures = k_temperatures
        self.mixup = mixup
        self.samples_per_class = int(batch_size/num_id)

    def forward(self, preds, q_id):
        batch_size = preds.shape[0]
        num_id = self.num_id
        anneal = self.anneal
        k_vals = self.k_vals
        k_temperatures = self.k_temperatures
        samples_per_class = int(batch_size/num_id)
        norm_vals = torch.Tensor([min(k, (samples_per_class-1)) for k in k_vals]).cuda()
        group_num = int(q_id/samples_per_class)
        q_id_ = group_num*samples_per_class

        sim_all = (preds[q_id]*preds).sum(1)
        sim_all_g = sim_all.view(num_id, int(batch_size/num_id))
        sim_diff_all = sim_all.unsqueeze(-1) - sim_all_g[group_num, :].unsqueeze(0).repeat(batch_size,1)
        sim_sg = sigmoid(sim_diff_all, temp=anneal)
        for i in range(samples_per_class): sim_sg[group_num*samples_per_class+i,i] = 0.
        sim_all_rk = (1.0 + torch.sum(sim_sg, dim=0)).unsqueeze(dim=0)

        sim_all_rk[:, q_id%samples_per_class] = 0.
        sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1,1,len(k_vals))
        k_vals = torch.Tensor(k_vals).cuda()
        k_vals = k_vals.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, samples_per_class, 1)
        sim_all_rk = k_vals - sim_all_rk
        for given_k in range(0, len(self.k_vals)):
            sim_all_rk[:,:,given_k] = sigmoid(sim_all_rk[:,:,given_k], temp=float(k_temperatures[given_k]))

        sim_all_rk[:,q_id%samples_per_class,:] = 0.
        k_vals_loss = torch.Tensor(self.k_vals).cuda()
        k_vals_loss = k_vals_loss.unsqueeze(dim=0)
        recall = torch.sum(sim_all_rk, dim=1)
        recall = torch.minimum(recall, k_vals_loss)
        recall = torch.sum(recall, dim=0)
        recall = torch.div(recall, norm_vals)
        recall = torch.sum(recall)/len(self.k_vals)
        return (1.-recall)/batch_size
    



class RecallAtKDataset(torch.utils.data.Dataset):
    """Dataset for contrastive learning training.

    Generated batches provide consecutive 'samples per class' for each class up to the batch size.
    For instance, a dataset has classes x samples: 1x8, 2x6, and 3x4, samples per class are 4,
    then first batch could look like: [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2].
    Also, `batch_size` / `samples_per_class` <= `num_classes`, or no batch will be constructed.
    To regenerate batches, reshuffle() must be explicitly called.
    The provided `dataset` is a subset of `train_df` samples.

    Parameters
    ----------
    train_df
        Dataframe with training data. Must have `class_id` and `image_path`.
    transform
        Image transform function.
    batch_size
    samples_per_class
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        transform: Callable,
        batch_size: int,
        samples_per_class: int,
        col_label: str = "identity",
        col_path: str = "path"
    ):
        class_to_images = defaultdict(list)
        for i, (class_id, image_path) in train_df[[col_label, col_path]].iterrows():
            class_to_images[class_id].append(image_path)

        self.class_to_images = class_to_images
        self.dataset = None
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        for class_id in self.class_to_images:
            self.class_to_images[class_id] = [
                (class_id, image_path) for image_path in self.class_to_images[class_id]
            ]

        self.available_classes = [*self.class_to_images.keys()]
        self.transform = transform
        self.reshuffle()

    def reshuffle(self):
        """Reshuffles data and regenerates batches / inner `dataset`."""
        class_to_images = copy.deepcopy(self.class_to_images)
        for class_id in class_to_images:
            random.shuffle(class_to_images[class_id])
        classes = copy.deepcopy(self.available_classes)
        random.shuffle(classes)
        total_batches, batch = [], []
        while True:
            for class_id in classes:
                if (len(class_to_images[class_id]) >= self.samples_per_class) and (
                    len(batch) < self.batch_size / self.samples_per_class
                ):
                    batch.append(class_to_images[class_id][: self.samples_per_class])
                    class_to_images[class_id] = class_to_images[class_id][self.samples_per_class :]

            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                assert len(total_batches) != 0, (
                    "No train data."
                    "Try reduce batch size, so batch_size / samples_per_class <= num_classes"
                )
                break

        random.shuffle(total_batches)
        self.dataset = [i for batch in total_batches for class_list in batch for i in class_list]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        batch_item = self.dataset[idx]
        class_id, file_path = batch_item

        image_pil = Image.open(file_path).convert("RGB")
        image = self.transform(image_pil)
        return image, class_id

    def __len__(self):
        return len(self.dataset)


import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import copy
import random


def pos_mixup(tensor, num_id):
    batch_size = tensor.shape[0]
    num_pos = int(batch_size/num_id)
    for i in range(0, batch_size, num_pos):
        if num_pos == 6:
            alpha = np.random.rand()
            fake_1 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+1,:]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+2,:]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+3,:]
            fake_3 = torch.unsqueeze(fake_3, 0)

            alpha = np.random.rand()
            fake_4 = alpha*tensor[i+3,:] + (1.-alpha)*tensor[i+4,:]
            fake_4 = torch.unsqueeze(fake_4, 0)

            alpha = np.random.rand()
            fake_5 = alpha*tensor[i+4,:] + (1.-alpha)*tensor[i+5,:]
            fake_5 = torch.unsqueeze(fake_5, 0)

            alpha = np.random.rand()
            fake_6 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+2,:]
            fake_6 = torch.unsqueeze(fake_6, 0)

            alpha = np.random.rand()
            fake_7 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+3,:]
            fake_7 = torch.unsqueeze(fake_7, 0)

            alpha = np.random.rand()
            fake_8 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+4,:]
            fake_8 = torch.unsqueeze(fake_8, 0)

            alpha = np.random.rand()
            fake_9 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+5,:]
            fake_9 = torch.unsqueeze(fake_9, 0)

            alpha = np.random.rand()
            fake_10 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+3,:]
            fake_10 = torch.unsqueeze(fake_10, 0)

            alpha = np.random.rand()
            fake_11 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+4,:]
            fake_11 = torch.unsqueeze(fake_11, 0)

            alpha = np.random.rand()
            fake_12 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+5,:]
            fake_12 = torch.unsqueeze(fake_12, 0)

            alpha = np.random.rand()
            fake_13 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+4,:]
            fake_13 = torch.unsqueeze(fake_13, 0)

            alpha = np.random.rand()
            fake_14 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+5,:]
            fake_14 = torch.unsqueeze(fake_14, 0)

            alpha = np.random.rand()
            fake_15 = alpha*tensor[i+3,:] + (1.-alpha)*tensor[i+5,:]
            fake_15 = torch.unsqueeze(fake_15, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3, fake_4, fake_5, fake_6, fake_7, fake_8, fake_9, fake_10, fake_11, fake_12, fake_13, fake_14, fake_15), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3, fake_4, fake_5, fake_6, fake_7, fake_8, fake_9, fake_10, fake_11, fake_12, fake_13, fake_14, fake_15), dim=0)

        if num_pos == 4:
            alpha = np.random.rand()
            fake_1 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+1,:]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+2,:]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+3,:]
            fake_3 = torch.unsqueeze(fake_3, 0)

            alpha = np.random.rand()
            fake_4 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+3,:]
            fake_4 = torch.unsqueeze(fake_4, 0)

            alpha = np.random.rand()
            fake_5 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+2,:]
            fake_5 = torch.unsqueeze(fake_5, 0)

            alpha = np.random.rand()
            fake_6 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+3,:]
            fake_6 = torch.unsqueeze(fake_6, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3, fake_4, fake_5, fake_6), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3, fake_4, fake_5, fake_6), dim=0)
        elif num_pos == 3:
            alpha = np.random.rand()
            fake_1 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+1,:]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+2,:]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+2,:]
            fake_3 = torch.unsqueeze(fake_3, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3), dim=0)
    ind = num_pos
    if num_pos == 6: num_fakes = 15
    elif num_pos == 4: num_fakes = 6
    elif num_pos == 3: num_fakes = 3
    for i in range(0, tensor_fake.shape[0], num_fakes):
        tensor = torch.cat((tensor[:ind,:], tensor_fake[i:i+num_fakes,:], tensor[ind:,:]), dim=0)
        ind += num_pos + num_fakes
    return tensor



from tqdm import tqdm
import torch.nn.functional as F
from wildlife_tools.train import BasicTrainer
import copy


class RecallAtKTrainer(BasicTrainer):

    def __init__(self, batch_size_base, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size_base = batch_size_base
        self.embedding_size = embedding_size

        
    def train_epoch(self, loader):
        model = self.model.train()
        losses = []
        self.optimizer.zero_grad()
        for images, class_labels in tqdm(loader):
            output = torch.zeros((len(images), self.embedding_size)).to(self.device)
            for j in range(0, len(images), self.batch_size_base):
                images_x = images[j:j+self.batch_size_base,:].to(self.device)
                x = F.normalize(model(images_x), p=2, dim=-1) # Normalize
                output[j:j+self.batch_size_base,:] = copy.copy(x)
                del x
                torch.cuda.empty_cache()
            if self.objective.mixup:
                output_mixup = pos_mixup(output, self.objective.num_id)
                num_samples = output_mixup.shape[0]
            else:
                num_samples = output.shape[0]

            output.retain_grad()
            loss = 0.0

            for q in range(0, num_samples):
                if self.objective.mixup:
                    loss += self.objective(output_mixup, q)
                else:
                    loss += self.objective(output, q)
            losses.append(loss.item())
            loss.backward()
            output_grad = copy.copy(output.grad)

            del loss
            del output
            if self.objective.mixup:
                del output_mixup
            torch.cuda.empty_cache()

            for j in range(0, len(images), self.batch_size_base):
                images_x = images[j:j+self.batch_size_base,:].to(self.device)
                x = F.normalize(model(images_x), p=2, dim=-1) # Normalize
                x.backward(output_grad[j:j+self.batch_size_base,:])
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()
        return {'train_loss_epoch_avg': np.mean(losses)}


    def train(self):
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=torch.utils.data.SequentialSampler(self.dataset),
            pin_memory=True,
            drop_last=True
        )
        for e in range(self.epochs):
            loader.dataset.reshuffle()
            epoch_data = self.train_epoch(loader)
            self.epoch += 1

            if self.epoch_callback:
                self.epoch_callback(trainer=self, epoch_data=epoch_data)
