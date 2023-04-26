import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''

BEGIN DATASET DEFINITION

'''
class AnimationDataset(Dataset):
    '''
    Class for loading the spectrogram data.
    '''
    def __init__(self, animations_file, transform=None):
        self.animations = np.load(animations_file)
        self.transform = transform
        

    def __len__(self):
        return len(self.animations)

    def __getitem__(self, idx):
        data = self.animations[idx]
        if self.transform:
            data = self.transform(data)

        return [x.to(torch.float32) for x in data]


'''

BEGIN MODEL DEFINITION

'''

class SimCLR(nn.Module):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs




        assert self.temperature > 0.0, 'The temperature must be a positive float!'
        self.net = nn.Sequential(
        
            nn.Flatten(),
            nn.Linear(150, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 150),
            nn.ReLU(),
            nn.BatchNorm1d(150),

            nn.Linear(150, 150),
            nn.ReLU(),

            nn.Linear(150, hidden_dim)
    )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.max_epochs,
                                                            eta_min=self.lr/50)
        return optimizer, lr_scheduler

    def info_nce_loss(self, batch, mode='train'):
        imgs = torch.cat(batch, dim=0)

        # Encode all images
        feats = self.net(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        # self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics
        top1 = (sim_argsort == 0).float().mean()
        top5 = (sim_argsort < 5).float().mean()
        # self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll, top1, top5
    
    def forward(self, x):
        return self.net(x)




'''
begin contrastive learning
'''

def add_noise_and_smooth(x):
    x = x + .1* np.random.randn(*x.shape)
    for i in range(3):
        x[:,i] = np.convolve(x[:,i], np.ones((5)), 'same') / 5
    return x

def add_constant(x):
   return x + np.ones(x.shape) * .05 * np.random.randn()

def scale(x):
   return x + x *.05 * np.random.randn()

def shift_period(x):
   new_start = np.random.randint(1, 10)
   return np.roll(x, new_start, axis=0)
   
def do_nothing(x):
   return x


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

contrast_transforms = transforms.Compose([
                                          transforms.RandomChoice([add_noise_and_smooth, shift_period, add_constant, scale, do_nothing]),
                                          transforms.RandomChoice([add_noise_and_smooth, shift_period, add_constant, scale, do_nothing]),
                                          transforms.ToTensor(),
                                         ])

animation_data = AnimationDataset(animations_file='../data/behaviors.npy', transform=ContrastiveTransformations(contrast_transforms, n_views=2))
data_loader = DataLoader(animation_data, batch_size=256, num_workers=2)





def train_contrastive_model(trial):

    dims = trial.suggest_int('dims', 4, 8)
    lr = trial.suggest_float('lr',1e-4, 1e-1, log=True)
    temp = trial.suggest_float('temp', .1, 10)
    weight_decay = trial.suggest_float('decay', .1, .9)
    epochs = trial.suggest_int('epochs', 10, 100)

    contrastive_network = SimCLR(dims, lr, temp, weight_decay)
    opt, sched = contrastive_network.configure_optimizers()

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f'Beginning epoch {epoch}')

        for batch in data_loader:
            opt.zero_grad()
            loss, top1, top5 = contrastive_network.info_nce_loss(batch)
            loss.backward()
            opt.step()
        # print(loss, top1, top5)
        sched.step()

    total_loss = 0
    for batch in data_loader:
        loss, top1, top5 = contrastive_network.info_nce_loss(batch)
        total_loss += loss

    return total_loss
    

import optuna
# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(train_contrastive_model, n_trials=10)