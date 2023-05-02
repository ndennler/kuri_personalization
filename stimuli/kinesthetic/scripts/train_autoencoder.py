import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        data = self.animations[idx] * 10
        if self.transform:
            data = self.transform(data)

        return torch.squeeze(data.to(torch.float32))


'''

BEGIN MODEL DEFINITION

'''

class VanillaEncoder(nn.Module):
  def __init__(self, nz):
    super().__init__()

    self.net = nn.Sequential(
        
        nn.Flatten(),
        nn.Linear(150, 50),
        nn.Dropout(),
        nn.Sigmoid(),
        # nn.BatchNorm1d(1024),

        # nn.Linear(1024, 50),
        # nn.ReLU(),
        # # nn.BatchNorm1d(1024),

        # nn.Linear(50, 50),
        # nn.ReLU(),
        # nn.BatchNorm1d(1024),

        nn.Linear(50, 50),
        nn.Dropout(),
        nn.LeakyReLU(),

        nn.Linear(50, 50),
        nn.Dropout(),
        nn.LeakyReLU(),

        # nn.Linear(300, 150),
        # nn.ReLU(),
        # nn.BatchNorm1d(150),

        # nn.Linear(150, 150),
        # nn.ReLU(),

        nn.Linear(50, nz)
    )
  def forward(self, x):
    return self.net(x)


class VanillaDecoder(nn.Module):
  def __init__(self, nz):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(nz, 1024),
        nn.ReLU(),
        # nn.BatchNorm1d(1024),

        nn.Linear(1024, 1024),
        nn.Dropout(p=.3),
        nn.ReLU(),
        # nn.BatchNorm1d(1024),

        # nn.Linear(1024, 1024),
        # nn.ReLU(),
        # # nn.BatchNorm1d(1024),

        # nn.Linear(1024, 1024),
        # nn.ReLU(),

        # nn.Linear(150, 150),
        # nn.ReLU(),
        # nn.BatchNorm1d(150),

        # nn.Linear(150, 300),
        # nn.ReLU(),
        # nn.BatchNorm1d(300),

        # nn.Linear(300, 150),
        # nn.ReLU(),
      
        nn.Linear(1024, 150),
        nn.Dropout(p=.9),
        nn.Sigmoid(),

        nn.Linear(150, 150),     
    )

  def forward(self, x):
    return self.net(x).reshape(-1, 50, 3)


class AutoEncoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    self.encoder = VanillaEncoder(nz)
    self.decoder = VanillaDecoder(nz)

  def forward(self, x):
    return self.decoder(self.encoder(x))

  def encode(self, x):
    return self.encoder(x)
  



'''

BEGIN TRAINING THE MODEL

'''
def train_model(trial=None):
   # To test your encoder/decoder, let's encode/decode some sample images
    # first, make a PyTorch DataLoader object to sample data batches
    batch_size = 4
    nworkers = 4        # number of wrokers used for efficient data loading
    nz = 16         # dimensionality of the learned embedding

    animation_data = AnimationDataset(animations_file='../data/behaviors.npy', transform=ToTensor())
    data_loader = DataLoader(animation_data, batch_size=batch_size, num_workers=nworkers, shuffle=True)

    epochs = 15
    learning_rate = .001

    # build AE model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
    ae_model = AutoEncoder(nz).to(device)    # transfer model to GPU if available
    ae_model = ae_model.train()   # set model in train mode (eg batchnorm params get updated)

    def init_weights(m):
      if type(m) == nn.Linear:
          torch.nn.init.xavier_uniform_(m.weight)
          m.bias.data.fill_(0.01)
      
    ae_model.apply(init_weights)

    opt = torch.optim.Adam(ae_model.parameters(), learning_rate)          # create optimizer instance
    criterion = nn.MSELoss()   # create loss layer instance


    train_it = 0
    ae_model.train()
    for ep in range(epochs):
        print(f"Run Epoch {ep}")

        for sample_seq in data_loader:
            opt.zero_grad()

            sample_seq_gpu = sample_seq.to(device)

            out = ae_model.forward(sample_seq_gpu)

            rec_loss = criterion(out, sample_seq_gpu)
            rec_loss.backward()
            opt.step()

            # print(sample_seq.shape, out.shape)

            if train_it % 50 == 0:
                print("It {}: Reconstruction Loss: {}".format(train_it, rec_loss))
            
            train_it += 1
    
    print("Done!")

    total_loss = 0
    ae_model.eval()
    for batch in data_loader:
        sample_seq_gpu = batch.to(device)
        out = ae_model.forward(sample_seq_gpu)
        rec_loss = criterion(out, sample_seq_gpu)
        total_loss += rec_loss

        # print(sample_seq_gpu.cpu().detach().numpy())

    torch.save(ae_model.state_dict(), 'ae_model.pth')
    return total_loss



import optuna
import matplotlib.pyplot as plt
# 3. Create a study object and optimize the objective function.

if __name__ == '__main__':
    

    print('TESTING MAKE DATASET...')
    data = AnimationDataset(animations_file='../data/behaviors.npy', transform=ToTensor())
    print('SUCCESS.')

    print('TESTING GET DATA...')
    for i, data in enumerate(data):
        if i % 1000 == 0:
            print(f'Shape: {data.shape}, min: {torch.min(data)}, max: {torch.max(data)}')
    print('SUCCESS.')

    print('TESTING FORWARD AND BACKWARD PASSES...')
    test_X = torch.zeros([16, 50, 3])
    print(test_X.dtype)

    enc = VanillaEncoder(32)
    print(enc(test_X).shape)

    dec = VanillaDecoder(32)
    print(dec(enc(test_X)).shape)

    ae = AutoEncoder(32)
    print(ae.forward(test_X).shape)
    print('SUCCESS.')

    train_model()


    model = AutoEncoder(16)
    model.load_state_dict(torch.load('ae_model.pth'))

    animation_data = AnimationDataset(animations_file='../data/behaviors.npy', transform=ToTensor())
    data_loader = DataLoader(animation_data, batch_size=4, num_workers=2, shuffle=True)

    model.eval()
    Xs = np.arange(0,5, .1)

    for batch in data_loader:
      out = model.forward(batch)

      traj = out.cpu().detach().numpy()[0]
      plt.clf()
      plt.plot(Xs, traj[:,0], label="pan recontruct")
      plt.plot(Xs, traj[:,1], label='tilt recon')
      plt.plot(Xs, traj[:,2], label='eyes recon')

      traj = batch.cpu().detach().numpy()[0]
      plt.plot(Xs, traj[:,0], label="pan", alpha =.1)
      plt.plot(Xs, traj[:,1], label='tilt', alpha =.1)
      plt.plot(Xs, traj[:,2], label='eyes', alpha =.1)

      plt.legend()
      plt.show()
       



    # study = optuna.create_study(direction='minimize')
    # study.optimize(train_model, n_trials=1)



