import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class AnimationDataset(Dataset):
    '''
    Class for loading the animation data.
    '''
    def __init__(self, animations_file, transform=None):
        self.animations = np.load(animations_file)
        self.transform = transform
        

    def __len__(self):
        return len(self.animations)

    def __getitem__(self, idx):
        data = self.animations[idx] * 25
        if self.transform:
            data = self.transform(data)

        return torch.squeeze(data.to(torch.float32))
    

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers//2, batch_first=True, dropout=dropout, bidirectional=True)
        
    def forward(self, x):
        # x has shape (batch_size, seq_len, input_size)
        # output has shape (batch_size, seq_len, hidden_size)
        # hidden has shape (num_layers, batch_size, hidden_size)
        output, hidden = self.gru(x,)
        self.h0 = hidden
        return output, hidden
    
    def init_hidden(self, batch_size):
        self.h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)

    


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, hidden):
        # x has shape (batch_size, seq_len, input_size)
        # hidden has shape (num_layers, batch_size, hidden_size)
        output, hidden = self.gru(x, hidden)
        # output has shape (batch_size, seq_len, hidden_size)
        # hidden has shape (num_layers, batch_size, hidden_size)
        output = self.linear(output)
        # output has shape (batch_size, seq_len, input_size)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)
        
    def forward(self, x):
        # x has shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        #encoding phase
        self.encoder.init_hidden(batch_size)
        _, hidden = self.encoder(x)
        
        # decoding phase
        input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
        output_seq = []

        for _ in range(seq_len):
            output, hidden = self.decoder(input, hidden)
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return output_seq
    
    def embed_single(self,x):
        x = x.unsqueeze(0)
        # x has shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        # print(x.shape)
        
        #encoding phase
        self.encoder.init_hidden(batch_size)
        _, hidden = self.encoder(x)

        return torch.reshape(hidden, (-1,))

    

def train_model(seq2seq_model, train_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        seq2seq_model.train()
        train_loss = 0.0
        
        for i, seq in enumerate(train_loader):

            input_seq = seq.to(device)
            target_seq = seq.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output_seq = seq2seq_model(input_seq)

            # Compute loss
            loss = criterion(output_seq, target_seq)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 10 == 0:
                print(f"Train iter {i} Loss: {loss.item():.4f}")

def visualize_samples(model, dataloader, device):
    Xs = np.arange(0,5, .1)

    for batch in dataloader:
      
      batch = batch.to(device)
      output_seq = model(batch)

      traj = output_seq.cpu().detach().numpy()[0]
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


if __name__ == '__main__':
    INPUT_SIZE = 3
    NZ = 32
    LR = .008
    N_LAYERS = 6
    EPOCHS = 150
    DROPOUT = .2
    BATCH_SIZE = 84

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available

    model = Seq2Seq(INPUT_SIZE, NZ, N_LAYERS, DROPOUT)
    model.to(device)
    
    animation_data = AnimationDataset(animations_file='../data/behaviors.npy', transform=ToTensor())
    data_loader = DataLoader(animation_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), LR) 

    # train_model(model, data_loader, nn.MSELoss(), opt, EPOCHS, device)
    # torch.save(model.state_dict(), 'large_seq2seq_ae_model.pth')
    # visualize_samples(model, data_loader, device)

    model.load_state_dict(torch.load('large_seq2seq_ae_model.pth'))
    data = np.load('../data/behaviors.npy')
    outs = []
    for entry in tqdm(data):
        input = torch.tensor(entry*25).to(torch.float32).to(device)
        outs.append(model.embed_single(input).cpu().detach().numpy())

    print(np.array(outs).shape)
    np.save('../data/large_embeddings.npy', np.array(outs))
    
