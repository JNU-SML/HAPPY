from utils import info_all_data
import numpy as np
import glob
import h5py
from torch.utils.data import DataLoader
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class h5_Dataset():
    def __init__(self, data_path):
        self.path = data_path
        with h5py.File(f'{data_path}','r') as h5f:
            data = h5f['train_data'][:]
            self.NCHARS = data.shape[2]
            self.MAX_LEN = data.shape[1]
    def load(self):
        trainset = Dataset(self.path,'train')
        testset  = Dataset(self.path,'test')
        return trainset, testset, self.MAX_LEN, self.NCHARS

class Dataset(torch.utils.data.Dataset):
    def __init__(self, location,data_name):
        with h5py.File(location,'r') as h5f:
            N      = h5f[f'{data_name}_data'][:].shape[0]
            D_size = h5f[f'{data_name}_data'][:].shape[1:]
            M      = 1
            self.x_data  = np.empty(shape=(N*M,D_size[0],D_size[1]),dtype=np.uint8)
            self.y_data = np.empty(shape=(N*M,1),dtype=np.float32)
            for i in range(M):
                self.x_data[N*i:N*(i+1)]  = h5f[f'{data_name}_data'][:]
                self.y_data[N*i:N*(i+1)]  = h5f[f'{data_name}_ans'][:]
            h5f.close()
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


train, test, MAX_LEN, NCHARS = h5_Dataset('../data/SMILES_dk_split_400.h5').load()
bs = 256


train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=bs, num_workers=8,shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=bs, shuffle=False,pin_memory=True)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(CVAE, self).__init__()

        self.conv_1 = nn.Conv1d(111, 9, kernel_size=5)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=5)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=7)
        self.linear_0 = nn.Linear(160, 256)
        self.BN1d_0   = nn.BatchNorm1d(256)
        self.linear_1 = nn.Linear(256, z_dim)
        self.linear_2 = nn.Linear(256, z_dim)

        self.linear_3 = nn.Linear(z_dim, z_dim)
        self.gru = nn.GRU(z_dim+c_dim, 384, 3, batch_first=True)
        self.linear_4 = nn.Linear(384, 30)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        x = self.BN1d_0(x)

        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):

        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 111, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y
        
    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

# build model
z_dim = 16
cvae = VAE(x_dim=25, h_dim1=256, h_dim2=128 ,z_dim=z_dim)
if torch.cuda.is_available():
    cvae.cuda()
    
optimizer = optim.Adam(cvae.parameters(),1e-3)
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var,cond):

    xent_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return xent_loss , kl_loss

import pandas as pd
dicts = pd.read_csv('../data/S0_dictionary',index_col=0)
def char(data,recon,dicts,dataBool=True):
    R = []
    for N in range(len(recon)):
        D1s = []
        D2s = []
        for i in range(len(recon[0])):
    
            D2 = int(torch.argmax(recon[N][i]))
            if dataBool : D1 = int(torch.argmax(data[N][i]))
    
            if dataBool : D1s.append(str(dicts.columns[np.where((dicts.loc[D1,:]==1)==True)[0]].values[0]))
            D2s.append(str(dicts.columns[np.where((dicts.loc[D2,:]==1)==True)[0]].values[0]))
        D2s[:] = (value for value in D2s if value != 'empty')
        if dataBool : D1s[:] = (value for value in D1s if value != 'empty')
        if dataBool : print("ANS","".join(D1s))
        print("GEN","".join(D2s))
        print('\n')
        R.append("".join(D2s))
    return R
import time

# train
def train(epoch,pbar):
    cvae.train()
    train_loss = 0
    for batch_idx, (data, cond) in enumerate(train_loader):
        data, cond = data.type(torch.FloatTensor).cuda(), cond.type(torch.FloatTensor).cuda()

        recon_batch, mu, log_var = cvae(data)
        BCE,KLD = loss_function(recon_batch, data, mu, log_var,cond)
        loss = BCE+KLD
        
        loss.backward()
        optimizer.step()
        for param in cvae.parameters():
            param.grad = None
    pbar.set_postfix({'logging' : epoch, "train_loss": loss.item() / len(data)})
    return data,recon_batch,mu,log_var,cond

def test():
    cvae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, cond in test_loader:
            data, cond = data.type(torch.FloatTensor), cond.type(torch.FloatTensor)
            data, cond = data.cuda() , cond.cuda()

            recon, mu, log_var = cvae(data)
            #  pred = predictor(data)
            # sum up batch loss
            BCE,KLD = loss_function(recon, data, mu, log_var, cond)
            test_loss += (BCE+KLD).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return data,recon

print("Train start")
 
pbar = tqdm(range(1, 4000))
for epoch in pbar:
    x,y,mu,var,cond = train(epoch,pbar)

    if epoch % 100 == 0:
        data,recon = test()
        char(data,recon,dicts)
        torch.save(cvae.state_dict(), f'./model/model_{epoch}')
        #  time.sleep(0.5)
print("----------------------------")
cvae.load_state_dict(torch.load('./model/model_2000'))

with torch.no_grad():
    mu,var = cvae.encode(x)
    z = cvae.sampling(mu,var)
    recon = cvae.decode(z)
    GEN = char('_',recon,dicts,dataBool=False)
with open("S0","w") as f:
    for i in GEN:
        f.write(i+'\n')



