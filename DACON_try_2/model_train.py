#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import pickle

import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms

path = "/home/inhamath/dacon"

class GH_Dataset(torch.utils.data.Dataset): 
    def __init__(self,X,Y):
        self.x_data = torch.from_numpy(X).type(dtype=torch.float32)
        self.y_data = torch.tensor(Y).resize_(len(X),1)

    def __len__(self): 
        return len(self.x_data)
    
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

with open(path + "/GH_DACON_2022_08/train_dataset.pickle", 'rb') as f:
    train_dataset = pickle.load(f)

with open(path + "/GH_DACON_2022_08/test_dataset.pickle", 'rb') as f:
    test_dataset = pickle.load(f)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
del train_dataset
del test_dataset

from model_GH import GH
model = GH()
device = 'cuda' if cuda.is_available() else 'cpu'
model.to(device)

from model_GH import pointnetloss
criterion = pointnetloss
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    train_loss = 0
    print('==================\nTrain epoch : {}'.format(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, m3x3, m64x64 = model(data.permute(0,2,1))
        loss = criterion(output, torch.squeeze(target), m3x3, m64x64)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train batch_idx : {} | Loss : {:.6f}'.format(batch_idx, loss.item()))
    print('Train batch_idx : {} | Loss : {:.6f}'.format(batch_idx, loss.item()))
    train_loss /= len(train_loader.dataset)
    return train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, m3x3, m64x64 = model(data.permute(0,2,1))
        test_loss += criterion(output, torch.squeeze(target), m3x3, m64x64).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    torch.save(model.state_dict(), path + f"/GH_DACON_2022_08/model_state/{epoch}.pt" )
    print(f'Test set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}'
          f'({100. * correct / len(test_loader.dataset):.0f}%)') 
    return test_loss

since = time.time()
f = open(path + "/GH_DACON_2022_08/log.txt", 'w')
for epoch in range(1,20):
    epoch_start = time.time()
    train_loss = train(epoch)
    test_loss = test(epoch)
    f.write(f"{train_loss:.8f},{test_loss:.8f}\n")
    m, s = divmod(time.time() - epoch_start, 60)
    print(f'Training time: {m:.0f}m {s:.0f}s')
f.close()
m, s = divmod(time.time() - since, 60)
print(f'Total time : {m:.0f}m {s: .0f}s \nModel was trained on {device}!')
