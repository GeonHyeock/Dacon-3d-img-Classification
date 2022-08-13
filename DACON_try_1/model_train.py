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
from data_processing import GH_Dataset

path = "/home/inhamath/dacon"

with open(path + "/GH_DACON_2022_08/train_loader.pickle", 'rb') as f:
    train_loader = pickle.load(f)

with open(path + "/GH_DACON_2022_08/test_loader.pickle", 'rb') as f:
    test_loader = pickle.load(f)


from model_GH import GH
model = GH()
device = 'cuda' if cuda.is_available() else 'cpu'
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.squeeze(target))
        loss.backward()
        optimizer.step()
        if batch_idx % 100000 == 0:
            print('==================\nTrain Epoch : {} | Loss : {:.6f}'.format(epoch, loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    best = torch.tensor(0)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    if correct > best:
        best = correct
        torch.save(model.state_dict(), path + "/GH_DACON_2022_08/model_state/best.pt" )
    print(f'Test set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}'
          f'({100. * correct / len(test_loader.dataset):.0f}%)') 

since = time.time()
for epoch in range(1,20):
    epoch_start = time.time()
    train(epoch)
    test()
    m, s = divmod(time.time() - epoch_start, 60)
    print(f'Training time: {m:.0f}m {s:.0f}s')
    
m, s = divmod(time.time() - since, 60)
print(f'Total time : {m:.0f}m {s: .0f}s \nModel was trained on {device}!')
