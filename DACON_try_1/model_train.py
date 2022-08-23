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

with open(path + "/GH_DACON_2022_08/train_dataset.pickle", 'rb') as f:
    train_dataset = pickle.load(f)

with open(path + "/GH_DACON_2022_08/test_dataset.pickle", 'rb') as f:
    test_dataset = pickle.load(f)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
    

from model_GH import GH
model = GH()
device = 'cuda' if cuda.is_available() else 'cpu'
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
best = torch.tensor(0)

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

def test(best):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print(correct,best)
    if correct > best:
        best = correct
        print(f"best : {best}")
        torch.save(model.state_dict(), path + "/GH_DACON_2022_08/model_state/best.pt" )
    print(f'Test set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}'
          f'({100. * correct / len(test_loader.dataset):.0f}%)') 
    return best

since = time.time()
for epoch in range(1,20):
    epoch_start = time.time()
    train(epoch)
    best = test(best)
    m, s = divmod(time.time() - epoch_start, 60)
    print(f'Training time: {m:.0f}m {s:.0f}s')
    
m, s = divmod(time.time() - since, 60)
print(f'Total time : {m:.0f}m {s: .0f}s \nModel was trained on {device}!')
