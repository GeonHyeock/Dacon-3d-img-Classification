#!/usr/bin/env python
# coding: utf-8
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import pickle

from model_GH import GH

class GH_Dataset_submission(data.Dataset):
    def __init__(self,X):
        self.x_data = torch.from_numpy(X).type(dtype=torch.float32)

    def __len__(self): 
        return len(self.x_data)
    
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        return x

path = "/home/inhamath/dacon"

with open(path + "/GH_DACON_2022_08/submission_dataset.pickle","rb") as f:
    submission_dataset = pickle.load(f)
submission_loader = torch.utils.data.DataLoader(dataset=submission_dataset,batch_size = 50)

device = 'cuda' if cuda.is_available() else 'cpu'
model = GH()
model.to(device)
model.load_state_dict(torch.load("/home/inhamath/dacon/GH_DACON_2022_08/model_state/15.pt"))

submission = pd.read_csv(path + "/GH_DACON_2022_08/open/sample_submission.csv")

result = []
for data in submission_loader:
    pred =  model(data.to(device).permute(0,2,1))[0].data.max(1, keepdim = True)[1]
    result += list(map(lambda x : int(x),pred))

submission["label"] = result
submission.to_csv(path + "/GH_DACON_2022_08/open/sample_submission.csv", index=False)



