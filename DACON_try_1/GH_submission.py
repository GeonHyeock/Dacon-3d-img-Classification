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

from data_processing import GH_Dataset_submission
from model_GH import GH

path = "/home/inhamath/dacon"

with open(path + "/GH_DACON_2022_08/submission_loader.pickle","rb") as f:
    submission_loader = pickle.load(f)
    
device = 'cuda' if cuda.is_available() else 'cpu'
model = GH()
model.load_state_dict(torch.load("/home/inhamath/dacon/GH_DACON_2022_08/model_state/best.pt"))
model.to(device)

submission = pd.read_csv(path + "/GH_DACON_2022_08/open/sample_submission.csv")
submission["label"] = [int(model(data.to(device)).data.max(1, keepdim=True)[1]) for data in submission_loader]
submission.to_csv(path + "/GH_DACON_2022_08/open/sample_submission.csv", index=False)



