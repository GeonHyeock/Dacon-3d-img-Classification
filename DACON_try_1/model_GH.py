#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms


class GH(nn.Module):
    def __init__(self):
        super(GH, self).__init__()
        
        self.pool = nn.MaxPool3d(2)

        self.Seq1 = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.Conv3d(1,32,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(32,32,3,padding=1),
            nn.ReLU(),
        )
        self.Seq2 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.Conv3d(32,32,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(32,32,3,padding=1),
            nn.ReLU(),
        )
        
        self.next_2to3 = nn.Conv3d(32,64,3,padding = 1)

        self.Seq3 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(64,64,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(64,64,3,padding=1),
            nn.ReLU(),
        )

        self.next_3to4 = nn.Conv3d(64,128,3,padding = 1)

        self.Seq4 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.Conv3d(128,128,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(128,128,3,padding=1),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.Dropout(),
            nn.Linear(512,128),
            nn.Dropout(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.pool(self.Seq1(x))
        x = self.next_2to3(self.pool(x + self.Seq2(x)))
        x = self.next_3to4(self.pool(x + self.Seq3(x)))
        x = self.pool(self.Seq4(x))
        
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x
        


