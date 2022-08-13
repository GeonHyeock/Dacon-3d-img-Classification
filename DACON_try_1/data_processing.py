#!/usr/bin/env python
# coding: utf-8

# # 라이브러리

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms

path = "/home/inhamath/dacon"

# # 함수, 클래스

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

class GH_Dataset_submission(data.Dataset):
    def __init__(self,X):
        self.x_data = torch.from_numpy(X).type(dtype=torch.float32)

    def __len__(self): 
        return len(self.x_data)
    
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        return x
    
def change_data(img):
    n_split = 32
    img = np.array(img)
    M,m = np.max(img),np.min(img)
    pivot = np.linspace(m - (M-m),M + (M-m),n_split)
    img = np.searchsorted(pivot,img)
    result = np.zeros(shape = (n_split,n_split,n_split))
    for x,y,z in img:
        result[x,y,z] += 1
    return result.reshape(1,n_split,n_split,n_split)
    
def rotation(value):
    x,y,z = np.random.uniform(-0.5,0.5),np.random.uniform(-0.5,0.5),np.random.uniform(-0.5,0.5)
    rot_x = np.array([[
        [1,0,0],
        [0,np.cos(x),-np.sin(x)],
        [0,np.sin(x),np.cos(x)]
    ]])
    
    rot_y = np.array([[
        [np.cos(y),0,np.sin(y)],
        [0,1,0],
        [-np.sin(y),0,np.cos(y)]
    ]])
    
    rot_z = np.array([[
        [np.cos(z),-np.sin(z),0],
        [np.sin(z),np.cos(z),0],
        [0,0,1]
    ]])

    return value @ (rot_x @ rot_y @ rot_z)[0]


# # 데이터 생성

# In[ ]:


if __name__ == "__main__":
    train_img = h5py.File(path + '/GH_DACON_2022_08/open/train.h5','r')
    train_label = pd.read_csv(path + "/GH_DACON_2022_08/open/train.csv")
    test_img = h5py.File(path + '/GH_DACON_2022_08/open/test.h5','r')
    test_label = pd.read_csv(path + "/GH_DACON_2022_08/open/sample_submission.csv")

    train_dict = {"train_img" : [change_data(rotation(train_img[idx])) for idx in train_label.ID.astype(str)],
                  "train_label" : list(train_label.label)}
    test_dict = {"test_img" : [change_data(test_img[idx]) for idx in test_label.ID.astype(str)]}

    batch_size = 256

    dataset = GH_Dataset(np.array(train_dict["train_img"]),train_dict["train_label"])
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[40000,10000])
    submission_dataset = GH_Dataset_submission(np.array(test_dict["test_img"]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
    submission_loader = torch.utils.data.DataLoader(dataset=submission_dataset)

    import pickle
    with open(path + '/GH_DACON_2022_08/train_loader.pickle', 'wb') as f:
        pickle.dump(train_loader, f)

    with open(path + '/GH_DACON_2022_08/test_loader.pickle', 'wb') as f:
        pickle.dump(test_loader, f)

    with open(path + '/GH_DACON_2022_08/submission_loader.pickle', 'wb') as f:
        pickle.dump(submission_loader, f)
