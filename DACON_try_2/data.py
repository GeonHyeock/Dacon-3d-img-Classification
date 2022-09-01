#!/usr/bin/env python
# coding: utf-8

# # 라이브러리

# In[1]:


import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import plotly.graph_objects as go
from copy import deepcopy
import pickle

# # 함수, 클래스

# In[2]:


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
    
def rotation(value):
    x,y,z = np.random.uniform(-1.5,1.5),np.random.uniform(-1.5,1.5),np.random.uniform(-1.5,1.5)
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

def data_scale(data):
    data = np.array(sorted(data, key = lambda x: [x[0],x[1],x[2]]))
    idx = np.linspace(0,data.shape[0]-1,1000).astype(int)
    return data[idx]
    
def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

def pcshow(data):
    xs,ys,zs = data[::,0],data[::,1],data[::,2]
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


# # 데이터 생성

# In[3]:


path = "/home/inhamath/dacon"
train_img = h5py.File(path + '/GH_DACON_2022_08/open/train.h5','r')
train_label = pd.read_csv(path + "/GH_DACON_2022_08/open/train.csv")

train_dict = {"train_img" : [data_scale(np.unique(train_img[idx],axis=0)) for idx in train_label.ID.astype(str)],
            "train_label" : list(train_label.label)}



pivot_data = deepcopy(train_dict["train_img"])
pivot_label = deepcopy(train_dict["train_label"])

for _ in range(5):
    train_dict["train_img"] += list(map(rotation,pivot_data))
    train_dict["train_label"] += pivot_label

dataset = GH_Dataset(np.array(train_dict["train_img"]),train_dict["train_label"])
train_dataset,test_dataset = torch.utils.data.random_split(dataset,[250000,50000])


with open(path + '/GH_DACON_2022_08/train_dataset.pickle', 'wb') as f:
    pickle.dump(train_dataset, f)

with open(path + '/GH_DACON_2022_08/test_dataset.pickle', 'wb') as f:
    pickle.dump(test_dataset, f)


test_img = h5py.File(path + '/GH_DACON_2022_08/open/test.h5','r')
test_label = pd.read_csv(path + "/GH_DACON_2022_08/open/sample_submission.csv")
test_dict = {"test_img" : [data_scale(np.unique(test_img[idx],axis=0)) for idx in test_label.ID.astype(str)]}

submission_dataset = GH_Dataset_submission(np.array(test_dict["test_img"]))

with open(path + '/GH_DACON_2022_08/submission_dataset.pickle', 'wb') as f:
    pickle.dump(submission_dataset, f)