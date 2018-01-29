# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:44:00 2018

@author: ankit
"""

import sys
sys.path.append("./models")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
#%%
# configration 
n_steps=28*28
input_dims=1
n_classes=10
cuda=False

# model config
cell_type = "RNN"
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 9
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
assert(len(hidden_structs) == len(dilations))

#learning config
batch_size=128
learning_rate =1.0e-3
training_iters = batch_size * 30000
testing_step = 5000
display_step = 100

#permutation seed 
seed = 92916
#%%


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
#%%
print('Total number of training batches: ' ,len(train_loader))
print('Total number of test batches: ' , len(test_loader))
#%%
if 'seed' in globals():
    rng_permute = np.random.RandomState(seed)
    idx_permute = rng_permute.permutation(n_steps)
else:
    idx_permute = np.random.permutation(n_steps)

#%%
print("Building a dRNN with %s cells" %cell_type)
#%%
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
#%%
loss = F.nll_loss(output,target)