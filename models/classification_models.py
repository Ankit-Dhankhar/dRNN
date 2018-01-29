from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from drnn import multi_RNN_with_dilations
#%%
def _construct_cells(input_dims,hidden_structs, cell_type):
    """
    This function constructs a list of cells.
    """
    #error checking
    if cell_type not in ["RNN" ,"LSTM" ,"GRU"]:
        raise ValueError("The cell type is not currently supported.")
        
    cells = []
    for hidden_dims in hidden_structs:
        if cell_type == "RNN":
            cell= nn.RNNCell(input_size,hidden_dims)
        elif cell_type == "LSTM":
            cell= nn.LSTMCell(input_size,hidden_dims)
        elif cell_type == "GRU":
            cell= nn.GRUCell(input_size, hidden_dims)
        cells.append(cell)

    return cells
#%%    
def _rnn_reformat (x, input_dims,n_steps):
    """This function input to the shape that standard RNN can take.

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- list of 'n_steps' tensor, each has shape (batch_size, iinput_dims).
    """
    #permute batch_size and n_steps
    x_ = x.permute(1,0,2)
    
    #reshape to (n_steps*batch_size ,input_dims)
    x_.resize_(-1,input_dims)
    
    #split to get a list of 'n_steps' tensor of shape (batch_size , input_dims)
    x_reformat = torch.split(x_, n_steps,0)
    #x_reformat = torch.chunk(x_,n_steps,0)
    return x_reformat
    
    
    
     
#%%
def drnn_classification(x,
                        hidden_structs,
                        dilations,
                        n_steps,
                        n_classes,
                        input_dims=1,
                        cell_type="RNN"):
    """
    This input function constructs a multiple dilated layer RNN for classification.
    Inputs:
        x -- a tensor of shape (batch_size, n_steps , input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- alist of the sequence
        n_steps -- the length of the sequence
        n_classes -- the number aof classes for the classification
        input_dims -- the input dimension
        cell_type -- the type of the cell, should be in ["RNN","LSTM","GRU"].
        
    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' doesnot pass any output functions.
    """
    #errror Checking 
    assert (len(hidden_structs) == len(dilations))
    
    #reshape inputs
    x_reformat == _rnn_reformat(x,input_dims,n_steps)
    
    #construct a list of cells
    cells= _construct_cells(input_dims,hidden_structs,cell_type)
    
    #define dRNN structures
    layer_outputs = multi_dRNN_with_dilations(cells,x_reformat, dilations)
    
    if dilations[0] ==1:
        #dilations starts at 1, no data dependency lost
        #define the output layer
        weights = torch.rand(hidden_structs[-1],n_classes)
        
        bias =torch.rand(n_classes)
        
    else : 
        weights =torch.rand(hidden_structs[-1] * dilations[0], n_classes)
        bias = torch.rand(n_classes)

        for idx, i in  enumerate(range(-dilation[0],0,1)):
            if idx==0:
                hidden_outputs_ =layer_outputs[i]
                else:
                    hidden_outputs_ = torch.cat((hidden_outputs,layer_outputs[i]),1)
        
        pred =hidden_outputs_.mul(weights).add(bias)
    
    return pred
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    