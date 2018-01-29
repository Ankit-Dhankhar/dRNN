# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:38:10 2018

@author: ankit
"""

import copy
import itertools
import numpy as np
import torch
#%%
def dRNN(cell, iinputs, rate, scope='default'):
    """
    This function  constructs a  layer of dilated RNN.
    Inputs:
        cell -- the dilation operation is implemented independent of the RNN cell.
    
        inputs -- the input for the RNN. inputs should be in the form of 
            a list of 'n_steps' tensor, Each has shape (batch_size, input_dims)
    
        rate -- the rate here refers to the  'dilations' in the original Wavenet paper.
    
        scope -- Variable scope.
    Outputs:
    outputs -- the output from the RNN
    """
    
    n_steps = len(inputs)
    if rate < 0 or rate >= n_steps:
        raise ValueError('The \'rate\' variable needs to be adjusted.') 
        print("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." %(
            scaope, n_steps,rate,inputs[0].get_shape()[1]))
            
    # make the length of the input divide 'rate' , by using zero-padding
    EVEN= (n_steps % rate)==0
    if not EVEN:
        #Create a tensor in shape(batch_size, input dims), which all elements are zero.
        #This is used fo r zero padding
    
    zero_tensor =tf.zero