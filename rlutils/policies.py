# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:33:17 2017

@author: thomas
"""
import numpy as np
### policies
def egreedy(Qs, epsilon=0.05):
    ''' e-greedy policy on Q values '''
    if Qs.ndim == 1:
        a = np.argmax(Qs)
        if np.random.rand() < epsilon:
            a = np.random.randint(np.size(Qs))
        return a
    else:
        raise ValueError('Qs.ndim should be 1')
    
def softmax(Qs, temp = 1):
    ''' Boltzmann policy on Q values '''
    if Qs.ndim == 1:
        x = Qs * temp
        e_x = np.exp(x - np.max(x))
        probs = e_x / e_x.sum()
        return np.where(np.random.multinomial(1,probs))[0]
    else:
        raise ValueError('Qs.ndim should be 1')
