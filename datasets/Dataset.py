# -*- coding: utf-8 -*-
"""
Dataset class 
@author: thomas
"""
import numpy as np
import random

class Dataset:
    ''' Bimodal, linear data ''' 
    
    def __init__(self,datasize):
        self.datasize = datasize
        N = int(datasize/2)
        X1 = self.new_x_data(N)
        X2 = self.new_x_data(N)
        Y1 = -4 * (self.X1) + 5 + np.random.normal(0,0.1,N)                
        Y2 = 4 * self.X2 + 1.6  + np.random.normal(0,0.1,N)
        
        self.X = np.append(X1,X2)
        self.Y = np.append(Y1,Y2)
        self.order = random.sample(range(datasize),datasize)

    def new_x_data(self,M):
        return np.random.uniform(-1,1,M)
    
    def next_batch_epoch(self,M):
        ''' epoch batch '''
        while len(self.order) < M:
            self.order.extend(random.sample(range(self.datasize),self.datasize))
        ind = self.order[0:M]
        del self.order[0:M]
        return self.X[ind], self.Y[ind]

    def next_batch_random(self,M):
        ''' random batch '''
        ind = random.sample(range(self.datasize),M)
        return self.X[ind], self.Y[ind]
        

