# -*- coding: utf-8 -*-
"""
Dataset class 
@author: thomas
"""
import numpy as np
import random

class Dataset:
    ''' Trimodal, non-linear data ''' 
    
    def __init__(self,datasize):
        self.datasize = datasize
        self.X = self.new_x_data(datasize)
        
        def data_fn(x):            
            if x < -0.3 :
                y = 2.5 + np.random.normal(0,0.1,1)   
            elif x < 0.3:
                # bimodal
                if np.random.binomial(1,0.2):
                    y = 4 * x + 4 + np.random.normal(0,0.1,1)
                else:
                    y = -4 * x + 1.5 + np.random.normal(0,0.1,1)      
            else: 
                # trimodal
                if np.random.binomial(1,0.3):
                    y = np.log(x+1)+5 + np.random.normal(0,0.1,1)
                elif np.random.binomial(1,0.5):
                    y = -1 * x + 0.2 + np.random.normal(0,0.1,1)
                else:
                    y = 5*x**2 + np.random.normal(0,0.1,1)
            return y
        
        self.Y = np.zeros(datasize)
        for i in range(datasize):
            self.Y[i] = data_fn(self.X[i])
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
        

