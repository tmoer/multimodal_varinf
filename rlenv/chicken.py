# -*- coding: utf-8 -*-
"""
Wet-Chicken benchmark
@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

class chicken_env(object):
    ''' Wet Chicken Benchmark '''
    
    def __init__(self,to_plot = True):
        self.state = np.array([0,0])        
        self.observation_shape = np.shape(self.get_state())[0]
        
        if to_plot:
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(111,aspect='equal')
            #ax1.axis('off')
            plt.xlim([-0.5,5.5])
            plt.ylim([-0.5,5.5])

            self.g1 = ax1.add_artist(plt.Circle((self.state[0],self.state[1]),0.1,color='red'))
            self.fig = fig
            self.ax1 = ax1
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def reset(self):
        self.state = np.array([0,0])
        return self.get_state()

    def get_state(self):
        return self.state/5

    def set_state(self,state):
        self.state = state
    
    def step(self,a):
        x = self.state[0]
        y = self.state[1]
        ax = a[0]
        ay = a[1]
        tau = np.random.uniform(-1,1)
        w=5.0
        l=5.0
        
        v = 3 * x * (1/w)
        s = 3.5 - v
        yhat = y + ay - 1 + v + s*tau
        
        # change x
        if x + ax < 0:
            x = 0
        elif yhat > l:
            x = 0
        elif x + ax > w:
            x = w
        else:
            x = x + ax
        
        # change y
        if yhat < 0:
            y = 0
        elif yhat > l:
            y = 0
        else:
            y = yhat
            
        self.state = np.array([x,y]).flatten()
        
        r = - (l - y)
        
        return self.state,r,yhat>l
            
    def plot(self):
        self.g1.remove()         
        self.g1 = self.ax1.add_artist(plt.Circle((self.state[0],self.state[1]),0.1,color='red'))
        self.fig.canvas.draw()    

# Test
if __name__ == '__main__':
    Env = chicken_env(True)
    s = Env.get_state()
    for i in range(500): 
        a = np.random.uniform(-1,1,2)
        s,r,dead = Env.step(a)
        if not dead:
            Env.plot()
        else:
            print('Died in step',i,', restarting')
            s = Env.reset() 
    print(Env.get_state())
    print('Finished')