# -*- coding: utf-8 -*-
"""
Plotting class
@author: thomas
"""
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    ''' Manage plotting ''' 
    
    def __init__(self):
        pass
    
    def plot_data(self,Data):

        self.fig = fig =  plt.figure()
        self.ax1 = ax1 = fig.add_subplot(311)
        self.ax2 = ax2 = fig.add_subplot(312)
        self.ax3 = ax3 = fig.add_subplot(313)
        
        ax1.scatter(Data.X,Data.Y,color='b')
        ax1.axis([-1, 1, -1, 7])
        plt.xlabel('S')
        plt.ylabel('S\'')
        plt.title('True data')
        
        new_dat, = ax2.plot([],[],'ko')
        self.new_dat = new_dat        
        ax2.axis([-1, 1, -1, 7])
        plt.xlabel('S')
        plt.ylabel('S\'')
        
        plt.subplot(313)
        new_dat_2, = ax3.plot([],[],'ro')
        self.new_dat_2 = new_dat_2
        self.t = np.zeros(300)
        self.lr = np.zeros(300)
        ax3.axis([0, 100000, 0, 0.01])
        plt.ylabel('learning rate')
        
        #plt.draw()
        #plt.show(block=False)
        
    def plot_samples(self,x,y):
        self.new_dat.set_xdata(x)
        self.new_dat.set_ydata(y)
        self.fig.canvas.draw()
        #plt.pause(0.001)
        #plt.show(block=False)  
        
    def plot_lr(self,t,lr):
        #plt.plot(t,lr)  
        self.t[:len(t)] = t
        self.lr[:len(t)] = lr
        #self.ax3.clear()
        #self.ax3.plot(t,lr)
        self.new_dat_2.set_xdata(self.t)
        self.new_dat_2.set_ydata(self.lr)
        self.fig.canvas.draw()
        #plt.pause(0.001)
        #plt.show(block=False)        

