# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:48:24 2017

@author: thomas
"""
#from layers import Latent_Layer
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tfutils.helpers import repeat_v2
from tfutils.distributions import logsumexp, discretized_logistic
from layers import Latent_Layer

class Network(object):
    ''' VAE template ''' 
    
    def __init__(self,hps):
        # placeholders
        self.x = x = tf.placeholder("float32", shape=[None,1])
        self.y = y = tf.placeholder("float32", shape=[None,1])
        self.is_training = is_training = tf.placeholder("bool") # if True: sample from q, else sample from p
        self.k = k = tf.placeholder('int32') # number of importance samples
        self.temp = temp = tf.Variable(5.0,name='temperature',trainable=False) # Temperature for discrete latents
        self.lamb = lamb = tf.Variable(1.0,name="lambda",trainable=False) # Lambda for KL annealing

        # Importance sampling: repeats along second dimension
        x_rep = repeat_v2(x,k)
        y_rep = repeat_v2(y,k)
        
        # Encoder x,y --> h
        xy = tf.concat([x_rep,y_rep],1) # concatenate along last dim
        h_up = slim.fully_connected(xy,hps.h_size,tf.nn.relu)
        
        # Initialize ladders
        layers = []
        for i in range(hps.depth):
            layers.append(Latent_Layer(hps=hps,var_type=hps.var_type[i],depth=i,is_top=(i==(hps.depth-1))))
            
        # Ladder up
        for i,layer in enumerate(layers):
            h_up = layer.up(h_up)
        
	   # Prior x --> p_z	      
        h_down = slim.fully_connected(x_rep,hps.h_size,tf.nn.relu)
        kl_sum = 0.0        
        kl_sample = 0.0
        # Ladder down
        for i,layer in reversed(list(enumerate(layers))):
            h_down, kl_cur, kl_sam = layer.down(h_down,is_training,temp,lamb)
            kl_sum += kl_cur
            kl_sample += kl_sam
        
        # Decoder: x,z --> y
        xz = tf.concat([slim.flatten(h_down),x_rep],1)
        dec1 = slim.fully_connected(xz,50,tf.nn.relu)
        dec2 = slim.fully_connected(dec1,50,tf.nn.relu)
        dec3 = slim.fully_connected(dec2,50,activation_fn=None)
        mu_y = slim.fully_connected(dec3,1,activation_fn=None)  
        if hps.ignore_sigma_outcome:
            log_dec_noise = tf.zeros(tf.shape(mu_y))
        else:
            log_dec_noise = slim.fully_connected(dec3,1,activation_fn=None)
            
	   # p(y|x,z)
        if hps.out_lik == 'normal':
            dec_noise = tf.exp(tf.clip_by_value(log_dec_noise,-10,10))
            outdist = tf.contrib.distributions.Normal(mu_y,dec_noise)
            self.log_py_x = log_py_x = tf.reduce_sum(outdist.log_prob(y_rep),axis=1)
            self.nats = -1*tf.reduce_mean(logsumexp(tf.reshape(log_py_x - kl_sample,[-1,k])) - tf.log(tf.to_float(k)))
        elif hps.out_lik == 'discretized_logistic':
            self.log_py_x = log_py_x = tf.reduce_sum(discretized_logistic(mu_y,log_dec_noise,binsize=1,sample=y_rep),axis=1)
            outdist = tf.contrib.distributions.Logistic(loc=mu_y,scale = tf.exp(log_dec_noise))
            self.nats = -1*tf.reduce_mean(logsumexp(tf.reshape(tf.reduce_sum(outdist.log_prob(y_rep),axis=1) - kl_sample,[-1,k]))- tf.log(tf.to_float(k)))
        elif hps.out_lik == 'squared_error':
            hps.ignore_sigma_outcome = True            
            self.log_py_x = log_py_x = -tf.reduce_sum(tf.pow(mu_y - y_rep, 2),axis=1) # Gaussian loglik has minus in front            
            self.nats = tf.zeros([1])
        
        # sample y
        if hps.ignore_sigma_outcome:
            self.y_sample = mu_y
        else:
            self.y_sample = outdist.sample()
            
        # To display KL
        self.kl = tf.reduce_mean(kl_sum)

        # ELBO
        log_divergence = tf.reshape(log_py_x - kl_sum,[-1,k]) # shape [batch_size,k]
        if np.abs(hps.alpha-1.0)>1e-3: # prevent zero division
            log_divergence = log_divergence * (1-hps.alpha)
            logF = logsumexp(log_divergence)
            self.elbo = elbo = tf.reduce_mean(logF - tf.log(tf.to_float(k)))/ (1-hps.alpha)
        else:
            logF = logsumexp(log_divergence)
            self.elbo = elbo = tf.reduce_mean(logF - tf.log(tf.to_float(k)))

        self.loss = loss = -elbo

        ### Optimizer        
        self.lr = lr = tf.Variable(0.001,name="learning_rate",trainable=False)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(loss,global_step=global_step)    
        self.init_op=tf.global_variables_initializer()

        #gvs = optimizer.compute_gradients(loss)                    
        #if hps.grad_clip > 0: # gradient clipping
        #    gvs = [(tf.clip_by_value(grad, -hps.grad_clip, hps.grad_clip), var) for grad, var in gvs]
        #self.train_op = optimizer.apply_gradients(gvs)
   

        