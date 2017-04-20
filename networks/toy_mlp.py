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
from tfutils.helpers import repeat_v2
from tfutils.distributions import logsumexp, DiagonalGaussian

class Network(object):
    ''' Generic VAE template ''' 
    
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
        
        if not hps.deterministic:
            # stochastic noise input to network
            z_dist = DiagonalGaussian(tf.zeros([tf.shape(x_rep)[0],hps.z_size]),tf.zeros([tf.shape(x_rep)[0],hps.z_size]))
            z = z_dist.sample
            logpz = tf.reduce_sum(z_dist.logps(z),axis=1)
            xz = tf.concat([z,x_rep],1)
        else:
            xz = x_rep
            logpz = 0.0
            hps.ignore_sigma_outcome = True
            
        # Decoder: x,z --> y
        dec1 = slim.fully_connected(xz,50,tf.nn.relu)
        dec2 = slim.fully_connected(dec1,50,tf.nn.relu)
        dec3 = slim.fully_connected(dec2,50,activation_fn=None)
        mu_y = slim.fully_connected(dec3,1,activation_fn=None)  
             
        if hps.ignore_sigma_outcome:
            log_dec_noise = tf.zeros(tf.shape(mu_y))
        else:
            log_dec_noise = slim.fully_connected(dec3,1,activation_fn=None)
        
        kl_sum = tf.zeros(tf.shape(mu_y))
	   # p(y|x,z)
        dec_noise = tf.exp(tf.clip_by_value(log_dec_noise,-10,10))
        outdist = tf.contrib.distributions.Normal(mu_y,dec_noise)
        self.log_py_x = log_py_x = tf.reduce_sum(outdist.log_prob(y_rep),axis=1)
        self.nats = -1*tf.reduce_mean(logsumexp(tf.reshape(log_py_x + logpz,[-1,k])) - tf.log(tf.to_float(k)))
            
        if hps.ignore_sigma_outcome:
            self.y_sample = mu_y
        else:
            self.y_sample = outdist.sample()
        # To display
        self.kl = tf.reduce_mean(kl_sum)

        # ELBO
        log_divergence = tf.reshape(log_py_x + logpz,[-1,k]) # shape [batch_size,k]
        logF = logsumexp(log_divergence)
        self.elbo = elbo = tf.reduce_mean(logF - tf.log(tf.to_float(k)))
        self.loss = loss = -elbo

        ### Optimizer        
        self.lr = lr = tf.Variable(0.001,name="learning_rate",trainable=False)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(loss,global_step=global_step)    
        self.init_op=tf.global_variables_initializer()