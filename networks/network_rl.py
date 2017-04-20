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
    ''' VAE & RL template ''' 
    
    def __init__(self,hps,state_dim,binsize=6):
        # binsize = the number of discrete categories per dimension of the state (x and y below). 
        # Input and output are normalized over this quantity.

        # placeholders
        self.x = x = tf.placeholder("float32", shape=[None,state_dim])
        self.y = y = tf.placeholder("float32", shape=[None,state_dim])
        self.a = a = tf.placeholder("float32", shape=[None,1])
        self.Qtarget = Qtarget = tf.placeholder("float32", shape=[None,1])
        
        self.is_training = is_training = tf.placeholder("bool") # if True: sample from q, else sample from p
        self.k = k = tf.placeholder('int32') # number of importance samples
        self.temp = temp = tf.Variable(5.0,name='temperature',trainable=False) # Temperature for discrete latents
        self.lamb = lamb = tf.Variable(1.0,name="lambda",trainable=False) # Lambda for KL annealing

        xa = tf.concat([x/binsize,a],axis=1)
        # Importance sampling: repeats along second dimension
        xa_rep = repeat_v2(xa,k)
        y_rep = repeat_v2(y/binsize,k)
        
        # RL part of the graph        
        with tf.variable_scope('q_net'):
            rl1 = slim.fully_connected(x,50,tf.nn.relu)
            rl2 = slim.fully_connected(rl1,50,tf.nn.relu)
            rl3 = slim.fully_connected(rl2,50,activation_fn=None)
            self.Qsa = Qsa = slim.fully_connected(rl3,4,activation_fn=None)   

        if hps.use_target_net:
           
            with tf.variable_scope('target_net'):
                rl1_t = slim.fully_connected(x,50,tf.nn.relu)
                rl2_t = slim.fully_connected(rl1_t,50,tf.nn.relu)
                rl3_t = slim.fully_connected(rl2_t,50,activation_fn=None)
                self.Qsa_t = slim.fully_connected(rl3_t,4,activation_fn=None)   
            
            copy_ops = []
            q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net')
            tar_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
                
            for tar,q in zip(q_var,tar_var):
                copy_op = q.assign(tar)
                copy_ops.append(copy_op)    
            self.copy_op = tf.group(*copy_ops, name='copy_op')          
       
        a_onehot = tf.one_hot(tf.to_int32(tf.squeeze(a,axis=1)),4,1.0,0.0)
        Qs = tf.reduce_sum(a_onehot*Qsa,reduction_indices=1) ## identify Qsa based on a
        self.rl_cost = rl_cost = tf.nn.l2_loss(Qs - Qtarget)
	   # Batch norm: skip for now 
        
        # Encoder x,y --> h
        xy = tf.concat([xa_rep,y_rep],1) # concatenate along last dim
        h_up = slim.fully_connected(xy,hps.h_size,tf.nn.relu)
        
        # Initialize ladders
        layers = []
        for i in range(hps.depth):
            layers.append(Latent_Layer(hps,hps.var_type[i],i))
            
        # Ladder up
        for i,layer in enumerate(layers):
            h_up = layer.up(h_up)
        
        # Ladder down
	   # Prior x --> p_z	      
        h_down = slim.fully_connected(xa_rep,hps.h_size,tf.nn.relu)
        kl_sum = 0.0        
        kl_sample = 0.0
        for i,layer in reversed(list(enumerate(layers))):
            h_down, kl_cur, kl_sam = layer.down(h_down,is_training,temp,lamb)    
            kl_sum += kl_cur
            kl_sample += kl_sam
        
        # Decoder: x,z --> y
        xz = tf.concat([slim.flatten(h_down),xa_rep],1)
        dec1 = slim.fully_connected(xz,250,tf.nn.relu)
        dec2 = slim.fully_connected(dec1,250,tf.nn.relu)
        dec3 = slim.fully_connected(dec2,250,activation_fn=None)
        mu_y = slim.fully_connected(dec3,state_dim,activation_fn=None)       
        
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
            y_sample = outdist.sample() if not hps.ignore_sigma_outcome else mu_y
            self.y_sample = tf.to_int32(tf.round(tf.clip_by_value(y_sample,0,1)*binsize))
        elif hps.out_lik == 'discretized_logistic':
            self.log_py_x = log_py_x = tf.reduce_sum(discretized_logistic(mu_y,log_dec_noise,binsize=1,sample=y_rep),axis=1)
            outdist = tf.contrib.distributions.Logistic(loc=mu_y,scale = tf.exp(log_dec_noise))
            self.nats = -1*tf.reduce_mean(logsumexp(tf.reshape(tf.reduce_sum(outdist.log_prob(y_rep),axis=1) - kl_sample,[-1,k]))- tf.log(tf.to_float(k)))
            y_sample = outdist.sample() if not hps.ignore_sigma_outcome else mu_y     
            self.y_sample = tf.to_int32(tf.round(tf.clip_by_value(y_sample,0,1)*binsize))
        elif hps.out_lik == 'discrete':
            logits_y = slim.fully_connected(dec3,state_dim*(binsize+1),activation_fn=None)
            logits_y = tf.reshape(logits_y,[-1,state_dim,binsize+1])
            disc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y,labels=tf.to_int32(tf.round(y_rep*6)))
            self.log_py_x = log_py_x = -tf.reduce_sum(disc_loss,[1])
            self.nats = -1*tf.reduce_mean(logsumexp(tf.reshape(log_py_x - kl_sample,[-1,k])) - tf.log(tf.to_float(k)))
            outdist = tf.contrib.distributions.Categorical(logits=logits_y)
            self.y_sample = outdist.sample() if not hps.ignore_sigma_outcome else tf.argmax(logits_y,axis=2)
            
        # To display
        self.kl = tf.reduce_mean(kl_sum)
        
        # ELBO
        log_divergence = tf.reshape(log_py_x - kl_sum,[-1,k]) # shape [batch_size,k]
        if np.abs(hps.alpha-1.0)>1e-3: # use Renyi alpha-divergence
            log_divergence = log_divergence * (1-hps.alpha)
            logF = logsumexp(log_divergence)
            self.elbo = elbo = tf.reduce_mean(logF - tf.log(tf.to_float(k)))/ (1-hps.alpha)
        else:
            # use KL divergence
            self.elbo = elbo = tf.reduce_mean(log_divergence)
        self.loss = loss = -elbo

        ### Optimizer        
        self.lr = lr = tf.Variable(0.001,name="learning_rate",trainable=False)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        if hps.max_grad != None:
            grads_and_vars = optimizer.compute_gradients(loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
              if grad is not None:
                grads_and_vars[idx] = (tf.clip_by_norm(grad, hps.max_grad), var)
            self.train_op = optimizer.apply_gradients(grads_and_vars)
            self.grads_and_vars = grads_and_vars
        else:
            self.train_op = optimizer.minimize(loss,global_step=global_step)
            self.grads_and_vars = tf.constant(0)
            
        self.train_op_rl = optimizer.minimize(rl_cost)
        self.init_op=tf.global_variables_initializer()        