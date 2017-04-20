# -*- coding: utf-8 -*-
"""
Tensorflow variational inference layers
@author: thomas
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tfutils.helpers import split
from tfutils.distributions import gumbel_softmax, DiagonalGaussian

class Latent_Layer(object):
    ''' Latent Layer object '''
    
    def __init__(self,hps,var_type,depth,is_top=True):
        self.hps = hps
        self.var_type = var_type # discrete or continuous
        self.depth = depth
        self.is_top = is_top
            
    def up(self,h):
        h_up = slim.fully_connected(h,self.hps.h_size,activation_fn=tf.nn.relu)
        
        if self.var_type == 'discrete':
            # q_z
            self.K = K = self.hps.K
            self.N = N = self.hps.N
            h_up = slim.fully_connected(h_up,K*N,activation_fn=None)
            self.logits_q  = tf.reshape(h_up,[-1,K]) # unnormalized logits for N separate K-categorical distributions (shape=(batch_size*N,K))
            
            h_out = slim.fully_connected(h_up,self.hps.h_size,activation_fn=None)
        
        elif self.var_type == 'continuous':
            hps =  self.hps
            z_size = hps.z_size
            h_size = hps.h_size
                    
            h_up = slim.fully_connected(h_up,h_size,activation_fn=None)
            h_up = slim.fully_connected(h,z_size*2 + h_size,activation_fn=None)
            self.qz_mean, self.qz_logsd, h_out = split(h_up, 1, [z_size, z_size, h_size])
            
        if self.hps.resnet:
            return h + 0.2 * h_out
        else:
            return h_out
        
    def down(self,h,is_training,temp,lamb):
        
        h_down = slim.fully_connected(h,self.hps.h_size,tf.nn.relu)

        if self.var_type == 'discrete':
            self.K = K = self.hps.K
            self.N = N = self.hps.N
            if self.is_top:
                h_down = slim.fully_connected(h_down,K*N + self.hps.h_size,activation_fn=None)
                logits_p, h_det = split(h_down,1,[K*N] + [self.hps.h_size]) 
                logits_q = self.logits_q
            else:
                # top down inference
                h_down = slim.fully_connected(h_down,2*K*N + self.hps.h_size,activation_fn=None)
                logits_p,logits_r, h_det = split(h_down,1,[K*N]*2 + [self.hps.h_size]) 
                logits_q = self.logits_q + tf.reshape(logits_r,[-1,K])                
                
            self.logits_p = logits_p = tf.reshape(logits_p,[-1,K])
            self.p_z = tf.nn.softmax(logits_p)
            self.q_z = tf.nn.softmax(logits_q)
            
            # Sample z
            z = tf.cond(is_training,
                    lambda: tf.reshape(gumbel_softmax(logits_q,temp,hard=False),[-1,N,K]),
                    lambda: tf.reshape(gumbel_softmax(logits_p,temp,hard=False),[-1,N,K])
                    )    
            
            # KL divergence
            kl_discrete = tf.reshape(self.q_z*(tf.log(self.q_z+1e-20)-tf.log(self.p_z+1e-20)),[-1,N,K])
            kl = tf.reduce_sum(kl_discrete,[2]) # sum over number of categories
            
            # pass on
            h_down = tf.concat([slim.flatten(z),h_det],1)
            h_out = slim.fully_connected(h_down,self.hps.h_size,activation_fn=tf.nn.relu)
            
        elif self.var_type == 'continuous': 
            hps =  self.hps
            z_size = hps.z_size
            h_size = hps.h_size
            
            h_down = slim.fully_connected(h_down,self.hps.h_size,activation_fn=None)
            
            if self.is_top:
                h_down = slim.fully_connected(h_down, 2 * z_size + h_size,activation_fn=None)
                pz_mean, pz_logsd, h_det = split(h_down, 1, [z_size] * 2 + [h_size] * 1)
                qz_mean = self.qz_mean
                qz_logsd = self.qz_logsd
            else:
                # top down inference
                h_down = slim.fully_connected(h_down, 4 * z_size + h_size,activation_fn=None)
                pz_mean, pz_logsd, rz_mean, rz_logsd, h_det = split(h_down, 1, [z_size] * 4 + [h_size] * 1)
                qz_mean = self.qz_mean + rz_mean
                qz_logsd = self.qz_logsd + rz_logsd
            
            # identify distributions
            if self.hps.ignore_sigma_latent:
                prior = DiagonalGaussian(pz_mean,tf.zeros(tf.shape(pz_mean)))
                posterior = DiagonalGaussian(qz_mean,tf.zeros(tf.shape(qz_mean)))
            else:
                prior = DiagonalGaussian(pz_mean,2*pz_logsd)
                posterior = DiagonalGaussian(qz_mean,2*qz_logsd)
            
            # sample z
            z = tf.cond(is_training,
                        lambda: posterior.sample,
                        lambda: prior.sample)
                    
            # KL Divergence with flow
            z, kl = tf.cond(is_training,
                             lambda: kl_train(z,prior,posterior,hps),
                             lambda: kl_test(z))
            
            # output                        # pass on
            h_down = tf.concat([slim.flatten(z),h_det],1)
            h_out = slim.fully_connected(h_down,self.hps.h_size,activation_fn=tf.nn.relu)        
        
        # Manipulate KL divergence
        kl_sample = kl
        if self.hps.kl_min > 0: # use free-bits/nats (Kingma, 2016)
            kl_ave = tf.reduce_mean(kl,[0],keep_dims=True) # average over mini-batch        
            kl_max = tf.maximum(kl_ave,self.hps.kl_min)
            kl = tf.tile(kl_max,[tf.shape(kl)[0],1]) # shape: [batch_size * k,latent_size]
        if self.hps.use_lamb: # use warm-up
            kl = lamb * kl
        
        kl_sum = tf.reduce_sum(kl,[1]) # shape [batch_size*k,]
        kl_sample = tf.reduce_sum(kl_sample,[1])
        
        if self.hps.resnet:
            return h + 0.2 * h_out, kl_sum
        else:
            return h_out, kl_sum, kl_sample

def kl_test(z):
    kl = tf.zeros([1,1])
    return z, kl
            
def kl_train(z,prior,posterior,hps):
    # push prior through AR layer
    logqs = posterior.logps(z)
    if hps.n_flow > 0:
        nice_layers = []
        print('Does this print')
        for i in range(hps.n_flow):
            nice_layers.append(nice_layer(tf.shape(z),hps,'nice{}'.format(i),ar=hps.ar))
        
        for i,layer in enumerate(nice_layers):
            z,log_det = layer.forward(z)
            logqs += log_det
            
    # track the KL divergence after transformation     
    logps = prior.logps(z)
    kl = logqs - logps
    return z, kl

### Autoregressive layers
class nice_layer:
    ''' Autoregressive layer with easy inverse (real-nvp layer) '''
    
    def __init__(self,input_shape,hps,name,n_hidden=20,pattern=2,ar=False):
        self.name = name # variable scope
        self.batch_size = input_shape[0]
        self.latent_size = input_shape[1]
        self.n_hidden = n_hidden
        self.hps = hps
        self.ar = ar
        # calculate mask
        self.mask = self._get_mask()
        
    def _get_mask(self):
        numbers = np.random.binomial(1,0.5,[1,self.hps.z_size])       
        mask = tf.tile(tf.constant(numbers,dtype='bool'),[self.batch_size,1])
        return mask
    
    def _get_mu_and_sigma(self,z):
        # predict mu and sigma
        z_mask = z * tf.to_float(self.mask)
        mid = slim.fully_connected(z_mask,self.n_hidden,activation_fn=tf.nn.relu)
        mu_pred = slim.fully_connected(mid,self.hps.z_size,activation_fn=None) 
        log_sigma_pred = slim.fully_connected(mid,self.hps.z_size,activation_fn=None)
        
        # inverse mask the outcome
        mu_out = mu_pred * tf.to_float(tf.logical_not(self.mask))
        log_sigma_out = log_sigma_pred * tf.to_float(tf.logical_not(self.mask))
        return mu_out, log_sigma_out
    
    def forward(self,z):
        if not self.ar:
            mu,log_sigma = self._get_mu_and_sigma(z)
        else:
            # permute z
            z = tf.reshape(z,[-1]+[1]*self.hps.z_size)
            perm = np.random.permutation(self.hps.z_size)+1
            z = tf.transpose(z,np.append([0],perm))
            z = tf.reshape(z,[-1,self.hps.z_size])
            mu,log_sigma = ar_layer(z,self.hps,n_hidden=self.n_hidden)
        log_sigma = tf.clip_by_value(log_sigma,-5,5)
        if not self.hps.ignore_sigma_flow:
            y = z * tf.exp(log_sigma) + mu
            log_det = -1 * log_sigma
        else:
            y = z + mu
            log_det = 0.0
        return y,log_det        
        
    def backward(self,y):
        mu,log_sigma = self._get_mu_and_sigma(y)
        log_sigma = tf.clip_by_value(log_sigma,-5,5)
        if not self.hps.ignore_sigma_flow:
            z = (y - mu)/tf.exp(log_sigma)
            log_det = log_sigma
        else:
            z = y - mu
            log_det = 0.0
        return z,log_det
 
def ar_layer(z0,hps,n_hidden=10):
    ''' old iaf layer '''
    # Repeat input
    z_rep = tf.reshape(tf.tile(z0,[1,hps.z_size]),[-1,hps.z_size])
    
    # make mask    
    mask = tf.sequence_mask(tf.range(hps.z_size),hps.z_size)[None,:,:]
    mask = tf.reshape(tf.tile(mask,[tf.shape(z0)[0],1,1]),[-1,hps.z_size])
    
    # predict mu and sigma
    z_mask = z_rep * tf.to_float(mask)
    mid = slim.fully_connected(z_mask,n_hidden,activation_fn=tf.nn.relu)
    pars = slim.fully_connected(mid,2,activation_fn=None)
    pars = tf.reshape(pars,[-1,hps.z_size,2])    
    mu, log_sigma = tf.unstack(pars,axis=2)
    return mu, log_sigma 