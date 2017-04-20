# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:26:00 2017

@author: thomas
"""
import numpy as np
from policies import egreedy

def rollout(Env,hps,model,sess,s,epsilon=0.05):
    ''' Q-learning + e-greedy roll-out '''
    s_batch = np.empty(np.append(hps.batch_size,Env.observation_shape),dtype='float32')
    a_batch = np.empty([hps.batch_size,1],dtype='float32')
    r_batch = np.empty([hps.batch_size,1],dtype='float32')
    term_batch = np.empty([hps.batch_size,1],dtype='float32')
    s1_batch = np.empty(np.append(hps.batch_size,Env.observation_shape),dtype='float32')
    for _ in range(hps.batch_size):
        Qsa = sess.run(model.Qsa, feed_dict = {model.x       : s[None,:],
                                               model.k       : 1})
        a = egreedy(Qsa[0],epsilon)
        s1,r,dead = Env.step([a])
        s_batch[_,],a_batch[_,],r_batch[_,],s1_batch[_,],term_batch[_,] = s,a,r,s1,dead
        s = s1
        if dead:
            s = Env.reset()
    
    # Calculate targets
    if hps.use_target_net:
        Qsa1 = sess.run(model.Qsa_t, feed_dict = {model.x : s1_batch,model.k : 1})
    else:
        Qsa1 = sess.run(model.Qsa, feed_dict = {model.x : s1_batch,model.k : 1})
        
    Qmax = np.max(Qsa1,axis=1)[:,None]
    Qmax *= (1. - term_batch)
    Qtarget_batch = r_batch + hps.gamma * Qmax
    
    return s_batch, a_batch, s1_batch, r_batch, term_batch, Qtarget_batch, s, Env