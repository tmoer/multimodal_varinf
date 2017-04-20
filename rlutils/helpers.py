# -*- coding: utf-8 -*-
"""
Grid-world environment
@author: thomas
"""
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rlenv.grid import grid_env as grid_env
from rlutils.policies import egreedy
from scipy.linalg import norm

### other stuff
def make_rl_data(Env,batch_size):
    world = np.zeros([7,7],dtype='int32')
    world[1:6,1] = 1
    world[1:3,4] = 1
    world[4:6,4] = 1
    
    s_data = np.zeros([batch_size,Env.observation_shape],dtype='int32')
    s1_data = np.zeros([batch_size,Env.observation_shape],dtype='int32')
    a_data = np.zeros([batch_size,1],dtype='int32')
    r_data = np.zeros([batch_size,1],dtype='int32')
    term_data = np.zeros([batch_size,1],dtype='int32')
    count = 0
    while count < batch_size:
        i,j,k,l,m,n =  np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1)
        a = np.random.randint(0,4,1)[0]
        do = not bool(world[i,j]) and (not bool(world[k,l])) and (not bool(world[m,n]))
        if do:
            s = np.array([i,j,k,l,m,n]).flatten()
            Env.set_state(s)
            s1 , r , dead = Env.step([a])  
            s_data[count,] = s
            s1_data[count,] = s1
            a_data[count] = a
            r_data[count] = r
            term_data[count] = dead
            count += 1
    return s_data, a_data, s1_data, r_data, term_data

def make_test_data(sess,model,Env,batch_size,epsilon=0.05):
    ''' on policy test data '''
    s_data = np.zeros([batch_size,Env.observation_shape],dtype='int32')
    s1_data = np.zeros([batch_size,Env.observation_shape],dtype='int32')
    a_data = np.zeros([batch_size,1],dtype='int32')
    r_data = np.zeros([batch_size,1],dtype='int32')
    term_data = np.zeros([batch_size,1],dtype='int32')
    count = 0
    s = Env.reset()
    while count < batch_size:
        Qsa = sess.run(model.Qsa, feed_dict = {model.x       :s[None,:],
                                               model.k       : 1,
                                               })
        a = np.array([egreedy(Qsa[0],epsilon)]) 
        s1 = sess.run(model.y_sample,{ model.x       : s[None,:],
             model.y       : np.zeros(np.shape(s))[None,:],
             model.a       : a[:,None],
             model.lamb : 1,
             model.temp    : 0.0001,
             model.is_training : False,
             model.k: 1})
        s_data[count,] = s
        a_data[count] = a
        s1_data[count,] = s1
        Env.set_state(s1)
        term_data[count,] = dead = Env._check_dead()
        if dead:
            s = Env.reset()
        else:
            s = s1
        count += 1           
    return s_data, a_data, s1_data, r_data, term_data
    
def plot_predictions(model,sess,n_row,n_col,run_id,hps,on_policy=False,s=np.array([0,0,1,3,5,3])):
    world = np.zeros([7,7],dtype='int32')
    world[1:6,1] = 1
    world[1:3,4] = 1
    world[4:6,4] = 1

    fig = plt.figure()#figsize=(7,7),dpi=600,aspect='auto')
    for row in range(n_row):
        for col in range(n_col):
            ax1 = fig.add_subplot(n_row,n_col,((n_col*col) + row + 1),aspect='equal')
            # plot the environment            
            ax1.axis('off')
            plt.xlim([-1,8])
            plt.ylim([-1,8])
            plot_predictions
            for i in range(7):
                for j in range(7):
                    if world[i,j]==1:
                        col = "black"
                    else:
                        col = "white"
                    ax1.add_patch(
                        patches.Rectangle(
                            (i,j),1,1,
                            #fill=False,
                            edgecolor='black',
                            linewidth = 2,
                            facecolor = col,),
                        )
                        
            # sample some state
            do = False
            if not on_policy:
                while not do:
                    i,j,k,l,m,n =  np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1),np.random.randint(0,7,1)
                    a = np.random.randint(0,4,1)
                    do = not bool(world[i,j]) and (not bool(world[k,l])) and (not bool(world[m,n]))
                s = np.array([i,j,k,l,m,n]).flatten()
            else:
                i,j,k,l,m,n = s
                Qsa = sess.run(model.Qsa, feed_dict = {model.x       :s[None,:],
                                                           model.k       : 1,
                                                           })
                a = np.array([egreedy(Qsa[0],0.01)])                                 

           # add the start
            ax1.add_artist(plt.Circle((m+0.5,n+0.5),0.3,color='blue'))                    
            ax1.add_artist(plt.Circle((k+0.5,l+0.5),0.3,color='red'))
            ax1.add_artist(plt.Circle((i+0.5,j+0.5),0.3,color='green'))
            if a == 0:
                action = 'up'
            elif a == 1:
                action = 'down'
            elif a == 2:
                action = 'right'
            elif a == 3:
                action = 'left'
            ax1.set_title(action)

            trans = predict(model,sess,s,a)
            for agent in range(3):
                for i in range(7):
                    for j in range(7):
                        if trans[i,j,agent]>0:
                            if agent == 0:
                                col = 'green'
                            elif agent == 1:
                                col = 'red'
                            elif agent == 2:
                                col = 'blue'
                                
                            ax1.add_patch(
                                patches.Rectangle(
                                    (i,j),1,1,
                                    fill=True,
                                    alpha = trans[i,j,agent],
                                    edgecolor='black',
                                    linewidth = 2,
                                    facecolor = col,),
                                )
            if on_policy:
                s1 = sess.run(model.y_sample,{ model.x       : s[None,:],
                             model.y       : np.zeros(np.shape(s[None,:])),
                             model.a       : a[:,None],
                             model.lamb : 1,
                             model.temp    : 0.0001,
                             model.is_training : False,
                             model.k: 1})
                s = s1[0]
    return s
    
def predict(model,sess,some_s,some_a,n_test_samp=200):
    freq = np.zeros([7,7,3])
    for m in range(n_test_samp):
        y_sample = sess.run(model.y_sample,{ model.x       : some_s[None,:],
                                             model.y       : np.zeros(np.shape(some_s))[None,:],
                                             model.a       : some_a[:,None],
                                             model.lamb : 1,
                                             model.temp    : 0.0001,
                                             model.is_training : False,
                                             model.k: 1})  
        y_sample = y_sample.flatten()
        freq[y_sample[0],y_sample[1],0] += 1
        freq[y_sample[2],y_sample[3],1] += 1
        freq[y_sample[4],y_sample[5],2] += 1

    trans = freq/n_test_samp
    return trans

def kl_preds_v2(model,sess,s_test,a_test,n_rep_per_item=200):
    ## Compare sample distribution to ground truth
    Env = grid_env(False)
    n_test_items,state_size = s_test.shape  
    distances = np.empty([n_test_items,3])
    
    for i in range(n_test_items):        
        state = s_test[i,:].astype('int32')
        action = np.round(a_test[i,:]).astype('int32')

        # ground truth   
        state_truth = np.empty([n_rep_per_item,s_test.shape[1]])     
        for o in range(n_rep_per_item):
            Env.set_state(state.flatten())
            s1,r,dead = Env.step(action.flatten())
            state_truth[o,:] = s1
        truth_count,bins = np.histogramdd(state_truth,bins=[np.arange(8)-0.5]*state_size) 
        truth_prob = truth_count/n_rep_per_item
        
        # predictions of model
        y_sample = sess.run(model.y_sample,{ model.x       : state[None,:].repeat(n_rep_per_item,axis=0),
                                           model.y       : np.zeros(np.shape(state[None,:])).repeat(n_rep_per_item,axis=0),
                                           model.a       : action[None,:].repeat(n_rep_per_item,axis=0),
                                           model.Qtarget : np.zeros(np.shape(action[None,:])).repeat(n_rep_per_item,axis=0),
                                           model.lr      : 0,
                                           model.lamb : 1,
                                           model.temp    : 0.00001,                                         
                                           model.is_training : False,
                                           model.k: 1})
        sample_count,bins = np.histogramdd(y_sample,bins=[np.arange(8)-0.5]*state_size) 
        sample_prob = sample_count/n_rep_per_item

        distances[i,0]= np.sum(truth_prob*(np.log(truth_prob+1e-5)-np.log(sample_prob+1e-5))) # KL(p|p_tilde)
        distances[i,1]= np.sum(sample_prob*(np.log(sample_prob+1e-5)-np.log(truth_prob+1e-5))) # Inverse KL(p_tilde|p)
        distances[i,2]= norm(np.sqrt(truth_prob) - np.sqrt(sample_prob))/np.sqrt(2)
        
    return np.mean(distances,axis=0)