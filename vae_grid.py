#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Inference on Gridworld
@author: thomas
"""
import tensorflow as tf
import numpy as np
from tfutils.helpers import anneal_linear, HParams
import matplotlib.pyplot as plt
from rlenv.grid import grid_env as grid_env
from rlutils.policies import egreedy
from rlutils.helpers import make_rl_data, make_test_data, plot_predictions, kl_preds_v2
from pythonutils.helpers import save, make_name, nested_list
import os
import logging

# Tensorflow parser
flags = tf.app.flags
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
flags.DEFINE_string("save_dir", "results/grid", "Results directory.")
flags.DEFINE_string("check_dir", "/tmp/best_model", "Checkpoint directory.")
FLAGS = flags.FLAGS

def get_hps():
    ''' Hyperparameter settings '''
    return HParams(
        # General learning set-up
        network = 1, # which network to run
        n_epochs = 75000, # number of batches
        batch_size = 32, # batch size
        eval_freq = 500, # Evaluation frequency
        test_size = 1500, # Test set size 
        debug = False, # tf debugging
        lr_init = 0.0005, # Initial learning rate
        lr_final_frac = 0.2, # lr_final = lr_init * lr_final_frac
        anneal_frac_lr = 0.7, # percentage of n_epochs to anneal over
        max_grad = None, # gradient clip size, None is no clipping
        verbose = True, # print verbosity
        
        # q(z|x,y) (Variational approximation)
        var_type = ['continuous'], # Distribution for q(z|x,y), either ['continuous'] for Gaussian, or ['discrete'] for categorical
        # repeated if depth>1, for multiple layers use: var_type = ['continuous','discrete'],     
        depth = 1, # depth of stochastic layers
        h_size = 100, # dimensionality in variatonal ladder (deterministic part) 
        resnet = False, # Whether to use Resnet like architecture
        
        # Discrete latent variables
        K = 3, # categories per discrete latent variable 
        N = 3, # number of variables
        tau_init = 2.0, # softmax temperature initial
        tau_final = 0.001, # softmax temperature final
        anneal_frac_tau = 0.7, # anneal fraction
        
        # Continuous latent variables
        z_size = 3, # number of continuous latent variables                
        n_flow = 5, # depth of flow (if n_flow=0 --> no flow)
        ar = False, # type of flow, if False: affine coupling layer (Real NVP), if True: inverse autoregressive flow (IAF)
        ignore_sigma_latent = False, # use sigma in variational approximation
        ignore_sigma_flow = False, # use sigma in flow transformations
       
        # p(y|x,z) (Decoder distribution)
        out_lik = 'discrete', # distribution for p(y|x,z), can be 'normal', 'discrete' or 'discretized logistic'
        ignore_sigma_outcome = True, # Whether to learn the SD of p(y|x,z). 
        #For discrete, whether to sample for categorical or deterministically argmax over the predicted class probabilities.
              
        # VAE objective
        k = 3, # number of importance samples
        alpha = 0.5, # alpha in Renyi alpha divergence
        kl_min = 0.07, # Number of "free bits/nats", only used if kl_min>0
        use_lamb = False, # KL annealing (alternative to kl_min)
        lamb_init = 0.1, # Initial contribution of KL to loss : L = p(y|x,z) + lambda*KL(q|p)
        lamb_final = 1.0, # Final lambda
        anneal_frac_lamb = 0.3, # anneal iteration fraction

        # Reinforcement learning settings
        artificial_data = True, # if True: no RL but sample data across state-space (decorrelated)
        use_target_net = False, # if True: use target net in DQN
        eps_init = 1.0, # initial epsilon in e-greedy action selection
        eps_final = 0.10, # final epsilon
        anneal_frac_eps = 0.6, # fraction of n_epochs to anneal over
        gamma = 0.99, # discount factor
        test_on_policy = False, # if True: plot evalutions while following policy (only useful with artificial_data=False)
            
        # Hyperparameter looping
        n_rep = 10, # number of repetitions per setting
        loop_hyper = False, # If False, no looping (ignores other settings below)
        item1 = 'kl_min', # First hyperparameter to loop over (should appear in settings above)
        seq1 = [0,0.04,0.07,0.10,0.20], # Values to loop over
        item2 = 'use_lamb', # Second hyperparameter
        seq2 = [False, True], # Second loop values
        )
        
def run(hps): 
    ''' Main function: run training and evaluation '''
    Env = grid_env(False)
    Test_env = grid_env(False)    
    if hps.artificial_data:
        s_valid_pre, a_valid_pre, s1_valid_pre, r_valid_pre, term_valid_pre = make_rl_data(Test_env,int(hps.test_size/2))
        s_test_pre, a_test_pre, s1_test_pre, r_test_pre, term_test_pre = make_rl_data(Test_env,hps.test_size)
    
    # Set-up hyperparameter loop
    n_rep = hps.n_rep
    seq1 = hps.seq1
    seq2 = hps.seq2
    results = np.empty([len(seq1),len(seq2),n_rep])
    results_elbo = np.empty([len(seq1),len(seq2),n_rep])
    results_distances = np.empty([len(seq1),len(seq2),3,n_rep])
    av_rewards = nested_list(len(seq1),len(seq2),n_rep)
    
    for j,item1 in enumerate(seq1):
        hps._set(hps.item1,item1)
        for l,item2 in enumerate(seq2):
            hps._set(hps.item2,item2)
            for rep in range(n_rep):
                tf.reset_default_graph()
                hps.lr_final = hps.lr_init*hps.lr_final_frac
                
                # Initialize anneal parameters
                np_lr= anneal_linear(0,hps.n_epochs * hps.anneal_frac_lr,hps.lr_final,hps.lr_init)
                np_temp = anneal_linear(0,hps.n_epochs * hps.anneal_frac_tau,hps.tau_final,hps.tau_init)
                np_lamb = anneal_linear(0,hps.n_epochs * hps.anneal_frac_lamb,hps.lamb_final,hps.lamb_init)
                np_eps = anneal_linear(0,hps.n_epochs * hps.anneal_frac_eps,hps.eps_final,hps.eps_init)
                
                # Build network
                if hps.network == 1:
                    import networks.network_rl as net
                    model = net.Network(hps,Env.observation_shape)
                
                # Check model size
                total_size = 0
                for v in tf.trainable_variables():
                        total_size += np.prod([int(s) for s in v.get_shape()])
                print("Total number of trainable variables: {}".format(total_size))
                
                # Session and initialization
                with tf.Session() as sess:
                    if hps.debug:
                        sess = tf.python.debug.LocalCLIDebugWrapperSession(sess)
                        sess.add_tensor_filter("has_inf_or_nan", tf.python.debug.has_inf_or_nan)
                        
                    sess.run(model.init_op)
                    saver = tf.train.Saver()
                    
                    # Some storage
                    t = []
                    lr = []
                    elbo_keep = []
                    train_nats_keep = []
                    valid_nats_keep = []
                    test_nats_keep = []
                    min_valid_nats = 1e50
                    best_test_nats = 0.0
                    best_elbo = 0.0
                    best_iter = 0
                    died_ep = []
                    epoch_reward = [] 
                    
                    # Train
                    print('Initialized, starting to train')
                    s = Env.reset()
                    for i in range(hps.n_epochs):
                    
                        if not hps.artificial_data: # roll out in Env
                            s_batch = np.empty(np.append(hps.batch_size,Env.observation_shape),dtype='float32')
                            a_batch = np.empty([hps.batch_size,1],dtype='float32')
                            r_batch = np.empty([hps.batch_size,1],dtype='float32')
                            term_batch = np.empty([hps.batch_size,1],dtype='float32')
                            s1_batch = np.empty(np.append(hps.batch_size,Env.observation_shape),dtype='float32')
                            for _ in range(hps.batch_size):
                                Qsa = sess.run(model.Qsa, feed_dict = {model.x       :s[None,:],
                                                                           model.k       : 1,
                                                                           })
                                a = egreedy(Qsa[0],np_eps)
                                s1,r,dead = Env.step([a])
                                s_batch[_,],a_batch[_,],r_batch[_,],s1_batch[_,],term_batch[_,] = s,a,r,s1,dead
                                s = s1
                                #Env.plot()
                                if dead:
                                    s = Env.reset() # process smaller batch
                                    died_ep.extend([i])
                                    
                        else: # Sample some transitions across state-space
                            s_batch, a_batch, s1_batch, r_batch,term_batch = make_rl_data(Env,hps.batch_size)
                        
                        # Calculate targets
                        if hps.use_target_net:
                            Qsa1 = sess.run(model.Qsa_t, feed_dict = {model.x : s1_batch,model.k : 1})
                        else:
                            Qsa1 = sess.run(model.Qsa, feed_dict = {model.x : s1_batch,model.k : 1})
                            
                        Qmax = np.max(Qsa1,axis=1)[:,None]
                        Qmax *= (1. - term_batch)
                        Qtarget_batch = r_batch + hps.gamma * Qmax
                        
                        # store stuff              
                        epoch_reward.extend([np.mean(r_batch)])  
                        
                        # draw batch
                        __,__, np_elbo = sess.run([model.train_op,model.train_op_rl,model.elbo],{ model.x      : s_batch,
                                                                            model.y       : s1_batch,
                                                                            model.a       : a_batch,
                                                                            model.Qtarget : Qtarget_batch,
                                                                            model.lr      : np_lr,
                                                                            model.lamb    : np_lamb,
                                                                            model.temp    : np_temp,
                                                                            model.is_training : True,
                                                                            model.k: hps.k} )
                    
                        # Annealing
                        if i % 250 == 1:
                            np_lr= anneal_linear(i,hps.n_epochs * hps.anneal_frac_lr,hps.lr_final,hps.lr_init)
                            np_temp = anneal_linear(i,hps.n_epochs * hps.anneal_frac_tau,hps.tau_final,hps.tau_init)
                            np_lamb = anneal_linear(i,hps.n_epochs * hps.anneal_frac_lamb,hps.lamb_final,hps.lamb_init)
                            np_eps = anneal_linear(i,hps.n_epochs * hps.anneal_frac_eps,hps.eps_final,hps.eps_init)
                        
                        # Evaluate    
                        if i % hps.eval_freq == 1:
                            if hps.use_target_net:                
                                sess.run([model.copy_op])
                                
                            if (not hps.artificial_data) and hps.test_on_policy:
                                    s_valid, a_valid, s1_valid, r_valid, term_valid = make_test_data(sess,model,Test_env,hps.test_size,epsilon=0.05)
                                    s_test, a_test, s1_test, r_test, term_test = make_test_data(sess,model,Test_env,hps.test_size,epsilon=0.05) 
                            else:
                                s_valid, a_valid, s1_valid, r_valid, term_valid = s_valid_pre, a_valid_pre, s1_valid_pre, r_valid_pre, term_valid_pre
                                s_test, a_test, s1_test, r_test, term_test = s_test_pre, a_test_pre, s1_test_pre, r_test_pre, term_test_pre
             
                            train_elbo,train_nats,train_kl,train_rl_cost = sess.run([model.elbo,model.nats,model.kl,model.rl_cost],{ model.x       : s_batch,
                                                                            model.y       : s1_batch,
                                                                            model.a       : a_batch,
                                                                            model.Qtarget : Qtarget_batch,
                                                                            model.lamb : np_lamb,
                                                                            model.temp    : 0.0001,                                         
                                                                            model.is_training : True,
                                                                            model.k: 40})

                            valid_nats = sess.run(model.nats,{ model.x       : s_valid,
                                                                            model.y       : s1_valid,
                                                                            model.a       : a_valid,
                                                                            model.Qtarget : np.zeros(np.shape(a_valid)),
                                                                            model.lamb : np_lamb,
                                                                            model.temp    : 0.0001,                                         
                                                                            model.is_training : True,
                                                                            model.k: 40})
                            test_nats = sess.run(model.nats,{ model.x       : s_test,
                                                                            model.y       : s1_test,
                                                                            model.a       : a_test,
                                                                            model.Qtarget : np.zeros(np.shape(a_test)),
                                                                            model.lamb : np_lamb,
                                                                            model.temp    : 0.0001,                                         
                                                                            model.is_training : True,
                                                                            model.k: 40})
                            if hps.verbose:
                                print('Step',i,'ELBO: ',train_elbo, 'Training nats:',train_nats, 'Training KL:',train_kl, 'RL cost:',train_rl_cost, 
                                ' \n Valid nats',valid_nats, ' Test set nats',test_nats,
                                  ' \n Average reward in last 50 batches',np.mean(epoch_reward[-50:]), 'Learning rate',np_lr,'Softmax Temp',np_temp,'Epsilon:',np_eps)
                
                            t.extend([i])
                            lr.extend([np_lr])
                            train_nats_keep.extend([train_nats])
                            valid_nats_keep.extend([valid_nats])
                            test_nats_keep.extend([test_nats])
                            elbo_keep.extend([train_elbo])
                            if valid_nats < min_valid_nats:
                                min_valid_nats = valid_nats
                                #best_sample = y_samples # keep the sample
                                best_test_nats = test_nats
                                best_elbo = train_nats
                                best_iter = i
                                saver.save(sess,FLAGS.check_dir)

                    # VAE storage
                    print('Best result in iteration',best_iter,'with valid_nats',min_valid_nats,'and test nats',best_test_nats)
                    saver.restore(sess,FLAGS.check_dir)
                    print('Restored best VAE model')

                    # nats
                    fig = plt.figure()
                    plt.plot(t,train_nats_keep,label='train nats')
                    plt.plot(t,valid_nats_keep,label='valid nats')
                    plt.plot(t,test_nats_keep,label='test nats')
                    plt.plot(t,elbo_keep,label='ELBO')
                    plt.legend(loc=0)
                    if hps.loop_hyper:
                        save(os.path.join(hps.my_dir,'nats_{}={}_{}={}_{}'.format(hps.item1,item1,hps.item2,item2,rep)))  
                    else:
                        save(os.path.join(hps.my_dir,'nats{}'.format(rep)),ext='png',close=True,verbose=False)  
                    results[j,l,rep] = best_test_nats
                    results_elbo[j,l,rep] = best_elbo

                    # Distances from true distribution
                    distances = kl_preds_v2(model,sess,s_test,a_test)
                    results_distances[j,l,:,rep] = distances

                    # Visualize some predictions
                    n_row = 2
                    n_col = 2
                    s_start=np.array([0,0,1,3,5,3])
                    for extra_rep in range(3):
                        if hps.test_on_policy:
                            s_start = plot_predictions(model,sess,n_row,n_col,rep,hps,True,s_start)
                        else:
                            s_start = plot_predictions(model,sess,n_row,n_col,rep,hps,False)                
                        if hps.loop_hyper:
                            name = os.path.join(hps.my_dir,'predictions_{}={}_{}={}_{}'.format(hps.item1,item1,hps.item2,item2,rep)) 
                        else:
                            name = os.path.join(hps.my_dir,'predictions_{}{}'.format(rep,extra_rep)) 
                        save(name,ext='png',close=True,verbose=False)
                        
                    ############# RL ################
                    if not hps.artificial_data:
                        window = 200
                        av_reward = np.convolve(epoch_reward, np.ones((window,))/window, mode='valid')
                        av_rewards[j][l][rep] = av_reward # average rewards of RL agent                    

                    # Show learned behaviour
                    if (not hps.artificial_data) and hps.verbose:
                        print('Start evaluating policy')
                        Env = grid_env(True)
                        Env.reset()
                        for lll in range(100):
                            Qsa = sess.run(model.Qsa, feed_dict = {model.x       :s[None,:],
                                                                           model.k       : 1,
                                                                           })
                            a = egreedy(Qsa[0],0.01)    
                            s,r,dead = Env.step([a])
                            Env.plot()
                            if dead:
                               print('Died in step',lll,', restarting')
                               s = Env.reset() 
                        plt.close()
    # Overall results       
    results_raw = results
    results = np.mean(results,axis=2)
    results_raw_elbo = results_elbo
    results_elbo = np.mean(results_elbo,axis=2)
    results_raw_distances = results_distances
    results_distances = np.mean(results_distances,axis=3)

    logging.info('-------------------- Overall Results --------------------------')
    logging.info('vae' if hps.network == 1 else 'mlp_{}'.format('deterministic' if hps.deterministic else 'stochastic'))
    logging.info('Latent type %s of depth %s',hps.var_type[0],hps.depth)
    logging.info('(z_size,n_flow) %s %s and (n,k) %s %s',hps.z_size,hps.n_flow,hps.N,hps.K)
    logging.info('Results over %s runs',n_rep)
    logging.info('Test nats: %s',results[0,0])
    logging.info('Elbo: %s',-1*results_elbo[0,0])
    logging.info('KL with true distr: %s',results_distances[0,0,:])
    logging.info('Raw data over repetitions \n %s \n %s \n %s',results_raw,results_raw_elbo,results_raw_distances)

    if hps.loop_hyper:
        fig = plt.figure()
        for i in range(results.shape[1]):
            plt.plot([i for i in range(len(seq1))],results[:,i],label='{} = {}'.format(hps.item2,hps.seq2[i]))
        plt.xlabel(hps.item1)
        plt.gca().set_xticklabels(hps.seq1)
        plt.legend(loc=0)
        save(os.path.join(os.getcwd(),FLAGS.save_dir,'run_{}/looped'.format(make_name(hps))),ext='png',close=True,verbose=False)  
    
    if not hps.artificial_data:
        fig = plt.figure()
        for ii in range(len(seq1)):
            for jj in range(len(seq2)):
                signal = np.mean(np.array(av_rewards[ii][jj]),axis=0)
                plt.plot([i for i in range(len(signal))],signal,label='{} = {},{} = {}'.format(hps.item1,hps.seq1[ii],hps.item2,hps.seq2[jj]))
        plt.xlabel('Steps')
        plt.legend(loc=0)
        save(os.path.join(os.getcwd(),FLAGS.save_dir,'run_{}/looped_reward'.format(make_name(hps))),ext='png',close=True,verbose=False)  
    
        
def init_logger(hps,my_dir=None):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
    handlers = [logging.FileHandler(os.path.join(my_dir,'results.txt'),mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level = logging.INFO, format = '%(message)s', handlers = handlers)

def main(_):
    hps = get_hps().parse(FLAGS.hpconfig)
    FLAGS.check_dir = FLAGS.check_dir + str(np.random.randint(0,1e7,1)[0])

    if hps.depth>1 and len(hps.var_type) == 1:
        hps.var_type = [hps.var_type[0] for i in range(hps.depth)]
        print(hps.var_type)    

    if not hps.loop_hyper:
        hps._set('seq1',[hps._items[hps.item1]])
        hps._set('seq2',[hps._items[hps.item2]])

    # logging and saving    
    hps.my_dir = os.path.join(os.getcwd(),FLAGS.save_dir,'{}'.format(make_name(hps)))
    init_logger(hps,hps.my_dir)    
    with open(os.path.join(hps.my_dir,'hps.txt'),'w') as file:
        file.write(repr(hps._items))
        
    run(hps)

if __name__ == "__main__":
    tf.app.run()


