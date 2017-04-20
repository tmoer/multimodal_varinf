#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational inference on Toy Domain
@author: thomas
"""
import tensorflow as tf
import numpy as np
import plotter.Plotter2 as Plotter
from tfutils.helpers import anneal_linear, HParams
import matplotlib.pyplot as plt
import os
from pythonutils.helpers import save, make_name
import logging

# settings
flags = tf.app.flags
flags.DEFINE_string("save_dir", "results/toy", "Logging directory.")
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
FLAGS = flags.FLAGS
    
def get_hps():
    ''' Hyperparameter settings '''
    return HParams(
        dataset = 2, # which dataset to run on
        network = 1, # which network to run
    #
        data_size = 2000,    
        n_epochs = 30000,
        batch_size = 64,
        eval_freq = 500, # Evaluate every .. steps
        debug = False,
    
    # Learning
        lr_init = 0.005,
        lr_final = 0.0005,
        anneal_frac_lr = 0.7,
        
    # Latent dimensions
        #var_type = ['continuous','discrete'],
        var_type = ['continuous'],        
        depth = 1, # depth of stochastic layers
        h_size = 30, # representation size in stoch layers 
        deterministic = False, # used for the MLP only
        
        # discrete
        K = 3, # categories per discrete latent 
        N = 3, # number of discrete latents
        tau_init = 2.0, 
        tau_final = 0.001,
        anneal_frac_tau = 0.7,
        
        # cont
        z_size = 3, # number of cont latents                
        n_flow = 5, # depth of flow (only for continuous latents, for now)
        ar = False,
        ignore_sigma_latent = False,
        ignore_sigma_flow = False,
         
    # KL divergence
        k = 3,
        alpha = 0.5,
        kl_min = 0.07, # Number of "free bits/nats", only used if >0
        use_lamb = False,
        lamb_init = 0.1, # Annealing KL contribution. Don't use together with kl_min>0
        lamb_final = 1.0,
        anneal_frac_lamb = 0.3,
    
    # Architecture
        ladder = True, # For ladder = True, top-down inference
        resnet = False, # Whether to use Resnet connections
        
    # Outcome
        out_lik = 'normal', #
        ignore_sigma_outcome = False, 
        
    # Specify how to loop over hyperparameter settings
        loop_hyper = False,
        item1 = 'lr_init',
        seq1 = [0.005],
        item2 = 'kl_min',
        seq2 = [0.07],
        n_rep = 10,
        verbose = True
        )
            
def run(hps): 
    if hps.verbose:
        plt.ion()
    ## Generate data
    if hps.dataset == 1:
        from datasets import Dataset as Dataset
    elif hps.dataset == 2:
        from datasets import Dataset2 as Dataset    
    
    Data = Dataset.Dataset(hps.data_size)
    Data_valid = Dataset.Dataset(int(hps.data_size/4))
    Data_test = Dataset.Dataset(int(hps.data_size/4))
    
    n_rep = hps.n_rep
    seq1 = hps.seq1
    seq2 = hps.seq2
    results = np.empty([len(seq1),len(seq2),n_rep])
    results_elbo = np.empty([len(seq1),len(seq2),n_rep])
    for j,item1 in enumerate(seq1):
        hps._set(hps.item1,item1)
        for l,item2 in enumerate(seq2):
            hps._set(hps.item2,item2)
            for rep in range(n_rep):
                tf.reset_default_graph()
            
                # Initialize anneal parameters
                np_lr= anneal_linear(0,hps.n_epochs * hps.anneal_frac_lr,hps.lr_final,hps.lr_init)
                np_temp = anneal_linear(0,hps.n_epochs * hps.anneal_frac_tau,hps.tau_final,hps.tau_init)
                np_lamb = anneal_linear(0,hps.n_epochs * hps.anneal_frac_lamb,hps.lamb_final,hps.lamb_init)
        
                Plot = Plotter.Plotter()
                Plot.plot_data(Data)
                # Build network
                if hps.network == 1:
                    import networks.toy_vae as net
                    model = net.Network(hps)
                elif hps.network == 2:
                    import networks.toy_mlp as net
                    model = net.Network(hps)
                    
                # Check model size
                total_size = 0
                for v in tf.trainable_variables():
                        total_size += np.prod([int(s) for s in v.get_shape()])
                print("Total number of trainable variables: {}".format(total_size))
                
                # Session and initialization
                sess = tf.Session()
                if hps.debug:
                    sess = tf.python.debug.LocalCLIDebugWrapperSession(sess)
                    sess.add_tensor_filter("has_inf_or_nan", tf.python.debug.has_inf_or_nan)
                    
                
                np_x, np_y = Data.next_batch_random(hps.batch_size)
                sess.run(model.init_op, feed_dict = {model.x       : np_x[:,None],
                                               model.y       : np_y[:,None]})
                
                # Some storage
                t = []
                lr = []
                neg_elbo_keep = []
                train_nats_keep = []
                valid_nats_keep = []
                test_nats_keep = []
                min_valid_nats = 1e50
                best_sample = []
                best_test_nats = 0.0
                best_elbo = 0.0
                best_iter = 0
                
                # Train
                print('Initialized, starting to train')
                for i in range(hps.n_epochs):
                    # draw batch
                    np_x, np_y = Data.next_batch_epoch(hps.batch_size)
                    _, np_elbo = sess.run([model.train_op,model.elbo],{ model.x       : np_x[:,None],
                                                                        model.y       : np_y[:,None],
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
                    
                    # Evaluate    
                    if i % hps.eval_freq == 1:
                        train_elbo,train_nats,train_kl = sess.run([model.elbo,model.nats,model.kl],{model.x: Data.X[:,None],
                                                         model.y: Data.Y[:,None],
                                                         model.lamb : np_lamb,
                                                         model.temp    : 0.0001,                                         
                                                         model.is_training : True, # to sample from q(z|y,x), we're not running the train_op anyway
                                                         model.k: 40})
    
                        valid_nats = sess.run(model.nats,{model.x: Data_valid.X[:,None],
                                                         model.y: Data_valid.Y[:,None],
                                                         model.lamb : np_lamb,
                                                         model.temp    : 0.0001,
                                                         model.is_training : True, # to sample from q(z|y,x), we're not running the train_op anyway
                                                         model.k: 40})
                                                         
                        test_nats = sess.run(model.nats,{model.x: Data_test.X[:,None],
                                                         model.y: Data_test.Y[:,None],
                                                         model.lamb : np_lamb,
                                                         model.temp    : 0.0001,
                                                         model.is_training : True, # to sample from q(z|y,x), we're not running the train_op anyway
                                                         model.k: 40})
                        if hps.verbose:
                            print('Step',i,'ELBO: ',train_elbo, 'Training nats:',train_nats, 'Training KL:',train_kl, 'Valid nats',valid_nats, 
                                  ' \n Test set nats',test_nats, 'Learning rate',np_lr,'Softmax Temp',np_temp,)
                
                        # draw new data
                        y_samples = sess.run(model.y_sample,{model.x: Data_test.X[:,None],
                                                             model.y: Data_test.X[:,None],
                                                             model.lamb : np_lamb,
                                                             model.temp    : 0.0001,
                                                             model.is_training : False,
                                                             model.k: 1})
                        t.extend([i])
                        lr.extend([np_lr])
                        train_nats_keep.extend([train_nats])
                        valid_nats_keep.extend([valid_nats])
                        test_nats_keep.extend([test_nats])
                        neg_elbo_keep.extend([train_elbo])
                        #Plot.plot_lr(t,lr)
                        if hps.verbose:
                            Plot.plot_samples(Data_test.X[:,None],y_samples)
                                        
                        if valid_nats < min_valid_nats:
                            min_valid_nats = valid_nats
                            best_sample = y_samples # keep the sample
                            best_test_nats = test_nats
                            best_elbo = train_nats
                            best_iter = i
                            
                Plot.plot_samples(Data_test.X[:,None],best_sample)
                save(os.path.join(hps.my_dir,'sample{}'.format(rep)),ext='png',close=True,verbose=False)    
                print('Best result in iteration',best_iter,'with valid_nats',min_valid_nats,'and test nats',best_test_nats)
                
                fig = plt.figure()
                plt.plot(t,train_nats_keep,label='train nats')
                plt.plot(t,valid_nats_keep,label='valid nats')
                plt.plot(t,test_nats_keep,label='test nats')
                plt.plot(t,test_nats_keep,label='negative ELBO')
                plt.legend(loc=0)
                fig.canvas.draw()
                save(os.path.join(hps.my_dir,'nats{}'.format(rep)),ext='png',close=True,verbose=False)  
                  
                results[j,l,rep] = best_test_nats
                results_elbo[j,l,rep] = best_elbo

    results_raw = results
    results_raw_elbo = results_elbo
    results = np.mean(results,axis=2)
    results_elbo = np.mean(results_elbo,axis=2)

    logging.info('-------------------- Overall Results --------------------------')
    logging.info('vae' if hps.network == 1 else 'mlp_{}'.format('deterministic' if hps.deterministic else 'stochastic'))
    logging.info('Latent type %s of depth %s',hps.var_type[0],hps.depth)
    logging.info('(z_size,n_flow) %s %s and (n,k) %s %s',hps.z_size,hps.n_flow,hps.N,hps.K)
    logging.info('Results over %s runs',n_rep)
    logging.info('Test nats: %s',results[0][0])
    logging.info('Elbo: %s',-1*results_elbo[0][0])
    logging.info('Raw data over repetitions \n %s \n %s',results_raw,results_raw_elbo)

    if hps.loop_hyper:
        fig = plt.figure()
        for i in range(results.shape[1]):
            plt.plot([i for i in range(len(seq1))],results[:,i],label='{} = {}'.format(hps.item2,hps.seq2[i]))
        plt.xlabel(hps.item1)
        plt.gca().set_xticklabels(hps.seq1)
        plt.legend(loc=0)
        save(os.path.join(os.getcwd(),FLAGS.save_dir,'run_{}/looped'.format(make_name(hps))),ext='png',close=True,verbose=False)  


def init_logger(hps,my_dir=None):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
    handlers = [logging.FileHandler(os.path.join(my_dir,'results.txt'),mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level = logging.INFO, format = '%(message)s', handlers = handlers)

def main(_):
    hps = get_hps().parse(FLAGS.hpconfig)

    if hps.depth>1 and len(hps.var_type) == 1:
        hps.var_type = [hps.var_type[0] for i in range(hps.depth)]
        print(hps.var_type)    

    if not hps.loop_hyper:
        hps._set('seq1',[hps._items[hps.item1]])
        hps._set('seq2',[hps._items[hps.item2]])
    
    # logging and saving    
    hps.my_dir = os.path.join(os.getcwd(),FLAGS.save_dir,'run_{}'.format(make_name(hps)))
    init_logger(hps,hps.my_dir)    
    with open(os.path.join(hps.my_dir,'hps.txt'),'w') as file:
        file.write(repr(hps._items))
        
    run(hps)

if __name__ == "__main__":
    tf.app.run()
