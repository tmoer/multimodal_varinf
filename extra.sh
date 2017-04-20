#!/bin/bash

echo "Starting extra simulations, with multiple layer VAEs"
python3 vae_main.py --hpconfig network=1,n_rep=10,var_type='continuous-discrete',depth=2,K=3,N=3,z_size=3,n_flow=5,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous-discrete',depth=2,K=8,N=4,z_size=8,n_flow=6,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=False,use_target_net=True,test_on_policy=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous-discrete',depth=2,z_size=8,n_flow=6,ar=False,verbose=False &&

echo "Starting extra simulations, with some hyperloop examples"
python3 vae_main.py --hpconfig network=1,n_rep=10,var_type='continuous-discrete',depth=2,K=3,N=3,z_size=3,n_flow=5,verbose=False &&
python3 vae_main.py --hpconfig network=1,n_rep=10,var_type='discrete',K=3,N=3,loop_hyper=True,item1='lr',seq1='0.005-0.0005-0.00005',item2='kl_min',seq2='0-0.05-0.10-0.20',verbose=False &&
python3 vae_grid.py --hpconfig var_type='discrete',K=4,N=8,loop_hyper=True,n_epochs=50000,n_rep=3,item1='lr_init',seq1='0.005-0.0005-0.00005',item2='kl_min',seq2='0-0.05-0.10-0.30-1.0',verbose=False &&
python3 vae_grid.py --hpconfig var_type='discrete',K=4,N=8,loop_hyper=True,n_rep=3,lr=0.0005,kl_min=0.30,item1='out_lik',seq1='discrete-discretized_logistic',item2='ignore_sigma_outcome',seq2='True-False',verbose=False &&




