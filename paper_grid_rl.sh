#!/bin/bash

echo "Starting simulations on gridworld with RL agent"

python3 vae_grid.py --hpconfig artificial_data=False,use_target_net=True,test_on_policy=True,network=1,n_epochs=75000,n_rep=5,var_type='discrete',K=4,N=8,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=False,use_target_net=True,test_on_policy=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous',z_size=8,n_flow=0,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=False,use_target_net=True,test_on_policy=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous',z_size=8,n_flow=6,ar=False,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=False,use_target_net=True,test_on_policy=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous',z_size=8,n_flow=3,ar=True,verbose=False &&

