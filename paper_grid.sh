#!/bin/bash

echo "Starting grid-world simulations on decorrelated data (randomly sampled transitions across state-space)"

python3 vae_grid.py --hpconfig artificial_data=True,network=1,n_epochs=75000,n_rep=5,var_type='discrete',K=4,N=8,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous',z_size=8,n_flow=0,verbose=False &&
python3 vae_grid.py --hpconfig artificial_data=True,network=1,n_epochs=75000,n_rep=5,var_type='continuous',z_size=8,n_flow=6,verbose=False &&





