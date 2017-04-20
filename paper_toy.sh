#!/bin/bash

echo "Start running simulations to reproduce paper results on toy domain"

python3 vae_main.py --hpconfig network=2,n_rep=10,deterministic=True,verbose=False &&
python3 vae_main.py --hpconfig network=2,n_rep=10,deterministic=False,z_size=3,verbose=False &&
python3 vae_main.py --hpconfig network=1,n_rep=10,var_type='discrete',K=3,N=3,verbose=False &&
python3 vae_main.py --hpconfig network=1,n_rep=10,var_type='continuous',z_size=3,n_flow=0,verbose=False &&
python3 vae_main.py --hpconfig network=1,n_rep=10,var_type='continuous',z_size=3,n_flow=5,verbose=False &&
