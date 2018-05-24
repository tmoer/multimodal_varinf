# Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning

Code for reproducing key results in the paper [Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning](http://thomasmoerland.nl/wp-content/uploads/2016/12/ecml_paper434.pdf) by Thomas M. Moerland, Joost Broekens and Catholijn M. Jonker. 

## Prerequisites
1. Install recent versions of:
- Python 3
- Tensorflow   
- Numpy (e.g. `pip install numpy`)
- Matplotlib

2. Clone this repository:
```sh
git clone https://github.com/tmoer/multimodal_varinf.git
```
## Syntax
Example:
```sh
python3 vae_main.py --logdir <logdir> --hpconfig network=1,n_rep=10,var_type='discrete',K=3,N=3,verbose=False
python3 vae_grid.py --logdir <logdir> --hpconfig network=1,n_epochs=75000,n_rep=5,var_type='continuous',z_size=8,n_flow=0,artificial_data=False,use_target_net=True,test_on_policy=True,verbose=False
```
For default hyper-parameters, look at the `get_hps()` function in the `vae_grid.py` and `vae_main.py` scripts. 

## Reproducing Paper Results
Run:
```sh
bash paper_toy.sh (Sec 4.1)
bash paper_grid.sh (Sec 4.2)
bash paper_grid_rl.sh (Sec 4.2)
``` 

## Citation
```
@proceedings{moerland2017learning,
	author = "Moerland, Thomas M. and Broekens, Joost and Jonker, Catholijn M.",
	note = "arXiv preprint arXiv:1705.00470",
	journal = "Scaling Up Reinforcement Learning (SURL) Workshop @ European Machine Learning Conference (ECML)",
	title = "{Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning}",
	year = "2017"
}
```

