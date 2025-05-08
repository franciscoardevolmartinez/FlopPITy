from floppity import Retrieval, helpers
import numpy as np
import torch
import pickle
from tqdm import trange
from floppity.simulators import read_ARCiS_input, ARCiS

if __name__ == "__main__":
    arcis_input='/Users/floppityflappity/input_j1828_free.dat'
    pars, obs_list = read_ARCiS_input(arcis_input)

    ARCiS_kwargs= dict(
        input_file = arcis_input,
        output_dir = '/Users/floppityflappity/j1828_free_test/',
    )

    training_kwargs= dict(
        stop_after_epochs = 10,
        num_atoms = 20,
        learning_rate=1e-3
    )

    flow_kwargs=dict(
        flow='nsf',
        bins=4,
        transforms=15,
        blocks=4,
        hidden=64,
        dropout=0.5
    )

    R = Retrieval(ARCiS)

    R.parameters=pars
    R.get_obs(obs_list)

    R.run_retrieval(flow_kwargs=flow_kwargs, resume=False, n_threads=2, 
                    training_kwargs=training_kwargs, simulator_kwargs=ARCiS_kwargs, 
                    n_rounds=2, n_samples_init=10, n_samples=10
                    )
    
    R.save(ARCiS_kwargs['output_dir']+'retrieval.pkl')