# FlopPITy
normalizing **Flo**w exo**p**lanet **P**arameter **I**nference **T**oolk**y**t

FlopPITy allows the user to easily perform atmospheric retrievals using 
SNPE-C (citation) and neural spline flows (citation).

## Installation guide
Currently FlopPITy doesn't work with python 3.13

```bash
$ conda create -n floppity_env python==3.12.9
$ conda activate floppity_env
$ pip install floppity
```
     
## Basic usage:
- First, import FlopPITy:
```python
from floppity import Retrieval
from floppity.simulators import read_ARCiS_input, ARCiS
```

- Now you can initialize the retrieval class with a simulator.
  A python wrapper for [ARCiS](https://github.com/michielmin/ARCiS) comes built-in (you need to install ARCiS on your own tho):
  
```python
R = Retrieval(ARCiS)
```

- Read in observations and define parameters to retrieve:
    
```python
R.get_obs(['path/to/obs_0', 'path/to/obs_1',..., 'path/to/obs_n'])
    
R.add_parameter(par_0, min, max)
R.add_parameter(par_1, min, max)
...
R.add_parameter(par_m, min, max)
```

- For ARCiS, the observations and parameters can be read from the ARCiS input file:
    
```python
pars, obs_list = read_ARCiS_input('path/to/ARCiS/input')
R.get_obs(obs_list)
R.parameters=pars
```

- For retrievals using ARCiS, the input file and output directory need to be passed in a dictionary:
  
```python
ARCiS_kwargs= dict(
                    input_file = arcis_input,
                    output_dir = 'path/to/output',
                  )
```

- You can now run the retrieval, indicating the number of rounds and samples per round:

```python
R.run_retrieval(n_rounds=10, n_samples=1000, simulator_kwargs=ARCiS_kwargs)
```

- Great! You can now inspect your posterior:

```python
fig = R.plot_corner()
```

## Writing a simulator

Writing a simulator to work for FlopPITy is relatively straightforward. All that's needed is a function that takes in observations and 
parameters and returns spectra. The spectra need to be returned in a dictionary where each key represents each of the observations simulated (e.g. `simulated[0]` contains PRISM spectra and `simulated[1]` contains MIRI/LRS spectra):

```python
def simulator(obs, parameters, **kwargs):
    wvl_0 = obs[0][:,0]
    wvl_1 = obs[1][:,0]
    ...
    wvl_n = obs[n][:,0]

    spectra={}
    spectra[0] = # array of shape (ndims, len(wvl_0))
    spectra[1] = # array of shape (ndims, len(wvl_1))
    ...
    spectra[n] = # array of shape (ndims, len(wvl_n))

    return spectra
```

## Advanced options:

- Additional post processing parameters (currently `RV`, `vrot`, `offset` and `scaling`) can be added, for example:
    
```python
R.add_parameter('RV', -100, 100, post_process=True) # km/s
```




  

  
  

  

  
