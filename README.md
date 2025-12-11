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
```

- Now you can initialize the retrieval class with a simulator. A simulator is a function that takes in parameters and returns spectra, look below to see specifically how it needs to be written.
  Functionality for [ARCiS](https://github.com/michielmin/ARCiS) and [PICASO](https://natashabatalha.github.io/picaso/) comes built-in (you need to install them separately). Look further down for examples.
  
```python
R = Retrieval(your_simulator_function)
```

- Read in observations and define parameters to retrieve:
    
```python
R.get_obs({obs_0:'path/to/obs_0', obs_1:'path/to/obs_1',..., obs_n:np.array(shape=[n_wvl,>3])})
    
R.add_parameter(par_0, min, max)
R.add_parameter(par_1, min, max)
...
R.add_parameter(par_m, min, max)
```

- You can now run the retrieval, indicating the number of rounds and samples per round:

```python
R.run(n_rounds=10, n_samples=1000, simulator_kwargs=simulator_kwargs)
```

- Great! You can now inspect your posterior:

```python
fig = R.plot_corner()
```

## ARCiS example:

- Firstly, initialize your retrieval object:

```python
from floppity import Retrieval
from floppity.simulators import read_ARCiS_input, ARCiS

R = Retrieval(ARCiS)
```

- For ARCiS, the observations and parameters can be read from the ARCiS input file:
    
```python
pars, obs_list = read_ARCiS_input('path/to/ARCiS/input')
R.get_obs(obs_list)
R.parameters=pars
```

- The input file and output directory need to be passed in a dictionary:
  
```python
ARCiS_kwargs= dict(
                    ARCiS_dir = "/path/to/ARCiS/executable", #only needs to be set if ARCiS is not on the default path
                    input_file = 'path/to/ARCiS/input',
                    output_dir = 'path/to/output',
                  )
```

- You can now run the retrieval as usual:

```python
R.run(n_rounds=10, n_samples=1000, simulator_kwargs=ARCiS_kwargs)
```

## PICASO example:

- Running a retrieval with PICASO is very similar (this only works with the gridtree branch):
    
```python
from floppity import Retrieval
from floppity.simulators import read_PICASO_config, PICASO

R = Retrieval(PICASO)

pars, obs_list = read_PICASO_config('path/to/config.toml')
R.get_obs(obs_list)
R.parameters=pars
```

- The configuration file needs to be passed as a kwarg:
  
```python
PICASO_kwargs= dict(
                    config_file = 'path/to/config.toml'
                  )
```

- You can now run the retrieval as usual:

```python
R.run(n_rounds=10, n_samples=1000, simulator_kwargs=PICASO_kwargs)
```


## Writing a simulator

Writing a simulator to work for FlopPITy is relatively straightforward. All that's needed is a function that takes in observations and 
parameters and returns spectra. The spectra need to be returned in a dictionary where each key represents each of the observations simulated (e.g. `simulated['prism']` contains PRISM spectra and `simulated['lrs']` contains MIRI/LRS spectra):

```python
def simulator(obs, parameters, **kwargs):
    wvl_prism = obs['prism'][:,0]
    wvl_lrs = obs['lrs'][:,0]
    ...
    wvl_n = obs[n][:,0]

    spectra={}
    spectra['prism'] = # array of shape (ndims, len(wvl_prism))
    spectra['lrs'] = # array of shape (ndims, len(wvl_lrs))
    ...
    spectra[n] = # array of shape (ndims, len(wvl_n))

    return spectra
```

## Advanced options:

- Additional post processing parameters (currently `RV`, `vrot`, `offset` and `scaling`) can be added, for example:
    
```python
R.add_parameter('RV', -100, 100, post_process=True) # km/s
```
- For offsets and scalings between different observations, the parameters should be named 'offset_{observation_key}'. For example, if we wanted to fit for a scaling factor between 0.95 and 1.05:

```python
R.add_parameter('scaling_obs2', 0.95, 1.05, post_process=True)




  

  
  

  

  
