This is a code to perform retrievals on spectra of substellar objects using 
SNPE-C (citation) and neural spline flows (citation).

- Known issues:
    - installing sbi needs blas first
        - conda install blas works, but goes down to python 3.9.22
     
# Basic usage:
- First, import FlopPITy:
```python
from floppity import Retrieval
from floppity.simulators import read_ARCiS_input, ARCiS
```

- Now you can initialize the retrieval class with a simulator. ARCiS comes built-in:
  
```python
R = Retrieval(ARCiS)
```

- Read in observations and define parameters to retrieve:
    
```python
R.get_obs(['path/to/obs1', 'path/to/obs2'])
    
R.add_parameter(par1, min, max)
R.add_parameter(par2, min, max)
...
R.add_parameter(parN, min, max)
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

## Advanced options:

- Additional post processing parameters (currently `RV`, `vrot`, `offset` and `scaling`) can be added, for example:
    
```python
R.add_parameter('RV', -100, 100, post_process=True) # km/s
```




  

  
  

  

  
