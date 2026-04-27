# FlopPITy

normalizing **Flo**w exo**p**lanet **P**arameter **I**nference **T**oolk**y**t

FlopPITy is a small Python package for atmospheric retrievals with
simulation-based inference. You provide observed spectra, define parameters,
and point FlopPITy at a simulator that returns model spectra. FlopPITy trains a
posterior with `sbi`.

This README is the shortest path to running a retrieval. For all options,
ARCiS details, binary/multi-component retrievals, output files, resuming,
plotting, PCA, and post-processing, see [docs/detailed_guide.md](docs/detailed_guide.md).

## Install

FlopPITy supports Python `>=3.10, <3.13`.

```bash
conda create -n floppity_env python=3.12.9
conda activate floppity_env
pip install floppity
```

## Observation Files

Each observation file is a plain text table with at least three columns:

```text
# wavelength    observed_value    uncertainty
```

## Minimal Retrieval

```python
import numpy as np
from floppity import Retrieval

def simulator(obs, parameters, thread=0, **kwargs):
    """Return model spectra keyed like obs.

    parameters has shape (n_samples, n_parameters).
    Each returned spectrum has shape (n_samples, n_wavelengths).
    """
    spectra = {}

    for key, obs_array in obs.items():
        wavelength = obs_array[:, 0]
        spectra[key] = # array of shape (n_samples, n_wavelengths)

    return spectra

R = Retrieval(simulator)

R.get_obs(["path/to/observation.txt"])

R.add_parameter("parameter_name", prior_min, prior_max)

R.run()
```

That is the core FlopPITy workflow:

1. Create a `Retrieval`.
2. Load observations with `get_obs`.
3. Add parameters with `add_parameter`.
4. Run with `run`.

## Multiple Observations

```python
R.get_obs({
    "prism": "path/to/prism.txt",
    "miri": "path/to/miri.txt",
})
```

Your simulator should return spectra with the same keys:

```python
return {
    "prism": prism_model,
    "miri": miri_model,
}
```

## Retrieval with ARCiS

FlopPITy comes with an ARCiS wrapper by default that can import the observations and parameters from an ARCiS input file. The workflow would then be:

```python
from floppity import Retrieval
from floppity.simulators import read_ARCiS_input, ARCiS

R = Retrieval(ARCiS)

ARCiS_kwargs = dict(
    input_file = 'path/to/ARCiS/input.txt',
    output_dir = 'path/to/ARCiS/output',
)

parameters, observations = read_ARCiS_input(ARCiS_kwargs[input_file])

R.get_obs(observations)
R.parameters = parameters

R.run(simulator_kwargs=ARCiS_kwargs)
```

## Inspecting Results

After `run()`, the trained posterior proposals are stored on the retrieval:

```python
posterior = R.proposals[-1]
samples = posterior.sample((1000,))
```

You can also save and load the retrieval checkpoint:

```python
R.save("retrieval.pkl")
R = Retrieval.load("retrieval.pkl")
```

`posterior_samples_round_X.txt` stores 1000 posterior samples from each round in
natural parameter units. If `save_data=True`, `rounds/round_XXX/training_data.npz`
stores the sampled parameters, simulated spectra, and per-sample metadata used
for that training round.

## Next Steps

- Full guide: [docs/detailed_guide.md](docs/detailed_guide.md)
- ARCiS example: [examples/ARCiS_retrieval.py](examples/ARCiS_retrieval.py)
- Binary ARCiS notebook: [examples/ARCiS_binary_retrieval.ipynb](examples/ARCiS_binary_retrieval.ipynb)
- Plotting notebook: [examples/plot_retrieval_results.ipynb](examples/plot_retrieval_results.ipynb)
