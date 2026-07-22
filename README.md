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

To train SBI on residuals instead of the simulated spectra, construct the
retrieval with `fit_residuals=True`. FlopPITy then uses
`simulation - observation` as the training feature and conditions the posterior
on a zero residual:

```python
R = Retrieval(simulator, fit_residuals=True)
R.preprocessing = []
```

Residuals are signed, so this mode cannot be combined with `log` or
`log_standardize` preprocessing. Use no preprocessing or a transformation that
supports signed values.

For positive emission spectra spanning a large dynamic range, log residuals
are usually preferable:

```python
R = Retrieval(simulator, obs_type="emis", fit_residuals=True)
R.preprocessing = ["log_residual"]
```

This trains on `log10(simulation) - log10(observation)` and conditions on zero.
Both arrays are clipped to `emission_flux_floor` before taking the logarithm.

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

parameters, observations = read_ARCiS_input(ARCiS_kwargs['input_file'])

R.get_obs(observations)
R.parameters = parameters

R.run(simulator_kwargs=ARCiS_kwargs)
```


## Retrieval with PICASO

FlopPITy also comes with a PICASO wrapper that can import the observations and
parameters from a PICASO retrieval TOML. The workflow mirrors the ARCiS wrapper:

```python
from floppity import Retrieval
from floppity.simulators import read_PICASO_input, PICASO

R = Retrieval(PICASO, obs_type='emis')
R.preprocessing = ['log']

PICASO_kwargs = dict(
    input_file='path/to/PICASO/input.toml',
    picaso_repo='path/to/PICASO',  # optional if picaso is importable
    # pysyn_cdbs='path/to/trds',   # optional if PICASO needs synphot refs
)

parameters, observations = read_PICASO_input(**PICASO_kwargs)

R.get_obs(observations)
R.parameters = parameters

R.run(simulator_kwargs=PICASO_kwargs)
```

For PICASO simulation fitting, plain `log` preprocessing is recommended over
the default `log_standardize`. For PICASO residual fitting, use
`fit_residuals=True` with `R.preprocessing = ["log_residual"]`. Raw residuals
remain available with `R.preprocessing = []`.

PICASO is imported lazily, so it is not a required FlopPITy dependency unless
you use this simulator. `read_PICASO_input` writes temporary FlopPITy
observation text files from PICASO's `[ObservationData]` block. By default those
files are written under `[InputOutput].retrieval_output` in the TOML.

The FlopPITy observations use the data-file errors directly. PICASO
`err_inf.*` priors are ignored because they are likelihood nuisance terms, not
part of the simulator training data. PICASO `scaling.*` and `offset.*` priors
are converted to FlopPITy post-processing parameters like `scaling:prism`.

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

If `save_posterior_samples=True`, `posterior_samples_round_X.txt` stores 1000
posterior samples from each round in natural parameter units. If `save_data=True`,
`rounds/round_XXX/training_data.npz`
stores the sampled parameters, simulated spectra, and per-sample metadata used
for that training round. It also writes `rounds/round_XXX/sbi_data.npz` with the
exact normalized `theta`, `x`, and `default_x` arrays passed to SBI.

For stochastic checks, `R.run_ensemble(...)` repeats the same retrieval,
reuses member 1's prior simulations, and writes aggregated samples/data under
an `aggregated/` folder. Use `resume=True, add_members=True` to append more
members, or `resume=True, extend_rounds=True` to continue every existing member
for more rounds.

## Basic troubleshooting

 - It takes a very long time to sample from the posterior. "Only xxx% of samples are accepted, consider changing to MCMC": This probably means that the samples are too dissimilar to the observation. Usual suspects: incorrectly setup model, incorrect priors, or mismatching wavelength axes.

## Next Steps

- Full guide: [docs/detailed_guide.md](docs/detailed_guide.md)
- ARCiS example: [examples/ARCiS_retrieval.ipynb](examples/ARCiS_retrieval.ipynb)
- Binary ARCiS notebook: [examples/ARCiS_binary_retrieval.ipynb](examples/ARCiS_binary_retrieval.ipynb)
- Plotting notebook: [examples/plot_retrieval_results.ipynb](examples/plot_retrieval_results.ipynb)
