# FlopPITy

normalizing **Flo**w exo**p**lanet **P**arameter **I**nference **T**oolk**y**t

FlopPITy is a small Python package for atmospheric retrievals with
simulation-based inference. It wraps an arbitrary spectral simulator and trains
an SNPE-C posterior estimator from `sbi`, using neural spline flows by default.

The typical workflow is:

1. Load one or more observed spectra.
2. Define the parameters and prior bounds.
3. Run simulator batches over multiple SNPE rounds.
4. Inspect or sample the trained posterior.

FlopPITy includes a toy Gaussian-feature simulator for quick tests and a Python
wrapper for [ARCiS](https://github.com/michielmin/ARCiS). ARCiS itself is not
bundled; install and configure it separately.

## Installation

FlopPITy currently supports Python `>=3.10, <3.13`. Python 3.13 is not
supported.

```bash
conda create -n floppity_env python=3.12.9
conda activate floppity_env
pip install floppity
```

For local development from this repository:

```bash
git clone https://github.com/franciscoardevolmartinez/FlopPITy.git
cd FlopPITy
pip install -e .
```

Some built-in helpers use `scipy` and `tqdm`; install them if your environment
does not already provide them.

## Core Concepts

### Observations

Observation files are plain text files read with `numpy.loadtxt`. Each file must
contain at least three columns:

```text
wavelength    observed_value    uncertainty
```

Additional columns are allowed and are kept in the observation array, but the
retrieval loop uses:

- column 0: wavelength grid
- column 1: observed spectrum
- column 2: observational uncertainty

Load one or more observations with:

```python
R.get_obs(["path/to/obs_0.txt", "path/to/obs_1.txt"])
```

By default, observations are stored as `R.obs[0]`, `R.obs[1]`, and so on. You
can also give them explicit names:

```python
R.get_obs(
    ["path/to/prism.txt", "path/to/miri.txt"],
    obs_names=["prism", "miri"],
)
```

Or pass a mapping directly:

```python
R.get_obs({
    "prism": "path/to/prism.txt",
    "miri": "path/to/miri.txt",
})
```

The simulator must return spectra with matching keys. Named observations are
especially useful for observation-specific post-processing parameters like
offsets and scaling factors.

### Observation Type

Create a retrieval with:

```python
R = Retrieval(simulator, obs_type="trans")
```

`obs_type` can be:

- `"trans"` for transmission spectra
- `"emis"` for emission spectra

For emission retrievals, non-positive observed and simulated values are clipped
to `1e-12` before log-style preprocessing can encounter them.

### Parameters

Each retrieved parameter is sampled in the unit cube and converted to physical
values using the bounds you provide.

```python
R.add_parameter("temperature", 500, 2500)
R.add_parameter("log_h2o", -12, -1)
```

The full signature is:

```python
R.add_parameter(
    parname,
    min_value,
    max_value,
    log_scale=False,
    post_process=False,
    universal=True,
)
```

Current notes:

- `min_value` and `max_value` define the linear unit-cube transform.
- `log_scale` is stored as metadata. If you need log-space sampling today, pass
  log-space bounds yourself.
- `post_process=True` marks a parameter as belonging to a post-processing
  function rather than the simulator.
- `universal` is stored as metadata for multi-model workflows.

## Quick Start With the Toy Simulator

The bundled `mock_simulator` creates spectra from Gaussian absorption features.
It expects parameters grouped as centers, widths, and amplitudes. For one
Gaussian feature, define three parameters.

```python
import numpy as np

from floppity import Retrieval
from floppity.helpers import create_obs_file
from floppity.simulators import mock_simulator

# Build a tiny synthetic observation.
wvl = np.linspace(1.0, 2.0, 100)
flux = 1.0 - 0.1 * np.exp(-0.5 * ((wvl - 1.45) / 0.08) ** 2)
err = np.full_like(wvl, 0.01)
obs = create_obs_file(wvl, flux, err)
np.savetxt("obs_mock.txt", obs)

R = Retrieval(mock_simulator, obs_type="trans")
R.get_obs(["obs_mock.txt"])

R.add_parameter("center", 1.1, 1.9)
R.add_parameter("sigma", 0.03, 0.2)
R.add_parameter("amplitude", 0.01, 0.3)

flow_kwargs = {
    "flow": "nsf",
    "transforms": 5,
    "hidden": 50,
    "blocks": 2,
    "bins": 8,
}

training_kwargs = {
    "max_num_epochs": 50,
}

R.run(
    n_rounds=3,
    n_samples=500,
    flow_kwargs=flow_kwargs,
    training_kwargs=training_kwargs,
    output_dir="output_mock",
)

fig = R.plot_corner(n_samples=2000)
```

## ARCiS Workflow

FlopPITy can parse ARCiS input files for observation paths and fitted
parameters:

```python
from floppity import Retrieval
from floppity.simulators import ARCiS, read_ARCiS_input

arcis_input = "path/to/arcis_input.in"
pars, obs_dict = read_ARCiS_input(arcis_input)

R = Retrieval(ARCiS, obs_type="emis")
R.parameters = pars
R.get_obs(obs_dict)
```

Pass ARCiS runtime options through `simulator_kwargs`:

```python
ARCiS_kwargs = {
    "input_file": arcis_input,
    "output_dir": "output_ARCiS",
    "ARCiS_dir": "/usr/local/bin/ARCiS",
    "num_threads": "4",
    "save_atmosphere": True,
}

R.run(
    n_rounds=5,
    n_samples_init=256,
    n_samples=256,
    n_threads=2,
    simulator_kwargs=ARCiS_kwargs,
    output_dir="output_retrieval",
)
```

ARCiS wrapper notes:

- `input_file` defaults to `"arcis_input.in"`.
- `output_dir` defaults to `"./arcis_outputs"`.
- `ARCiS_dir` defaults to `"/usr/local/bin/ARCiS"`.
- `num_threads` sets `OMP_NUM_THREADS` for the ARCiS subprocess.
- `save_atmosphere` defaults to `True`. When enabled, FlopPITy gathers every
  model's `mixingratios.dat` into one file per retrieval round.
- `atmosphere_file` defaults to `"mixingratios.dat"`.
- `atmosphere_output` can override the output filename. By default the wrapper
  writes `mixingratios_round_<round>.dat` in the ARCiS output directory.
- The wrapper forces or adds `makeai=.true.` in a copied input file.
- Temporary ARCiS model directories are removed after spectra are read.
- ARCiS logs are written inside the ARCiS output directory.

The combined atmosphere file contains one block per simulated model, prefixed
by a comment with the retrieval round, global model index, thread index, local
model index, and parameter vector. This is designed to preserve atmospheric
structures even though the temporary ARCiS model directories are cleaned up.

See `examples/ARCiS_retrieval.py` for a complete script-style example.

### Binary Or Multi-Component Models

FlopPITy itself does not need special binary-retrieval logic. It samples
whatever parameters you define, converts them to physical values, and passes
the resulting parameter matrix to the simulator. A binary retrieval therefore
works by defining one block of parameters per component and using a simulator
that splits those blocks and combines the spectra.

Use `make_binary_simulator` to wrap any simulator that follows the FlopPITy
simulator contract:

```python
from floppity import Retrieval
from floppity.simulators import ARCiS, make_binary_simulator, read_ARCiS_input

arcis_input = "path/to/arcis_input.in"
base_parameters, obs_dict = read_ARCiS_input(arcis_input)

binary_simulator, binary_parameters = make_binary_simulator(
    ARCiS,
    base_parameters,
    shared_parameters=["log_h2o", "log_ch4"],
    weight_parameters={"column_fraction": (0, 1)},
)

R = Retrieval(binary_simulator, obs_type="emis")
R.parameters = binary_parameters
R.get_obs(obs_dict)

R.run(
    simulator_kwargs=ARCiS_kwargs,
    n_rounds=5,
    n_samples=256,
)
```

In this example, `log_h2o` and `log_ch4` are sampled once and reused for both
components, while all other parameters are duplicated with `_1` and `_2`
suffixes. This lets you keep shared chemistry while allowing parameters such as
temperature, radius, or gravity to differ between components. The
`column_fraction` parameter is also sampled once; for two components it combines
the spectra as `column_fraction * component_1 + (1 - column_fraction) *
component_2`.

The wrapper expects the underlying simulator to accept a single-component
parameter matrix and return spectra keyed like the observations. It calls that
simulator once per component and combines the returned spectra.

Combination options:

- Direct sum, useful for additive binary fluxes:
  `make_binary_simulator(ARCiS, base_parameters, combine="sum")`.
- Fixed weights, useful when the ratio is known:
  `make_binary_simulator(ARCiS, base_parameters, component_weights=[0.3, 0.7])`.
- Sampled binary fraction, useful for two atmospheric columns:
  `make_binary_simulator(..., weight_parameters={"column_fraction": (0, 1)})`.
- Sampled weights for `N` components:
  `make_multi_component_simulator(..., n_components=N, weight_parameters={...})`.

Weights are normalized by default. For example, sampled weights `[1, 1, 2]`
are applied as `[0.25, 0.25, 0.5]`. Pass `normalize_weights=False` if you want
absolute multiplicative weights.

For more than two components, use `make_multi_component_simulator` and pass
`n_components=N`.

`ARCiS_binary` and `ARCiS_multiple` are still available for older scripts, but
new code should prefer the generic wrappers.

## Running Retrievals

The main entrypoint is:

```python
R.run(
    n_threads=1,
    n_samples=100,
    n_samples_init=None,
    resume=False,
    n_rounds=10,
    n_aug=1,
    flow_kwargs=None,
    training_kwargs=None,
    simulator_kwargs=None,
    output_dir="output_FlopPITy",
    save_data=False,
    sample_prior_method="sobol",
    reuse_prior=None,
    alpha=0,
    pca_components=None,
)
```

Important options:

- `n_rounds`: number of SNPE rounds to run in this call.
- `n_samples_init`: samples for the first round. Defaults to `n_samples`.
- `n_samples`: samples for later rounds.
- `n_threads`: number of Python worker processes for simulator calls.
- `n_aug`: repeats each simulated spectrum with fresh observational noise.
- `flow_kwargs`: arguments forwarded to `density_builder`.
- `training_kwargs`: arguments forwarded to `sbi` training.
- `simulator_kwargs`: arguments forwarded to your simulator.
- `output_dir`: checkpoint, setup log, and optional data output directory.
- `save_data`: if `True`, writes per-round compressed NumPy archives at
  `rounds/round_<NNN>/training_data.npz`.
- `sample_prior_method`: one of `"sobol"`, `"lhs"`, or `"random"` for the
  initial round.
- `reuse_prior`: path to previous saved training data to reuse initial prior
  simulations. New outputs use `rounds/round_<NNN>/training_data.npz`; legacy
  `data_<round>.pkl` files are still readable.
- `alpha`: posterior inflation fraction. If `alpha > 0`, later rounds sample
  an `alpha` fraction of parameters from the prior and `1 - alpha` from the
  latest uninflated posterior proposal.
- `pca_components`: optional number of PCA components to train on the
  preprocessed spectra before they are passed to the neural density estimator.
- `resume`: continue training from an already trained and loaded retrieval.

### Sampling Methods

The initial round can be sampled with:

- `"sobol"`: low-discrepancy Sobol samples. The sample count is rounded to the
  nearest power of two because Sobol sequences require it.
- `"lhs"`: Latin hypercube sampling.
- `"random"`: samples from the unit-cube prior.

Later rounds always sample from the latest posterior proposal.

### Flow Options

`flow_kwargs` are passed into `sbi.neural_nets.posterior_nn`. Common options:

```python
flow_kwargs = {
    "flow": "nsf",
    "transforms": 10,
    "hidden": 50,
    "blocks": 3,
    "bins": 8,
    "dropout": 0.05,
    "z_score_theta": "independent",
    "z_score_x": "independent",
    "use_batch_norm": True,
}
```

### Training Options

`training_kwargs` are passed to `sbi`'s `.train(...)` method. For example:

```python
training_kwargs = {
    "max_num_epochs": 100,
    "stop_after_epochs": 20,
    "learning_rate": 1e-3,
    "training_batch_size": 128,
}
```

The accepted keys depend on your installed `sbi` version.

## Saving, Loading, and Resuming

Retrieval objects can be saved manually:

```python
R.save("retrieval.pkl")
```

And loaded with:

```python
from floppity import Retrieval

R = Retrieval.load("retrieval.pkl")
```

`run(...)` also writes checkpoints:

- `retrieval_setup.json`: JSON summary of the run setup, including observation
  sources and shapes, parameter metadata, preprocessing, simulator name,
  `flow_kwargs`, `training_kwargs`, and `simulator_kwargs`.
- `retrieval_pre_round_<N>.pkl`: state after generating data for round `N`, but
  before training that round.
- `retrieval.pkl`: state after each successfully completed training round.
- `rounds/round_<NNN>/training_data.npz`: optional per-round training arrays
  written when `save_data=True`. These archives store the sampled unit-cube
  parameters, natural parameters when available, raw simulator spectra,
  post-processed spectra, and per-sample source labels (`prior` or `proposal`)
  while preserving observation keys such as `obs1` or `miri`.

Pickle is still used for retrieval checkpoints because those contain trained
Python and `sbi` objects. Array-heavy training data is written as `.npz`, which
is smaller, easier to inspect with NumPy, and less coupled to Python object
internals.

The output paths and training-data serialization are centralized in
`floppity.RetrievalOutput`. You usually do not need to instantiate it directly,
but it is useful for inspecting saved round data:

```python
from floppity import RetrievalOutput

data = RetrievalOutput.load_round_data(
    "output_FlopPITy/rounds/round_000/training_data.npz"
)
theta = data["par"]
spectra = data["spec"]
retrieved_spectra = data.get("post_spec", spectra)
sample_sources = data.get("sample_sources")
```

Resume from a completed checkpoint with:

```python
R = Retrieval.load("output_FlopPITy/retrieval.pkl")
R.run(
    resume=True,
    n_rounds=3,
    n_samples=500,
    output_dir="output_FlopPITy",
)
```

When `resume=True`, FlopPITy samples from the last stored posterior proposal.
It does not repeat the initial prior-sampling round.

## Preprocessing

Set `R.preprocessing` to a list of preprocessing function names before running:

```python
R.preprocessing = ["log", "standardize_1v1"]
```

Available preprocessing functions:

- `softclip`: soft-clips values with `x / (1 + abs(x / 100))`.
- `log`: applies `log10`; all values must be positive.
- `standardize_1v1`: standardizes each spectrum independently and appends that
  row's mean and standard deviation.
- `standardize_global`: column-wise standardization.

Preprocessing is applied to simulated training spectra and the default
observation used for posterior conditioning.

### PCA Compression

Large spectra can be compressed before training by passing `pca_components` to
`run(...)`:

```python
R.preprocessing = ["log"]
R.run(
    n_rounds=5,
    n_samples=512,
    pca_components=50,
)
```

PCA is fit once on the first generated training batch after the normal
preprocessing chain has been applied. The fitted transform is stored on the
retrieval object and reused for every later round, for posterior conditioning,
and when resuming from `retrieval.pkl`. This keeps the feature space fixed
across the whole retrieval.

Notes:

- `pca_components=None` disables PCA.
- If the requested component count is larger than the available rank, FlopPITy
  uses the largest possible count and reports that adjustment.
- Older code that passes `n_pca=...` to `run(...)` is still accepted as an alias
  for `pca_components`.
- Older code that creates `Retrieval(..., do_pca=True)` is still accepted, but
  new code should prefer `pca_components` because it makes the component count
  explicit.

## Post-Processing Parameters

Post-processing parameters are parameters that transform simulator output before
noise is added and before training. Add them with `post_process=True`.

```python
R.add_parameter("RV", -100, 100, post_process=True)
R.add_parameter("vrot", 0, 50, post_process=True)
R.add_parameter("wvl_offset", -0.01, 0.01, post_process=True)
```

Supported function names correspond to functions in `floppity.postprocessing`:

- `RV`: radial velocity Doppler shift in km/s.
- `vrot`: rotational broadening.
- `wvl_offset`: additive wavelength offset.
- `instrumental_broadening`: Gaussian broadening.
- `offset:<obs_key>`: additive flux offset applied only to one observation.
- `scaling:<obs_key>`: multiplicative flux scaling applied only to one
  observation.

Examples for observation-specific parameters:

```python
R.add_parameter("offset:prism", -0.01, 0.01, post_process=True)
R.add_parameter("scaling:miri", 0.8, 1.2, post_process=True)
```

In those examples, `offset:prism` applies to `R.obs["prism"]` and
`scaling:miri` applies to `R.obs["miri"]`.

The underscore form also works, for example `offset_prism`. For older
integer-key workflows, names like `offset1` and `scaling2` still refer to
`R.obs[1]` and `R.obs[2]`.

## Writing a Simulator

A simulator is a callable with this shape:

```python
def simulator(obs, parameters, thread=0, **kwargs):
    ...
    return spectra
```

Inputs:

- `obs`: dictionary of observation arrays, keyed `0`, `1`, ...
- `parameters`: NumPy array of natural parameter values with shape
  `(n_samples, n_simulator_parameters)`.
- `thread`: integer worker index, useful for unique output files in parallel
  runs.
- `**kwargs`: any simulator-specific options passed through `simulator_kwargs`.

Output:

- dictionary with the same keys as `obs`.
- each value must have shape `(n_samples, n_wavelengths_for_that_obs)`.

Minimal example:

```python
def flat_simulator(obs, parameters, thread=0, **kwargs):
    spectra = {}
    level = parameters[:, 0]

    for key, obs_array in obs.items():
        n_wavelengths = len(obs_array[:, 0])
        spectra[key] = np.repeat(level[:, None], n_wavelengths, axis=1)

    return spectra
```

Then:

```python
R = Retrieval(flat_simulator, obs_type="trans")
R.get_obs(["obs.txt"])
R.add_parameter("level", 0.95, 1.05)
R.run(n_rounds=2, n_samples=200)
```

## Inspecting Results

Draw a corner plot from any stored proposal:

```python
fig = R.plot_corner(proposal_id=-1, n_samples=5000)
```

Draw posterior samples directly:

```python
samples_unit_cube = R.proposals[-1].sample((5000,))
samples_physical = helpers.convert_cube(
    samples_unit_cube.detach().numpy(),
    R.parameters,
)
```

Find a MAP estimate from a proposal:

```python
from floppity.helpers import find_MAP

map_estimate = find_MAP(R.proposals[-1])
```

## Helper Functions

`floppity.helpers` includes:

- `create_obs_file(wvl, spectrum, error, *extras)`: create an observation array.
- `convert_cube(thetas, pars)`: convert unit-cube samples to physical values.
- `compute_moments(distribution)`: estimate moments from a PyTorch distribution.
- `find_MAP(proposal)`: call `proposal.map(...)` with FlopPITy's defaults.
- `reduced_chi_squared(obs_dict, sim_dict, n_params=0)`: compute reduced
  chi-squared values.
- `find_best_fit(obs_dict, sim_dict)`: find the lowest chi-squared simulation.

## Project Layout

```text
src/floppity/flappity.py       Retrieval class and training loop
src/floppity/simulators.py     mock simulator and ARCiS wrapper
src/floppity/postprocessing.py spectral post-processing functions
src/floppity/preprocessing.py  preprocessing functions
src/floppity/helpers.py        small utilities
examples/                      notebooks and script examples
tests/                         unit tests
```

## Development Checks

From a local checkout:

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m py_compile src/floppity/flappity.py src/floppity/helpers.py
```

If imports fail inside `sbi`, check that your `torch`, `pyro`, and `sbi`
versions are compatible with each other and with the supported Python range.

## Current Limitations

- Python 3.13 is not supported.
- ARCiS must be installed separately.
- `log_scale` is metadata; pass log-space bounds directly when needed.
- Simulator outputs must already be sampled on the observation wavelength grids.
- Resuming depends on pickled `sbi` objects, so it is safest to resume in a
  compatible Python/package environment.
