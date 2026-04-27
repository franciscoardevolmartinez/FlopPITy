import numpy as np
import os
import time
from tqdm import trange, tqdm
import subprocess
import fcntl
import shutil
from copy import deepcopy

def mock_simulator(obs, pars, thread=0):
    '''
    Simulate observations using Gaussian features.
        obs (dict): Dictionary of observations with wavelengths.
        pars (np.ndarray): Parameters for Gaussian features.
                          Columns: centers, sigmas, amplitudes.
        thread (int, optional): Thread index for parallelism. Defaults to 0.
        dict: Simulated data for each observation key.
    '''
    def gaussian(x, centers, sigmas, amplitudes):
        """Add multiple Gaussian features.

        Args:
            x (np.ndarray): Input x-axis (e.g., wavelengths).
            centers (list): List of Gaussian centers.
            sigmas (list): List of Gaussian standard deviations.
            amplitudes (list): List of Gaussian amplitudes.

        Returns:
            np.ndarray: Sum of all Gaussian features evaluated at x.
        """
        y = np.ones_like(x)
        for c, s, a in zip(centers, sigmas, amplitudes):
            y -= a * np.exp(-0.5 * ((x - c) / (s)) ** 2)
        return y

    ndims=pars.shape[1]

    c=pars[:,0:ndims//3]
    s=pars[:,ndims//3:2*ndims//3]
    a=pars[:,2*ndims//3:3*ndims//3]
    
    x = {}
    for key in obs.keys():
        wvl = obs[key][:,0]
        x[key] = np.empty([len(c), len(wvl)])
        for i in range(len(c)):
            x[key][i]=gaussian(wvl, c[i], s[i], a[i])
    return x

def read_ARCiS_input(input_path):
    """
    Parses an ARCiS input file and extracts model parameter specifications
    and observation file paths.

    This function reads an ARCiS configuration file, identifying the free
    parameters to be fitted along with their prior bounds, whether to use
    logarithmic scaling, and observation file paths.

    Parameters
    ----------
    input_path : str
        Path to the ARCiS input text file.

    Returns
    -------
    par_dict : dict
        A dictionary where each key is a parameter name, and the value is
        another dictionary with:
            - 'min': float, lower bound of the parameter (log-scaled if
              applicable)
            - 'max': float, upper bound of the parameter (log-scaled if
              applicable)
            - 'log': bool, whether the parameter is treated in log space
            - 'post_processing': bool, currently always False (can be
              toggled later if needed)
            - 'global': bool, currently always False (for global parameter
              flagging)

    obs_dict : dict
        A dictionary of observation names and file paths specified in the
        ARCiS input. Keys are named like ``obs1``, ``obs2``, and so on.
    """
    with open(input_path, 'rb') as arcis_input:
        lines = arcis_input.readlines()

    # Decode lines and remove comments/blank lines
    clean_in = []
    for line in lines:
        line = line.decode().strip()
        if line and not line.startswith('*'):
            clean_in.append(line)

    par_dict = {}
    i = 0

    # Extract parameter definitions
    while i < len(clean_in):
        if 'fitpar:keyword' in clean_in[i]:
            param_name = clean_in[i].split('=')[-1].strip().strip('"')

            # Extract and convert prior bounds
            lower_raw = clean_in[i+1].split('=')[-1].replace('d', 'e')
            upper_raw = clean_in[i+2].split('=')[-1].replace('d', 'e')
            lower, upper = float(lower_raw), float(upper_raw)

            # Determine whether parameter is in log space
            log_flag = clean_in[i+3].strip() == 'fitpar:log=.true.'
            if log_flag:
                lower, upper = np.log10(lower), np.log10(upper)

            # Store parameter metadata
            par_dict[param_name] = {
                'min': lower,
                'max': upper,
                'log': log_flag,
                'post_processing': False,
                'universal': False
            }

            i += 4  # Advance past this parameter block
        else:
            i += 1

    # Extract observation file paths
    obs_dict = {}
    obsn = 1
    for line in clean_in:
        key = f'obs{obsn}:file'
        if key in line:
            path = line.split('=')[-1].strip().strip('"')
            obs_dict[f'obs{obsn}'] = path
            obsn += 1

    return par_dict, obs_dict

def ARCiS(obs, parameters, thread=0, **kwargs):
    """
    Run ARCiS simulations and return model spectra for each observation.

    Parameters
    ----------
    obs : dict
        Dictionary where each key is an integer (0, 1, ...) and each value 
        is a 2D numpy array with at least three columns: wavelength, 
        observed spectrum, and error.
    parameters : np.ndarray
        Array of shape (n_spectra, n_parameters), where each row is a 
        parameter vector.
    thread : int
        Integer passed to keep track of multiple parallel threads.
    **kwargs : dict
        Optional keyword arguments:
            - input_file (str): Path to ARCiS input file.
            - output_dir (str): Directory where outputs are written.
            - ARCiS_dir (str): The location where ARCiS is installed.
            If it's not passed, it's assumed to be \'/usr/local/bin/ARCiS\'.
            - save_atmosphere (bool): If True, append each model's
              mixingratios.dat to one round-level output file.
            - atmosphere_file (str): Atmospheric structure file to collect
              from each ARCiS model directory. Defaults to mixingratios.dat.
            - atmosphere_output (str): Combined atmosphere filename. Defaults
              to mixingratios_round_<round>.dat.
            - log_dir (str): Directory for ARCiS logs. Relative paths are
              created inside output_dir. Defaults to arcis_logs.

    Returns
    -------
    spectra : dict
        Dictionary where each key matches obs (0, 1, ...) and each value 
        is an array of shape (n_spectra, n_points) containing the modeled 
        spectra for each observation.
    """
    input_file = kwargs.get('input_file', 'arcis_input.in')
    output_dir = kwargs.get('output_dir', './arcis_outputs')
    ARCiS = kwargs.get('ARCiS_dir', '/usr/local/bin/ARCiS')
    verbose = kwargs.get('verbose', True)
    num_threads = kwargs.get('num_threads', "4")
    round_index = kwargs.get('_round_index', 0)
    sample_offset = kwargs.get('_sample_offset', 0)
    save_atmosphere = kwargs.get('save_atmosphere', True)
    atmosphere_file = kwargs.get('atmosphere_file', 'mixingratios.dat')
    atmosphere_output = kwargs.get(
        'atmosphere_output',
        f'mixingratios_round_{round_index}.dat'
    )
    log_dir = _arcis_log_dir(output_dir, kwargs.get('log_dir', 'arcis_logs'))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Copy and modify input file
    input_copy = os.path.join(output_dir, os.path.basename(input_file))
    with open(input_file, 'r') as f:
        lines = f.readlines()

    found_makeai = False
    for i, line in enumerate(lines):
        if 'makeai=' in line.lower():
            found_makeai = True
            if '.false.' in line.lower():
                print('Warning: Found "makeai=.false." — changing to "makeai=.true."')
                lines[i] = 'makeai=.true.\n'
            break

    if not found_makeai:
        print('Warning: No "makeai=" found — adding "makeai=.true."')
        lines.append('makeai=.true.\n')

    with open(input_copy, 'w') as f:
        f.writelines(lines)

    output_base = os.path.join(output_dir, f'outputARCiS_{thread}')

    n_spectra = parameters.shape[0]
    obs_indices = list(obs.keys())
    obs_files = [_arcis_obs_file_name(key) for key in obs_indices]

    # Write parameter grid file
    param_file = os.path.join(output_dir, f'parametergridfile_{thread}.dat')
    np.savetxt(param_file, parameters)

    # Run ARCiS
    log_files = [f for f in os.listdir(log_dir) if f.startswith(f'arcis_run_{thread}_') and f.endswith('.log')]
    log_nums = []
    for f in log_files:
        try:
            num = int(f[len(f'arcis_run_{thread}_'):-len('.log')])
            log_nums.append(num)
        except ValueError:
            continue
    next_log_num = max(log_nums) + 1 if log_nums else 1
    log_file = os.path.join(log_dir, f'arcis_run_{thread}_{next_log_num}.log')

    with open(log_file, 'w') as log:
        try:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = num_threads
            subprocess.run(
                [ARCiS, input_copy, "-o", output_base, "-s", f"parametergridfile={param_file}"],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            # if verbose:
            #     check_ARCiS_status(proc, output_base, n_spectra, thread)
            print(f'ARCiS finished successfully. Output logged to: {log_file}')
        except subprocess.CalledProcessError:
            print(f'ARCiS failed. Check log for details: {log_file}')
            raise

    # Initialize spectra output
    spectra = {k: [] for k in obs_indices}

    print('Reading ARCiS output...')
    for i in trange(n_spectra):
        model_dir = f'{output_base}/model{i+1:06d}/'
        global_model_index = sample_offset + i

        for k, obs_file in zip(obs_indices, obs_files):
            try:
                phase = np.loadtxt(os.path.join(model_dir, obs_file))[:, 1]
            except Exception as e:
                print(f'Warning: Could not read {obs_file} in {model_dir}: {e}')
                phase = -1 * np.ones_like(obs[k][:, 1])  # fallback
            spectra[k].append(phase)

        if save_atmosphere:
            _append_arcis_atmosphere_structure(
                model_dir=model_dir,
                atmosphere_file=atmosphere_file,
                output_path=os.path.join(output_dir, atmosphere_output),
                round_index=round_index,
                thread=thread,
                local_model_index=i,
                global_model_index=global_model_index,
                parameters=parameters[i],
            )

    for k in spectra:
        spectra[k] = np.array(spectra[k])  # shape: (n_spectra, n_points)

    # Remove temporary ARCiS model output.
    print('Removing files...')
    shutil.rmtree(output_base, ignore_errors=True)

    return spectra


def _arcis_log_dir(output_dir, log_dir):
    """Return the directory where ARCiS subprocess logs should be written."""
    if os.path.isabs(log_dir):
        return log_dir
    return os.path.join(output_dir, log_dir)


def ARCiS_binary(obs, parameters, thread=0, **kwargs):
    """Run a two-component ARCiS model and sum the component spectra."""
    kwargs = dict(kwargs)
    kwargs["n_components"] = 2
    return ARCiS_multiple(obs, parameters, thread=thread, **kwargs)


def ARCiS_multiple(obs, parameters, thread=0, **kwargs):
    """Run and sum multiple ARCiS components.

    Parameters are expected to be stacked component-by-component. For a binary
    retrieval with ``m`` parameters per object, the first ``m`` columns belong
    to component 1 and the next ``m`` columns belong to component 2.
    """
    n_components = int(kwargs.get("n_components", 2))
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    parameters = np.asarray(parameters)
    if parameters.ndim != 2:
        raise ValueError("parameters must be a 2D array.")

    n_samples, n_total_params = parameters.shape
    if n_total_params % n_components != 0:
        raise ValueError(
            f"Parameter count {n_total_params} is not divisible by "
            f"n_components={n_components}."
        )

    n_component_params = n_total_params // n_components
    component_blocks = parameters.reshape(n_samples, n_components, n_component_params)
    component_order = np.argsort(-component_blocks[:, :, 0], axis=1)
    component_blocks = np.take_along_axis(
        component_blocks,
        component_order[:, :, None],
        axis=1,
    )
    combined = None

    for component_index in range(n_components):
        print(f"Computing ARCiS component {component_index + 1}/{n_components}")
        component_parameters = component_blocks[:, component_index, :]

        component_kwargs = dict(kwargs)
        component_thread = f"{thread}_component{component_index + 1}"
        component_spectra = ARCiS(
            obs,
            component_parameters,
            thread=component_thread,
            **component_kwargs,
        )

        if combined is None:
            combined = {
                key: np.array(value, copy=True)
                for key, value in component_spectra.items()
            }
        else:
            for key, value in component_spectra.items():
                combined[key] += value

    return combined


class MultiComponentSimulator:
    """Wrap any FlopPITy simulator as a multi-component simulator."""

    def __init__(
        self,
        simulator,
        parameter_names,
        n_components=2,
        shared_parameters=None,
        combine="sum",
        component_weights=None,
        weight_parameter_names=None,
        normalize_weights=True,
    ):
        self.simulator = simulator
        self.parameter_names = list(parameter_names)
        self.n_components = int(n_components)
        self.shared_parameters = set(shared_parameters or [])
        self.combine = combine
        self.component_weights = component_weights
        self.weight_parameter_names = list(weight_parameter_names or [])
        self.normalize_weights = normalize_weights
        self.__name__ = f"{_callable_name(simulator)}_multi_component"
        self.sort_parameter_name = self._sort_parameter_name()
        if self.combine == "sum" and (
            self.component_weights is not None or self.weight_parameter_names
        ):
            self.combine = "weighted_sum"

        if self.n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        missing = self.shared_parameters.difference(self.parameter_names)
        if missing:
            raise ValueError(
                "shared_parameters contains names that are not in parameter_names: "
                + ", ".join(str(name) for name in sorted(missing, key=str))
            )
        if self.combine not in {"sum", "weighted_sum"}:
            raise ValueError("combine must be either 'sum' or 'weighted_sum'.")
        if self.component_weights is not None and self.weight_parameter_names:
            raise ValueError(
                "Use either component_weights or weight_parameters, not both."
            )
        if self.component_weights is not None:
            self.component_weights = np.asarray(self.component_weights, dtype=float)
            if self.component_weights.shape != (self.n_components,):
                raise ValueError(
                    "component_weights must have one value per component."
                )
        if (
            self.weight_parameter_names
            and len(self.weight_parameter_names) not in {1, self.n_components}
        ):
            raise ValueError(
                "weight_parameters must contain either one binary fraction "
                "parameter or one weight parameter per component."
            )
        if len(self.weight_parameter_names) == 1 and self.n_components != 2:
            raise ValueError(
                "A single weight parameter is only defined for two components."
            )

        self.input_parameter_names = self._input_parameter_names()
        self._input_index = {
            name: index for index, name in enumerate(self.input_parameter_names)
        }

    def __call__(self, obs, parameters, thread=0, **kwargs):
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            raise ValueError("parameters must be a 2D array.")
        if parameters.shape[1] != len(self.input_parameter_names):
            raise ValueError(
                f"Expected {len(self.input_parameter_names)} parameters for "
                f"multi-component simulator, got {parameters.shape[1]}."
            )

        combined = None
        weights = self._component_weights(parameters)
        component_order = self._component_order(parameters)
        weights = np.take_along_axis(weights, component_order, axis=1)
        for component_index in range(self.n_components):
            component_parameters = self._component_parameters(
                parameters,
                component_index,
                component_order,
            )
            component_spectra = self.simulator(
                obs,
                component_parameters,
                thread=f"{thread}_component{component_index + 1}",
                **kwargs,
            )

            if combined is None:
                combined = {
                    key: self._weighted_spectra(value, weights[:, component_index])
                    for key, value in component_spectra.items()
                }
            else:
                for key, value in component_spectra.items():
                    combined[key] += self._weighted_spectra(
                        value,
                        weights[:, component_index],
                    )
        return combined

    def _input_parameter_names(self):
        names = []
        for name in self.parameter_names:
            if name in self.shared_parameters:
                names.append(name)
            else:
                names.extend(
                    _component_parameter_name(name, component_index)
                    for component_index in range(1, self.n_components + 1)
                )
        names.extend(self.weight_parameter_names)
        return names

    def _component_parameters(self, parameters, component_index, component_order):
        all_parameters = np.empty(
            (parameters.shape[0], self.n_components, len(self.parameter_names)),
            dtype=parameters.dtype,
        )
        for original_component_index in range(self.n_components):
            component_number = original_component_index + 1
            for output_index, name in enumerate(self.parameter_names):
                if name in self.shared_parameters:
                    input_name = name
                else:
                    input_name = _component_parameter_name(name, component_number)
                all_parameters[:, original_component_index, output_index] = parameters[
                    :,
                    self._input_index[input_name],
                ]

        sorted_parameters = np.take_along_axis(
            all_parameters,
            component_order[:, :, None],
            axis=1,
        )
        return sorted_parameters[:, component_index, :]

    def _sort_parameter_name(self):
        for name in self.parameter_names:
            if name not in self.shared_parameters:
                return name
        return None

    def _component_order(self, parameters):
        if self.sort_parameter_name is None:
            return np.repeat(
                np.arange(self.n_components).reshape(1, -1),
                parameters.shape[0],
                axis=0,
            )

        values = np.column_stack([
            parameters[
                :,
                self._input_index[
                    _component_parameter_name(self.sort_parameter_name, component)
                ],
            ]
            for component in range(1, self.n_components + 1)
        ])
        return np.argsort(-values, axis=1)

    def _component_weights(self, parameters):
        if self.combine == "sum":
            return np.ones((parameters.shape[0], self.n_components))

        if self.component_weights is not None:
            weights = np.repeat(
                self.component_weights.reshape(1, -1),
                parameters.shape[0],
                axis=0,
            )
        elif len(self.weight_parameter_names) == 1:
            fraction = parameters[:, self._input_index[self.weight_parameter_names[0]]]
            weights = np.column_stack([fraction, 1 - fraction])
        elif len(self.weight_parameter_names) == self.n_components:
            weights = np.column_stack([
                parameters[:, self._input_index[name]]
                for name in self.weight_parameter_names
            ])
        else:
            raise ValueError(
                "combine='weighted_sum' requires component_weights or "
                "weight_parameters."
            )

        if self.normalize_weights:
            weight_sum = np.sum(weights, axis=1, keepdims=True)
            if np.any(weight_sum == 0):
                raise ValueError("Component weights cannot sum to zero.")
            weights = weights / weight_sum
        return weights

    @staticmethod
    def _weighted_spectra(spectra, weights):
        return np.asarray(spectra) * weights[:, None]


def make_multi_component_simulator(
    simulator,
    base_parameters,
    n_components=2,
    shared_parameters=None,
    combine="sum",
    component_weights=None,
    weight_parameters=None,
    normalize_weights=True,
):
    """Create a multi-component simulator and matching parameter dictionary."""
    parameter_names = list(base_parameters.keys())
    normalized_weight_parameters = _normalize_weight_parameters(weight_parameters)
    if combine == "sum" and (
        component_weights is not None or normalized_weight_parameters
    ):
        combine = "weighted_sum"
    wrapped_simulator = MultiComponentSimulator(
        simulator=simulator,
        parameter_names=parameter_names,
        n_components=n_components,
        shared_parameters=shared_parameters,
        combine=combine,
        component_weights=component_weights,
        weight_parameter_names=normalized_weight_parameters.keys(),
        normalize_weights=normalize_weights,
    )
    parameters = make_multi_component_parameters(
        base_parameters,
        n_components=n_components,
        shared_parameters=shared_parameters,
        weight_parameters=normalized_weight_parameters,
    )
    return wrapped_simulator, parameters


def make_multi_component_parameters(
    base_parameters,
    n_components=2,
    shared_parameters=None,
    weight_parameters=None,
):
    """Build a reduced multi-component parameter dictionary."""
    n_components = int(n_components)
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    shared_parameters = set(shared_parameters or [])
    parameter_names = list(base_parameters.keys())
    missing = shared_parameters.difference(parameter_names)
    if missing:
        raise ValueError(
            "shared_parameters contains names that are not in base_parameters: "
            + ", ".join(str(name) for name in sorted(missing, key=str))
        )

    parameters = {}
    for name, metadata in base_parameters.items():
        if name in shared_parameters:
            parameters[name] = deepcopy(metadata)
        else:
            for component_index in range(1, n_components + 1):
                parameters[_component_parameter_name(name, component_index)] = deepcopy(
                    metadata
                )
    parameters.update(_normalize_weight_parameters(weight_parameters))
    return parameters


def make_binary_simulator(
    simulator,
    base_parameters,
    shared_parameters=None,
    combine="sum",
    component_weights=None,
    weight_parameters=None,
    normalize_weights=True,
):
    """Create a two-component simulator and matching parameter dictionary."""
    return make_multi_component_simulator(
        simulator=simulator,
        base_parameters=base_parameters,
        n_components=2,
        shared_parameters=shared_parameters,
        combine=combine,
        component_weights=component_weights,
        weight_parameters=weight_parameters,
        normalize_weights=normalize_weights,
    )


def _component_parameter_name(name, component_index):
    return f"{name}_{component_index}"


def _callable_name(value):
    return getattr(value, "__name__", value.__class__.__name__)


def _normalize_weight_parameters(weight_parameters):
    if weight_parameters is None:
        return {}
    if isinstance(weight_parameters, str):
        weight_parameters = {weight_parameters: (0, 1)}

    normalized = {}
    for name, metadata in weight_parameters.items():
        if isinstance(metadata, dict):
            item = deepcopy(metadata)
        else:
            min_value, max_value = metadata
            item = {"min": min_value, "max": max_value}
        item.setdefault("log", False)
        item.setdefault("post_processing", False)
        item.setdefault("universal", True)
        normalized[name] = item
    return normalized


def check_ARCiS_status(proc, output_dir, n_models, thread):
    """
    Monitor the progress of a Fortran process generating models.

    Args:
        proc (subprocess.Popen): The running Fortran process.
        output_dir (str): Directory where models are generated.
        n_models (int): Total number of models to compute.

    Behavior:
        Tracks and displays progress of model generation in real-time.
        Updates progress bar as new models are detected in output_dir.
        Handles KeyboardInterrupt to stop monitoring gracefully.

    Note:
        Assumes model directories start with "model".
    """
    progress = tqdm(total=n_models, desc=f"Thread {thread}")

    seen = set()  # To avoid double-counting

    try:
        while proc.poll() is None:
            # Find new model directories
            subdirs = [d for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("model")]

            # Count how many are new since last check
            new = set(subdirs) - seen
            progress.update(len(new))
            seen.update(new)

            time.sleep(1)
    except KeyboardInterrupt:
        print("Monitoring interrupted.")
    finally:
        proc.wait()
        progress.close()

def _arcis_obs_file_name(obs_key):
    """Convert a FlopPITy observation key to an ARCiS output filename."""
    if isinstance(obs_key, int):
        return f'obs{obs_key + 1:03d}'

    obs_key = str(obs_key)
    if obs_key.startswith('obs') and obs_key[3:].isdigit():
        return f'obs{int(obs_key[3:]):03d}'

    return obs_key

def _append_arcis_atmosphere_structure(
    model_dir,
    atmosphere_file,
    output_path,
    round_index,
    thread,
    local_model_index,
    global_model_index,
    parameters,
):
    source_path = os.path.join(model_dir, atmosphere_file)
    if not os.path.exists(source_path):
        print(f'Warning: Could not find {atmosphere_file} in {model_dir}')
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(source_path, 'r') as source:
        contents = source.read().rstrip()

    columns = _arcis_atmosphere_columns(contents)
    contents = _arcis_atmosphere_numeric_contents(contents)
    parameter_text = ' '.join(f'{value:.17g}' for value in np.asarray(parameters))
    with open(output_path, 'a') as destination:
        fcntl.flock(destination, fcntl.LOCK_EX)
        try:
            if columns:
                destination.write(f'# columns={" ".join(columns)}\n')
            destination.write(
                f'# round={round_index} '
                f'global_model={global_model_index} '
                f'thread={thread} '
                f'local_model={local_model_index} '
                f'parameters={parameter_text}\n'
            )
            destination.write(contents)
            destination.write('\n\n')
        finally:
            fcntl.flock(destination, fcntl.LOCK_UN)


def _arcis_atmosphere_columns(contents):
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        candidate = stripped.lstrip('#').strip()
        tokens = [
            token
            for token in candidate.split()
            if not (token.startswith('[') and token.endswith(']'))
        ]
        if len(tokens) >= 2 and any(char.isalpha() for char in ''.join(tokens)):
            return tokens
    return []


def _arcis_atmosphere_numeric_contents(contents):
    numeric_lines = []
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        try:
            float(stripped.split()[0])
        except (ValueError, IndexError):
            continue
        numeric_lines.append(line)
    return '\n'.join(numeric_lines)
