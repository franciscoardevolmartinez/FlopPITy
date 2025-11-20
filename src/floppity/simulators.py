import numpy as np
import os
import time
from tqdm import trange, tqdm
import subprocess
import importlib

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

    obs_list : list of str
        A list of file paths to the observational datasets specified in the
        ARCiS input.
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
            obs_dict[f'obs{obsn}']=path
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
    os.makedirs(output_dir, exist_ok=True)

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
    # obs_indices = sorted(obs.keys())
    obs_indices = obs.keys()
    obs_files = [f'obs{idx+1:03d}' for idx in range(len(obs_indices))]

    # Write parameter grid file
    param_file = os.path.join(output_dir, f'parametergridfile_{thread}.dat')
    np.savetxt(param_file, parameters)

    # Run ARCiS
    log_files = [f for f in os.listdir(output_dir) if f.startswith(f'arcis_run_{thread}_') and f.endswith('.log')]
    log_nums = []
    for f in log_files:
        try:
            num = int(f[len(f'arcis_run_{thread}_'):-len('.log')])
            log_nums.append(num)
        except ValueError:
            continue
    next_log_num = max(log_nums) + 1 if log_nums else 1
    log_file = os.path.join(output_dir, f'arcis_run_{thread}_{next_log_num}.log')

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

        for k, obs_file in zip(obs_indices, obs_files):
            try:
                phase = np.loadtxt(os.path.join(model_dir, obs_file))[:, 1]
            except Exception as e:
                print(f'Warning: Could not read {obs_file} in {model_dir}: {e}')
                phase = -1 * np.ones_like(obs[k][:, 1])  # fallback
            spectra[k].append(phase)

    for k in spectra:
        spectra[k] = np.array(spectra[k])  # shape: (n_spectra, n_points)

    #Remove files
    print('Removing files...')
    for root, dirs, files in os.walk(output_base, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for sub_root, sub_dirs, sub_files in os.walk(dir_path, topdown=False):
                for sub_file in sub_files:
                    os.remove(os.path.join(sub_root, sub_file))
                for sub_dir in sub_dirs:
                    os.rmdir(os.path.join(sub_root, sub_dir))
            os.rmdir(dir_path)

    return spectra

### This needs to be made general for any model and any number of objects/atmospheric columns.
def ARCiS_multiple(obs, parameters, thread=0, **kwargs):
    """
    Generalized ARCiS multi-component model (binary, triple, ...).

    Automatically sorts components per row so that the first parameter
    of component 1 >= component 2 >= component 3 ... etc.

    Args
    ----
    obs : dict
        Observational data.
    parameters : ndarray (N_samples, N_params_total)
        Parameters for all components stacked horizontally.
    thread : int
        Thread number for parallel execution.
    kwargs :
        - n_components : int
        - anything else passed to ARCiS()

    Returns
    -------
    combined : dict
        Combined multi-component spectra.
    """

    n_components = kwargs.get("n_components", 2)
    N_samples, N_total_params = parameters.shape

    # if N_total_params % n_components != 0:
    #     raise ValueError(
    #         f"Parameter count {N_total_params} not divisible by n_components={n_components}"
    #     )

    nparams = N_total_params // n_components

    # ----------------------------------------------------------
    # 1) RESHAPE: (N_samples, n_components, nparams)
    # ----------------------------------------------------------
    # params_3d = parameters.reshape(N_samples, n_components, nparams)

    # ----------------------------------------------------------
    # 2) SORT blocks per row by first parameter DESCENDING
    # ----------------------------------------------------------
    # Sorting key = params_3d[:, :, 0]  (shape = N_samples x n_components)
    # sort_idx = np.argsort(-params_3d[:, :, 0], axis=1)  

    # Gather sorted parameters
    # row_indices = np.arange(N_samples)[:, None]
    # params_sorted = params_3d[row_indices, sort_idx]

    # ----------------------------------------------------------
    # 3) Flatten back to original shape
    # ----------------------------------------------------------
    # parameters_sorted = params_sorted.reshape(N_samples, N_total_params)

    # ----------------------------------------------------------
    # 4) Evaluate ARCiS for each component
    # ----------------------------------------------------------
    objects = {}

    for i in range(n_components):
        print(f"Computing models for component {i+1} (sorted block)")

        component_params = parameters[:, i*nparams:(i+1)*nparams]

        objects[i] = ARCiS(
            obs,
            component_params,
            thread=thread,
            **kwargs
        )

    # ----------------------------------------------------------
    # 5) Combine fluxes linearly
    # ----------------------------------------------------------
    combined = {}
    first_key = list(objects[0].keys())[0]

    for k in objects[0]:
        combined[k] = sum(objects[i][k] for i in range(n_components))

    return combined

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

def PICASO(obs, parameters, thread=0, **kwargs):

    try:
        tomllib = importlib.import_module('tomllib')
    except ImportError:
        raise ImportError(
            f"Package tomllib not found. "
            f"Please install it separately."
        )
    try:
        toml = importlib.import_module('toml')
    except ImportError:
        raise ImportError(
            f"Package toml not found. "
            f"Please install it separately."
        )
    
    try:
        picaso = importlib.import_module('picaso')
    except ImportError:
        raise ImportError(
            f"Picaso not found. "
            f"Please install it separately."
        )

    config_f=kwargs.get('config_file', None)
    assert config_f is not None, "Config file can not be None!"

    with open(config_f, "rb") as f:
            config = tomllib.load(f)

    os.makedirs(config['InputOutput']['retrieval_output'], exist_ok=True)

    output_file_name = config['InputOutput']['retrieval_output']+"/inputs.toml"
    with open(output_file_name, "w") as toml_file:
        toml.dump(config, toml_file)

    OPA = picaso.justdoit.opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        )

    prior_config=config['retrieval']
    
    fitpars=picaso.driver.prior_finder(prior_config)
    ndims=len(fitpars)

    preload_cloud_miefs = picaso.driver.find_values_for_key(config ,'condensate')
    virga_mieff   = config['OpticalProperties'].get('virga_mieff',None)
    param_tools = picaso.parameterizations.Parameterize(load_cld_optical=preload_cloud_miefs,
                                    mieff_dir=virga_mieff)
    
    for i,key in enumerate(fitpars.keys()):
        if fitpars[key]['log']:
            parameters[:,i] = 10**parameters[:,i]
    
    DATA_DICT = picaso.driver.get_data(config)
    x = picaso.driver.MODEL(parameters, fitpars, config, OPA, param_tools, DATA_DICT)

    return x

def read_PICASO_config(config_file):

    #load necessary packages
    try:
        tomllib = importlib.import_module('tomllib')
    except ImportError:
        raise ImportError(
            f"Package toml not found. "
            f"Please install it separately."
        )
    
    try:
        picaso = importlib.import_module('picaso')
    except ImportError:
        raise ImportError(
            f"Picaso not found. "
            f"Please install it separately."
        )
    
    try:
        xarray = importlib.import_module('xarray')
    except ImportError:
        raise ImportError(
            f"Package xarray not found. "
            f"Please install it separately."
        )

    # Get fitted parameters and priors
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    prior_config=config['retrieval']
    fitpars=picaso.driver.prior_finder(prior_config)
    par_dict={}

    for i,key in enumerate(fitpars.keys()):
        if fitpars[key]['prior'] == 'uniform':
            minn=fitpars[key]['min']
            maxx=fitpars[key]['max']
            
            if fitpars[key]['log']:
                # minn=10**minn
                # maxx=10**maxx
                log_flag=True
            else:
                log_flag=False

        par_dict[key] = {
                'min': minn,
                'max': maxx,
                'log': log_flag,
                'post_processing': False,
                'universal': False
            }
        
    # Read observations
    data_dict = picaso.driver.get_data(config)

    obs_dict={}
    for key in data_dict:
        obs_dict[key] = np.column_stack(data_dict[key])
        
    return par_dict, obs_dict