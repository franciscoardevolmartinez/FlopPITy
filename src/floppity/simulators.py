import numpy as np
import os
from tqdm import trange
import subprocess

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
    obs_list = []
    obsn = 1
    for line in clean_in:
        key = f'obs{obsn}:file'
        if key in line:
            path = line.split('=')[-1].strip().strip('"')
            obs_list.append(path)
            obsn += 1

    return par_dict, obs_list

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
    obs_indices = sorted(obs.keys())
    obs_files = [f'obs{idx+1:03d}' for idx in obs_indices]

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
            subprocess.run(
                [ARCiS, input_copy, "-o", output_base, "-s", f"parametergridfile={param_file}"],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
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
