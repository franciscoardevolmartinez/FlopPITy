import numpy as np
import torch


def create_obs_file(wvl, spectrum, error, *args):
    """
    Combines wavelength, spectrum, error, and optional extra arrays 
    into a single observation array.

    Parameters
    ----------
    wvl : array_like
        1D array of wavelengths.
    spectrum : array_like
        1D array of spectral flux/transit depths values corresponding to `wvl`.
    error : array_like
        1D array of uncertainties associated with `spectrum`.
    *args : array_like
        Optional additional 1D arrays of the same length as `wvl` to be 
        included in the output.

    Returns
    -------
    obs : ndarray
        A 2D NumPy array of shape (len(wvl), 3 + len(args)), where each row contains
        `[wavelength, spectrum, error, *args]` for a given wavelength index.
    """
    extras = len(args)
    obs = np.empty([len(wvl), 3+extras])
    obs[:,0]=wvl
    obs[:,1]=spectrum
    obs[:,2]=error
    for i in range(extras):
        obs[:,3+i]=args[i]

    return obs

def convert_cube(thetas, pars):
    """
    Converts unit hypercube samples to physical parameter values.

    Parameters
    ----------
    thetas : ndarray
        A 2D NumPy array of shape (n_samples, n_parameters) containing 
        samples from a unit hypercube (values in [0, 1]).
    pars : dict
        Dictionary containing parameter metadata. Each key corresponds 
        to a parameter name, and each value is a dictionary with at 
        least the keys `'min_value'` and `'max_value'`.

    Returns
    -------
    natural : ndarray
        A 2D NumPy array of the same shape as `thetas` containing the 
        transformed samples scaled to the corresponding parameter 
        ranges defined in `pars`.
    """
    natural = np.empty_like(thetas)
    key = list(pars.keys())
    for i in range(len(pars)):
        minv=pars[key[i]]['min']
        maxv=pars[key[i]]['max']
        natural[:,i] = minv+(maxv-minv)*thetas[:,i]
    return natural

def compute_moments(distribution):
    """
    Computes the first four moments (mean, variance, skewness, kurtosis) of a PyTorch distribution.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        A PyTorch distribution object.

    Returns
    -------
    moments : dict
        A dictionary containing the first four moments:
        - 'mean': Mean of the distribution.
        - 'variance': Variance of the distribution.
        - 'skewness': Skewness of the distribution.
        - 'kurtosis': Kurtosis of the distribution.
    """
    samples = distribution.sample((10000,))

    mean = torch.mean(samples)
    variance = distribution.variance

    # Skewness and kurtosis require sampling
    samples = distribution.sample((100000,))
    skewness = torch.mean(((samples - mean) / torch.sqrt(variance))**3).item()
    kurtosis = torch.mean(((samples - mean) / torch.sqrt(variance))**4).item() - 3

    return {
        'mean': mean.item(),
        'variance': variance.item(),
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def find_MAP(proposal):
    """
    Finds the Maximum A Posteriori (MAP) estimate for a given 
    proposal.

    Parameters:
        proposal (object): The proposal object that contains the 
                           method `map` to perform the MAP 
                           estimation.

    Returns:
        object: The result of the MAP estimation process.

    Notes:
        The `map` method of the proposal object is called with the 
        following parameters:
            - x (None): Placeholder for input data, set to None by 
              default.
            - num_iter (int): Number of iterations for the 
              optimization process (default: 100).
            - num_to_optimize (int): Number of parameters to 
              optimize (default: 100).
            - learning_rate (float): Learning rate for the 
              optimization process (default: 0.01).
            - init_method (str): Initialization method for the 
              optimization (default: 'posterior').
            - num_init_samples (int): Number of initial samples to 
              generate (default: 1000).
            - save_best_every (int): Frequency (in iterations) to 
              save the best result (default: 10).
            - show_progress_bars (bool): Whether to display progress 
              bars during optimization (default: True).
            - force_update (bool): Whether to force an update during 
              optimization (default: False).
    """
    return proposal.map(x=None, num_iter=100, num_to_optimize=100, 
                        learning_rate=0.01, init_method='posterior', 
                       num_init_samples=1000, save_best_every=10, 
                       show_progress_bars=True, force_update=False
                       )

def reduced_chi_squared(obs_dict, sim_dict, n_params=0):
    """
    Compute reduced chi-squared values for each simulation vs observation.

    Parameters
    ----------
    obs_dict : dict
        Keys are observation names. Values are arrays of shape (n_points, ≥3)
        with columns: wavelength, observed spectrum, error.
    sim_dict : dict
        Keys are the same as obs_dict. Values are arrays of shape
        (n_samples, n_points), corresponding to simulated spectra.
    n_params : int
        Number of fitted parameters to subtract from degrees of freedom (default: 0).

    Returns
    -------
    chi2_dict : dict
        Dictionary with same keys, values are arrays of shape (n_samples,)
        with reduced chi-squared values.
    """

    chi2_dict = {}

    for key in obs_dict:
        obs_array = obs_dict[key]
        sims = sim_dict[key]

        if obs_array.shape[1] < 3:
            raise ValueError(f"Observation {key} must have at least 3 columns (wvl, spectrum, error)")

        obs_spectrum = obs_array[:, 1]
        obs_error = obs_array[:, 2]

        if sims.shape[1] != len(obs_spectrum):
            raise ValueError(f"Shape mismatch for {key}: sims shape {sims.shape}, obs length {len(obs_spectrum)}")

        # Avoid division by zero in error
        error_safe = np.where(obs_error == 0, 1e-10, obs_error)

        # Compute chi-squared
        residuals = (sims - obs_spectrum) / error_safe
        chi2 = np.sum(residuals**2, axis=1)

        # Degrees of freedom
        dof = len(obs_spectrum) - n_params
        if dof <= 0:
            raise ValueError(f"Non-positive degrees of freedom for {key}: {dof}")

        chi2_red = chi2 / dof
        chi2_dict[key] = chi2_red

    return chi2_dict

def find_best_fit(obs_dict, sim_dict):
    """
    Find the simulation with the lowest reduced chi-squared value 
    for each observation.

    Parameters
    ----------
    obs_dict : dict
        Keys are observation names. Values are arrays of shape 
        (n_points, ≥3) with columns: wavelength, observed spectrum, 
        error.
    sim_dict : dict
        Keys are the same as obs_dict. Values are arrays of shape 
        (n_samples, n_points), corresponding to simulated spectra.

    Returns
    -------
    best_fit_dict : dict
        Dictionary with the same keys as obs_dict. Values are tuples 
        of the form (best_fit_index, best_fit_chi2), where:
        - best_fit_index is the index of the simulation with the 
          lowest chi-squared value.
        - best_fit_chi2 is the corresponding reduced chi-squared 
          value.
    """
    chi2_dict = reduced_chi_squared(obs_dict, sim_dict)
    best_fit_dict = {}

    for key, chi2_values in chi2_dict.items():
        best_fit_index = np.argmin(chi2_values)
        best_fit_chi2 = chi2_values[best_fit_index]
        best_fit_dict[key] = (best_fit_index, best_fit_chi2)

    return best_fit_dict
