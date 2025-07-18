import numpy as np
# from geomloss import SamplesLoss
import torch
from scipy.special import logsumexp


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

    mean = no.mean(samples)
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

# def W2_distance(proposals, n_mc=100, n_draws=1000):
    # """
    # Computes the Wasserstein-2 (W2) distance between the last two 
    # posteriors to gauge convergence.

    # Parameters:
    # -----------
    # proposals : list
    #     A list of proposal distributions. The last two elements of 
    #     the list (`proposals[-2]` and `proposals[-1]`) are used to 
    #     compute the W2 distance.
    
    # n_mc : int, optional, default=100
    #     The number of Monte Carlo estimations of the W2 distance. 
    #     This controls how many times the W2 distance is computed 
    #     with different random draws to estimate its mean and error.

    # n_draws : int, optional, default=1000
    #     The number of samples to draw from each proposal distribution 
    #     in each Monte Carlo estimation.

    # Returns:
    # --------
    # self.w2 : float
    #     The mean Wasserstein-2 distance computed across all Monte 
    #     Carlo simulations.

    # self.w2_err : float
    #     The standard error of the W2 distance, computed from the 
    #     Monte Carlo simulations.

    # Example:
    # --------
    # proposals = [proposal_1, proposal_2]  # Example proposals
    # w2_dist = some_object.W2_distance(proposals, n_mc=100, n_draws=1000)
    
    # print(f"W2 distance: {w2_dist.w2}")
    # print(f"Standard Error: {w2_dist.w2_err}")
    
    # Notes:
    # ------
    # - This function uses the `SamplesLoss` class from `geomloss` to compute the 
    # Sinkhorn approximation of the Wasserstein-2 distance.
    # - The parameter `blur` in `SamplesLoss` controls the level of regularization 
    # applied to the Wasserstein distance computation (default is 0.05).
    # - The function assumes that `proposals` contains at least two proposal distributions, 
    # and it uses the last two for the comparison (`proposals[-2]` and `proposals[-1]`).
    # """
    # loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    # old_thetas = proposals[-2].sample((n_draws,)).reshape((
    #     n_draws,-1))
    # new_thetas = proposals[-1].sample((n_draws,)).reshape((
    #     n_draws,-1))

    # emd_inter=[]
    # for i in range(n_mc):
    #     emd_inter.append(loss_fn(old_thetas, new_thetas)*n_draws)

    # w2=np.mean(emd_inter)
    # w2_err=np.std(emd_inter)

    # return w2, w2_err

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

def likelihood(x_dict, obs_dict):
    """Calculate the log likelihood of model spectra compared to observations.

    Args:
        x_dict (dict): Dictionary of simulated spectra (keys are instruments, values are arrays).
        obs_dict (dict): Dictionary of observed spectra (keys are instruments, values are arrays).

    Returns:
        np.ndarray: Log likelihood values for each simulated spectrum across all instruments.
    """
    log_likelihoods = []

    for key in obs_dict:
        obs = obs_dict[key]
        x = x_dict[key]

        if obs.shape[1] < 3:
            raise ValueError(f"Observation {key} must have at least 3 columns (wavelength, spectrum, error)")

        if x.shape[1] != len(obs[:, 1]):
            raise ValueError(f"Shape mismatch for {key}: x shape {x.shape}, obs length {len(obs[:, 1])}")

        residual = obs[:, 1] - x
        log_likelihood = -0.5 * np.sum((residual**2 / obs[:, 2]**2) + np.log(2 * np.pi * obs[:, 2]**2), axis=1)
        log_likelihoods.append(log_likelihood)

    # Concatenate log likelihoods from all instruments
    return np.concatenate(log_likelihoods)

def importance_weights(log_likelihoods, log_priors, log_proposal):
    """Calculate importance weights for a proposal distribution.

    (From Gebhard+25)

    Args:
        x (np.ndarray): Model spectra.
        o (np.ndarray): Observed spectra.
        s (np.ndarray): Errors in the observed spectra.
        P (np.ndarray): Proposal distribution probabilities.

    Returns:
        np.ndarray: Importance weights normalized to sum to 1.
    """
    # log_likelihoods = likelihood(x,o,s)
    log_weights = (log_likelihoods + log_priors - log_proposal).detach().numpy()
    N = len(log_weights)
    normalized_weights = np.exp(
        np.log(N) + log_weights - logsumexp(log_weights)
    )
    return log_weights, normalized_weights

def eff(weights):    
    n_eff = float(np.sum(weights) ** 2 / np.sum(weights**2))
    sampling_efficiency = n_eff / len(weights)
    return n_eff, sampling_efficiency

def IS_evidence(weights):
    n = len(weights)
    n_eff,_ = eff(weights)

    log_evidence = float(logsumexp(weights) - np.log(n))
    log_evidence_std = float(np.sqrt((n - n_eff) / (n * n_eff)))

    return log_evidence, log_evidence_std
