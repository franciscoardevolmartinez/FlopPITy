import numpy as np
import torch
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from sbi.inference import SNPE_C
from floppity import helpers
from sbi.utils import RestrictedPrior, get_density_thresholder
from floppity import postprocessing
from floppity import preprocessing
import multiprocessing as mp
import cloudpickle as pickle
from corner import corner
from scipy.stats.qmc import LatinHypercube, Sobol
import os

class Retrieval():
    def __init__(self, simulator, obs_type):
        """
        simulator (callable): A function or callable object that 
            simulates data based on the provided parameters. 
            The simulator must take as input the observation dictionary
            and an array of parameters of shape (n_samples, n_dims).
            Additionally, it must return a dictionary with the same keys
            as the observation.

        obs_type (str): either 'emis' or 'trans'
        """

        self.simulator = simulator
        self.parameters = {}
        self.preprocessing=None
        self.obs_type=obs_type

    def save(self, fname, **options):
        """
        Save the current object to a file using pickle.

        Args:
            fname (str): The file path where the object will be saved.
            **options: Additional options for future extensions.

        Raises:
            IOError: If the file cannot be written.
        """
        with open(fname, 'wb') as file:
            pickle.dump(self, file)

    def __getstate__(self):
        """
        Customize object state for pickling by excluding specific attributes.

        Removes 'noisy_x', 'post_x', 'x', 'nat_thetas', and 'thetas' from 
        the object's state dictionary before serialization.
        """
        state = self.__dict__.copy()
        state.pop('noisy_x', None)
        state.pop('post_x', None)
        state.pop('x', None)
        state.pop('nat_thetas', None)
        state.pop('thetas', None)
        return state

    def __setstate__(self, state):
        """
        Restore the object's state from the provided state dictionary.

        Args:
            state (dict): The state dictionary containing attributes to 
            restore.
        """
        self.__dict__.update(state)

    @classmethod   
    def load(cls, fname):
        """
        Load an object from a file using pickle.

        Args:
            fname (str): The path to the file to load.

        Returns:
            object: The object loaded from the file.
        """
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    def get_obs(self, fnames):
        '''
        Read observation(s) to run retrievals on. Needs to be in the 
        format required by the simulator used.

        Parameters
        ----------
        fnames : list of strings
            list with the locations of all observations to analyze

        Returns
        -------
        obs : dict
            dictionary with all the observations, keyed 0, 1, 2...
        '''
        self.obs={}
        for i in range(len(fnames)):
            self.obs[i] = np.loadtxt(fnames[i])

        self.default_obs=np.concatenate(list(self.obs.values()), 
                                        axis=0)[:,1].reshape(1,-1)
         
    def add_parameter(self, parname, min_value, max_value, 
                      log_scale=False, post_process=False, 
                      universal=True):
        """
        Add a parameter to the internal parameter dictionary.

        Parameters
        ----------
        parname : str
            Name of the parameter to add.
        min_value : float
            Minimum allowed value for the parameter.
        max_value : float
            Maximum allowed value for the parameter.
        log_scale : bool, optional
            If True, indicates that the parameter should be sampled 
            logarithmically. Default is False.
        post_process : bool, optional
            If True, it indicates that this is a parameter from a 
            post-processing function and not the simulator. 
            Default is False.
        universal : bool, optional
            Only used when combining multiple models. If True, the 
            parameter is kept the same for all models. Default is True.

        Returns
        -------
        None
        """
        self.parameters[parname]={}
        self.parameters[parname]['min']=min_value
        self.parameters[parname]['max']=max_value
        self.parameters[parname]['log']=log_scale
        self.parameters[parname]['post_processing']=post_process
        self.parameters[parname]['universal']=universal

    def get_thetas(self, proposal, n_samples):
        """
        Draw a set of parameter samples (thetas) from the proposal 
        distribution.

        Parameters
        ----------
        proposal : object
            A proposal distribution object that implements a `sample(n)`
              method.

        n_samples : int
            The number pf parameter vectors to draw.

        Returns
        -------
        None
            The sampled parameter vectors are stored in `self.thetas`.
        """
        self.n_samples=n_samples
        thetas = proposal.sample((self.n_samples,))
        self.thetas=thetas.cpu().detach().numpy().reshape([-1, len(self.parameters)])
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)
    
    def lhs_thetas(self, n_samples):
        """
        Generate Latin Hypercube samples for parameter space.

        This method creates `n_samples` samples using a Latin Hypercube 
        sampling strategy based on the dimensions of the `parameters` 
        attribute. The generated samples are stored in `self.thetas`, 
        and their natural parameter space equivalents are stored in 
        `self.nat_thetas`.

        Args:
            n_samples (int): Number of samples to generate.

        Attributes:
            thetas (ndarray): Generated samples in the unit hypercube.
            nat_thetas (ndarray): Samples converted to natural parameter 
                space.
        """
        self.n_samples=n_samples
        dims = len(self.parameters)
        sampler = LatinHypercube(d=dims, optimization='random-cd')
        self.thetas = sampler.random(n=n_samples)
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)

    def sobol_thetas(self, n_samples):
        """
        Generate Latin Hypercube samples for parameter space.

        This method creates `n_samples` samples using a Latin Hypercube 
        sampling strategy based on the dimensions of the `parameters` 
        attribute. The generated samples are stored in `self.thetas`, 
        and their natural parameter space equivalents are stored in 
        `self.nat_thetas`.

        Args:
            n_samples (int): Number of samples to generate.

        Attributes:
            thetas (ndarray): Generated samples in the unit hypercube.
            nat_thetas (ndarray): Samples converted to natural parameter 
                space.
        """
        self.n_samples=n_samples
        dims = len(self.parameters)
        sampler = Sobol(d=dims, optimization='random-cd')
        self.thetas = sampler.random(n=n_samples)
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)
    
    def get_x(self, x):
        """
        Ingests the simulated spectra into the class.

        Parameters
        ----------
        x: dict
            A dictionary with simulated spectra, with one key per observation.
        """
        self.x={}
        for key in x.keys():
            self.x[key] = x[key].reshape(self.n_samples, 
                                        len(self.obs[key][:,0]))

    def create_prior(self):
        """
        Constructs a uniform prior distribution based on the min and max
        values of parameters in self.parameters.

        Raises:
            KeyError: If any parameter is missing 'min_value' or 
            'max_value'.
        """
        # prior_mins = []
        # prior_maxs = []
        # for keys in self.parameters.keys():
        #     prior_mins.append(self.parameters[keys]['min_value'])
        #     prior_maxs.append(self.parameters[keys]['max_value'])
        
        # prior_mins = torch.tensor(np.asarray(prior_mins).reshape(1,-1))
        # prior_maxs = torch.tensor(np.asarray(prior_maxs).reshape(1,-1))

        self.prior=utils.BoxUniform(low=torch.zeros(len(self.parameters)),
                                    high=torch.ones(len(self.parameters)))

    def density_builder(self, flow='nsf', transforms=10, hidden=50, 
                        blocks=3, bins=8, dropout=0.05, 
                        z_score_theta='independent', z_score_x='independent',
                        use_batch_norm=True):
        """
        Build the density estimator for the posterior distribution.
        This function initializes a neural network model for posterior
        inference using the specified parameters.
        Parameters
        ----------
        flow : str
            The type of flow model to use (e.g., 'NSF', 'MAF').
        transforms : int
            The number of transformations to apply in the flow model.
        hidden : int
            The number of hidden units per layer in the neural network.
        blocks : int
            The number of residual blocks in the neural network.
        bins : int
            The number of bins in the NSF model.
        dropout : float
            The dropout probability for the neural network.
        Returns
        -------
        None
            The density estimator is stored in `self.density`.
        """
        self.density = posterior_nn(model=flow, 
                                    num_transforms=transforms, 
                                    hidden_features=hidden, 
                                    num_blocks=blocks, 
                                    num_bins=bins, 
                                    dropout_probability=dropout,
                                    z_score_theta=z_score_theta,
                                    z_score_x=z_score_x,
                                    use_batch_norm=use_batch_norm)
        self.inference = SNPE_C(prior=self.prior, 
                                density_estimator=self.density)

    def train(self, theta, x, proposal, **kwargs):
        """
        Train the density estimator using the provided parameter samples
        and observations.
        Returns
        -------
        None
            The trained density estimator is stored in `self.density`.
        """

        self.posterior_estimator = self.inference.append_simulations(theta, x, 
                                          proposal=proposal).train(show_train_summary=True,
                              **kwargs)

    def get_posterior(self):
        """
        Build and configure the posterior distribution from the 
        trained inference object.

        This function constructs the posterior using the trained 
        inference method and the specified posterior estimator. It 
        also sets the default observation `self.default_obs` for 
        future sampling from the posterior.

        Parameters
        ----------
        x0 : torch.Tensor
            The observation to condition the posterior on.

        Attributes
        ----------
        self.posterior : sbi.inference.posteriors.Posterior
            Posterior distribution built from the inference object, 
            conditioned on `self.default_obs`.
        """
        self.default_obs_norm = self.do_preprocessing(self.default_obs)

        self.posterior=self.inference.build_posterior(
            self.posterior_estimator)
    
    def generate_training_data(self, proposal, r, n_samples, n_samples_init,
                           sample_prior_method, n_threads, simulator_kwargs, 
                           n_aug, reuse_prior=None):
        
        if (reuse_prior is not None) and (r == 0):
            print(f"Reusing prior data from {reuse_prior}")
            prior_data = pickle.load(open(reuse_prior, 'rb'))

            #Are there enough samples to reuse?
            reused_n = min(len(prior_data['par']), n_samples)
            remaining_n = n_samples - reused_n

            reused_thetas = prior_data['par'][:reused_n]
            reused_x = {key: value[:reused_n] for key, value in prior_data['spec'].items()}

            if remaining_n > 0:
                print(f"Generating {remaining_n} additional samples.")
                self._sample_initial_thetas(sample_prior_method, remaining_n)
                self.sim_pars = sum(1 for key in self.parameters if not self.parameters[key]['post_processing'])

                # Simulate for additional parameters
                if n_threads == 1:
                    new_x = self.simulator(self.obs, self.nat_thetas[:, :self.sim_pars], **simulator_kwargs)
                else:
                    new_x = self.psimulator(self.obs, self.nat_thetas[:, :self.sim_pars], **simulator_kwargs)

                # Combine thetas
                all_thetas = np.concatenate([reused_thetas, self.thetas], axis=0)

                # Combine x dicts
                all_x = {}
                for key in reused_x:
                    all_x[key] = np.concatenate([reused_x[key], new_x[key]], axis=0)
            
            else:
                all_thetas = reused_thetas
                all_x = reused_x

            self.thetas = all_thetas
            self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)
            self.n_samples = all_thetas.shape[0]
            self.get_x(all_x)

        else:
            print('Generating training examples.')
            # Sample parameters
            if len(self.proposals) == 0:
                self._sample_initial_thetas(sample_prior_method, n_samples_init)
            else:
                self.get_thetas(proposal, n_samples)

            # Determine how many parameters to simulate
            self.sim_pars = sum(1 for key in self.parameters if not self.parameters[key]['post_processing'])

            # Simulate
            if n_threads == 1:
                xs = self.simulator(self.obs, self.nat_thetas[:, :self.sim_pars], **simulator_kwargs)
            else:
                xs = self.psimulator(self.obs, self.nat_thetas[:, :self.sim_pars], **simulator_kwargs)

            self.get_x(xs)

        # Postprocessing
        self.do_postprocessing()
        self.augment(n_aug)
        self.add_noise()

        if self.obs_type == 'emis':
            for key in self.noisy_x.keys():
                self.noisy_x[key][self.noisy_x[key] <= 0] = 1e-11

        # Convert to tensors
        theta_tensor = torch.tensor(self.augmented_thetas, dtype=torch.float32)
        x_tensor = torch.tensor(np.concatenate(list(self.noisy_x.values()), axis=1), dtype=torch.float32)
        return theta_tensor, x_tensor

    def run(self, n_threads=1, n_samples=100, n_samples_init=None, n_agg=1,
                        resume=False, n_rounds=10, n_aug=1, flow_kwargs=dict(), 
                        training_kwargs=dict(), simulator_kwargs=dict(), output_dir='output_FlopPITy',
                        save_data=False, sample_prior_method='sobol',
                        reuse_prior=None, IS=True
                        ):
        """
        Executes the retrieval process over multiple rounds, 
        involving prior creation, density estimation, simulation, 
        post-processing, noise addition, and training.

        Args:
            n_rounds: int
                Number of rounds to train iteratively.
            **flow_kwargs: Additional keyword arguments for the 
            density builder.
            **training_kwargs: Additional keyword arguments for the 
            training process.

        Workflow:
            1. Initializes the proposals list and creates the prior 
               distribution.
            2. Builds the density estimator using the provided flow 
               arguments.
            3. Iteratively performs the following for each round:
               - Samples parameters (thetas) from the current proposal 
             distribution.
               - Masks parameters based on the 'post_process' flag in 
             self.parameters.
               - Simulates data (xs) using the masked parameters and 
             observed data (R.obs).
               - Processes the simulated data and adds noise.
               - Trains the model using the masked parameters and noisy 
             data.
               - Updates the posterior distribution and sets it as the 
             new proposal.
               - Appends the current proposal to the proposals list.

        Notes:
            - The method relies on several other methods within the 
              class, such as `create_prior`, `density_builder`, 
              `get_thetas`, `get_x`, `do_postprocessing`, `add_noise`, 
              `train`, and `get_posterior`.

        Raises:
            Any exceptions raised by the simulator, training, or other 
            internal methods will propagate up to the caller.
        """
        self.n_threads=n_threads

        if n_samples_init==None:
            n_samples_init=n_samples

        if resume:
            print('Resuming training...')
            proposal=self.proposals[-1]
        else: 
            print('Starting training...')
            self.IS_sampling_efficiency=[]
            self.logZ=[]
            self.dlogZ=[]
            self.create_prior()
            self.proposals=[self.prior]
            self.density_builder(**flow_kwargs)
            proposal=self.prior

        for r in range(n_rounds):
            print(f"Round {r+1}")

            theta_tensor, x_tensor = self.generate_training_data(
                proposal=proposal, 
                r=r,
                n_samples=n_samples, 
                n_samples_init=n_samples_init, 
                sample_prior_method=sample_prior_method,
                n_threads=n_threads, 
                simulator_kwargs=simulator_kwargs, 
                n_aug=n_aug,
                reuse_prior=reuse_prior if r == 0 else None
            )

            # Do IS stuff
            if IS:
                n_eff, sampling_efficiency, logZ, dlogZ= self.run_IS()
                self.IS_sampling_efficiency.append(sampling_efficiency)
                self.logZ.append(logZ)
                self.dlogZ.append(dlogZ)
                print(f'ε = {sampling_efficiency:.3g}')
                print(f'log(Z) = {logZ:.3g} +- {dlogZ:.3g}')

            os.makedirs(output_dir, exist_ok=True)
            self.save(f'{output_dir}/retrieval.pkl')
            if save_data==True:
                pickle.dump(dict(par=self.thetas, spec=self.x), open(f'{output_dir}/data_{r}.pkl', 'wb'))

            x_norm_tensor = self.do_preprocessing(x_tensor)
                
            self.train(theta_tensor, x_norm_tensor, proposal, **training_kwargs)
            
            self.get_posterior()
            proposal = self.posterior.set_default_x(self.default_obs_norm)
            self.proposals.append(self.posterior)
            self.loss_val=self.inference._summary['best_validation_loss']
    
    def _sample_initial_thetas(self, method, n_samples):
        if method == 'random':
            self.get_thetas(self.prior, n_samples)
        elif method == 'lhs':
            self.lhs_thetas(n_samples)
        elif method == 'sobol':
            if (n_samples & (n_samples - 1)) != 0 or n_samples <= 0:
                n_samples_new = 2 ** int(np.round(np.log2(n_samples)))
                print(f'n_samples must be a power of 2 for Sobol sampling. I will sample the prior with {n_samples_new} samples and then go back to {n_samples} for the following rounds.')
                n_samples = n_samples_new
            self.sobol_thetas(n_samples)

    def run_IS(self):
        log_likelihoods= torch.tensor(helpers.likelihood(self.post_x, self.obs))
        log_priors = self.prior.log_prob(torch.tensor(self.thetas))
        log_proposal=self.proposals[-1].log_prob(torch.tensor(self.thetas))

        self.raw_IS_weights, self.IS_weights=helpers.importance_weights(log_likelihoods, log_priors, log_proposal)

        n_eff, sampling_efficiency = helpers.eff(self.IS_weights)

        logZ, dlogZ = helpers.IS_evidence(self.raw_IS_weights)

        return n_eff, sampling_efficiency, logZ, dlogZ
    
    def add_noise(self):
        """
        Add Gaussian noise to the spectra based on the observational errors.

        For each observation in `self.obs`, this function adds 
        random Gaussian noise to the corresponding spectrum in `self.x`.
        The standard deviation of the noise for each point is taken from 
        the third column (index 2) of the observation's data array.

        The resulting noisy predictions are stored in `self.noisy_x`.

        Parameters
        ----------
        self.x : dict
            Dictionary of simulated spectra, with one array per observation.
        self.obs : dict
            Dictionary of observational data arrays, where each array's third column
            (index 2) provides the standard deviation (uncertainty) for noise generation.

        Attributes
        ----------
        self.noisy_x : dict
            Dictionary of noisy model predictions created by adding Gaussian noise
            to `self.x` according to the uncertainties in `self.obs`.

        """
        self.noisy_x={}
        for key in self.obs.keys():
            self.noisy_x[key] = (self.augmented_x[key]+self.obs[key][:,2]
                                 *np.random.standard_normal(len(
                                     self.obs[key][:,1])))

    def do_preprocessing(self, x):
        """
        Applies a sequence of preprocessing functions to the input data.

        Args:
            x (torch.Tensor or numpy.ndarray): The input data to be preprocessed.

        Returns:
            torch.Tensor or numpy.ndarray: The preprocessed data after applying the 
            sequence of preprocessing functions. The output type matches the input type.

        Notes:
            - The `self.preprocessing` attribute should be a list of preprocessing 
              function names (as strings) that are defined in the `preprocessing` module.
            - If `self.preprocessing` is None, the input data `x` is returned unchanged.
            - If the input `x` is a `torch.Tensor`, it is converted to a NumPy array 
              before applying each preprocessing function, and then converted back to 
              a `torch.Tensor` after each function is applied.
        """

        if self.preprocessing is not None:
            xnorm = x
            for i in range(len(self.preprocessing)):
                preprocessing_fun = getattr(preprocessing, self.preprocessing[i])
                if isinstance(xnorm, torch.Tensor):
                    xnorm = preprocessing_fun(xnorm.cpu().numpy())
                    xnorm = torch.tensor(xnorm, dtype=torch.float32)
                else:
                    xnorm = preprocessing_fun(xnorm)
        else:
            xnorm = x
        return xnorm

    def do_postprocessing(self):
        """
        Perform post-processing on the data stored in `self.x` using the 
        specified post-processing functions defined in `self.parameters`.

        For each key in `self.parameters`, if the `post_process` attribute 
        is set, the corresponding post-processing function is dynamically 
        retrieved from `helpers.postprocessing` and applied to the 
        observations in `self.x`. The results are stored in `self.post_x`.

        Raises:
            AttributeError: If the specified post-processing function does 
            not exist in `helpers.postprocessing`.

            KeyError: If a required key is missing in `self.parameters` or 
            `self.x`.

        Note:
            - `self.parameters` is expected to be a dictionary where each 
              key maps to a dictionary containing a `post_process` key that 
              specifies the name of the post-processing function.
            - `self.x` is expected to be a dictionary of observations to be 
              processed.
            - `self.post_x` is a dictionary where the processed results 
              will be stored.
        """
        self.post_x={}
        par_idx=0
        for key in self.parameters.keys():
            if self.parameters[key]['post_processing']:
                if 'offset' in key:
                    postprocessing_function = postprocessing.offset
                    obs_key=int(key[6:])
                    self.post_x[0] = self.x[0]
                    self.post_x[obs_key] = postprocessing_function(self.nat_thetas[:, par_idx], 
                                            self.obs[obs_key][:,0], self.x[obs_key])
                elif 'scaling' in key:
                    postprocessing_function = postprocessing.scaling
                    obs_key=int(key[7:])
                    self.post_x[0] = self.x[0]
                    self.post_x[obs_key] = postprocessing_function(self.nat_thetas[:, par_idx], 
                                            self.obs[obs_key][:,0], self.x[obs_key])                                
                else:
                    postprocessing_function = getattr(postprocessing,
                                                    key)
                    for obs_key in self.x.keys():
                        self.post_x[obs_key] = postprocessing_function(self.nat_thetas[:, par_idx], 
                                                self.obs[obs_key][:,0], self.x[obs_key])
            else:
                for obs_key in self.x.keys():
                    self.post_x[obs_key] = self.x[obs_key]
            par_idx+=1 

    def psimulator(self, obs, parameters, **kwargs):
        """
        Run the simulator in parallel using n_threads, splitting the parameter
        grid.

        Parameters
        ----------
        obs : dict
            Observation dictionary.
        parameters : np.ndarray
            Full parameter grid (n_samples, n_params).
        n_threads : int
            Number of parallel process.
        **kwargs : dict
            Passed to each self.simulator() call.

        Returns
        -------
        combined_spectra : dict
            Dictionary matching obs keys with arrays of shape 
            (n_, n_points), combined across threads.
        """

        n_total = len(parameters)
        if n_total < self.n_threads:
            self.n_threads = n_total

        chunk_size = n_total // self.n_threads
        chunks = [parameters[i * chunk_size: (i + 1) * chunk_size] for i in range(self.n_threads)]
        
        # If there's a remainder, add it to the last chunk
        remainder = n_total % self.n_threads
        if remainder:
            chunks[-1] = np.vstack([chunks[-1], parameters[-remainder:]])

        args = [
            (self.simulator, self.obs, chunk, i, kwargs)
            for i, chunk in enumerate(chunks)
        ]

        with mp.get_context("spawn").Pool(processes=self.n_threads) as pool:
            spectra_parts = pool.map(_run_single_chunk, args)

        # Combine results across threads
        combined_spectra = {k: [] for k in obs}
        for partial in spectra_parts:
            for k in partial:
                combined_spectra[k].append(partial[k])

        # Concatenate chunks into final arrays
        for k in combined_spectra:
            combined_spectra[k] = np.vstack(combined_spectra[k])  # shape: (n_total, n_points)

        return combined_spectra

    def plot_corner(self, proposal_id=-1, n_samples=1000, IS_weights=False, **CORNER_KWARGS):
        """
        Generates a corner plot for the posterior samples of a proposal distribution.

        Parameters
        ----------
        proposal : object
            The proposal distribution object. It should have a `sample` method 
            that generates samples from the posterior.
        n_samples : int
            The number of samples to draw from the proposal distribution.
        **CORNER_KWARGS : dict, optional
            Additional keyword arguments to pass to the `corner.corner` function 
            for customizing the plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the corner plot.

        Notes
        -----
        - This function requires the `corner` library for generating the corner plot.
        - The proposal distribution should return samples in a 2D array of shape 
        (n_samples, n_parameters).
        """

        # Draw samples from the proposal distribution
        samples = self.proposals[proposal_id].sample((n_samples,)).detach().numpy()

        # Generate the corner plot
        fig = corner(helpers.convert_cube(samples, self.parameters), 
                        labels=list(self.parameters.keys()), 
                        weights=self.IS_weights if IS_weights else np.ones(n_samples),
                        **CORNER_KWARGS)

        # Show the plot
        plt.show()

        return fig

    def augment(self, n_augment=1):
        """
        Augments the spectra in self.post_x by creating multiple copies 
        with added Gaussian noise.

        Parameters
        ----------
        n_augment : int, optional
            Number of augmented copies to create for each spectrum. 
            Default is 5.

        Returns
        -------
        None
            The augmented spectra are stored in self.augmented_x.
        """
        self.augmented_x = {}
        self.augmented_thetas = []
        for key, spectrum in self.post_x.items():
            augmented_spectra = [spectrum.copy() for _ in range(n_augment)]
            self.augmented_x[key] = np.vstack(augmented_spectra)
        
        for _ in range(n_augment):
            self.augmented_thetas.append(self.thetas.copy())
        
        self.augmented_thetas = np.vstack(self.augmented_thetas)

def _run_single_chunk(args):
    """
    Standalone helper for multiprocessing — must be top-level (not nested).
    """
    simulator_func, obs, parameters_chunk, thread_idx, kwargs = args
    return simulator_func(obs, parameters_chunk, thread_idx, **kwargs)



