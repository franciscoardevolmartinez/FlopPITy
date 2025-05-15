import numpy as np
import torch
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from sbi.inference import SNPE_C
from floppity import helpers
from sbi.utils import RestrictedPrior, get_density_thresholder
from floppity import postprocessing
import multiprocessing as mp
import cloudpickle as pickle
from corner import corner

class Retrieval():
    def __init__(self, simulator):
        """
        simulator (callable): A function or callable object that 
            simulates data based on the provided parameters. 
            The simulator must take as input the observation dictionary
            and an array of parameters of shape (n_samples, n_dims).
            Additionally, it must return a dictionary with the same keys
            as the observation.
        """
        self.simulator = simulator
        self.parameters = {}

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
        state.pop('noisy_x')
        state.pop('post_x')
        state.pop('x')
        state.pop('nat_thetas')
        state.pop('thetas')
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
                                        axis=0)[:,1].reshape(-1,)
         
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

    def get_thetas(self, proposal,n_samples):
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
                        z_score_theta='independent', z_score_x='independent'):
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
                                    z_score_x=z_score_x)
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

        # normed_thetas = 

        # self.concatenated=torch.tensor(np.concatenate(list(self.noisy_x.values()), 
                                            #  axis=1), dtype=torch.float32)

        self.inference.append_simulations(theta, x, 
                                          proposal=proposal)
        self.posterior_estimator = self.inference.train(show_train_summary=True,
                                force_first_round_loss=True,
                                retrain_from_scratch=False,
                                use_combined_loss=True,
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
        self.posterior=self.inference.build_posterior(
            self.posterior_estimator).set_default_x(self.default_obs)
    
    def run(self, n_threads=1, n_samples=100, n_samples_init=None, n_agg=1,
                        resume=False, n_rounds=10, n_aug=1, flow_kwargs=dict(), 
                        training_kwargs=dict(), simulator_kwargs=dict(),
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
            self.proposals=[]
            self.create_prior()
            self.density_builder(**flow_kwargs)
            proposal=self.prior
        for r in range(n_rounds):
            print(f"Round {r+1}")
            if len(self.proposals)==0:
                self.get_thetas(self.prior, n_samples_init)
            else:
                self.get_thetas(self.proposals[-1], n_samples)
            
            # Create a mask for parameters with post_process as False
            self.sim_pars=0
            for key in self.parameters.keys():
                if not self.parameters[key]['post_processing']:
                    self.sim_pars+=1
            if n_threads == 1:
                xs = self.simulator(self.obs, self.nat_thetas[:, :self.sim_pars], **simulator_kwargs)
            else:
                xs = self.psimulator(self.obs, 
                                self.nat_thetas[:, :self.sim_pars],
                                **simulator_kwargs
                                )
            self.get_x(xs)
            self.do_postprocessing()
            self.augment(n_aug)
            self.add_noise()
            self.train(torch.tensor(self.augmented_thetas, dtype=torch.float32), 
                    torch.tensor(np.concatenate(list(self.noisy_x.values()),
                    axis=1), 
                    dtype=torch.float32), proposal, **training_kwargs)
            self.get_posterior()
            proposal = self.posterior
            self.proposals.append(proposal)
            self.loss_val=self.inference._summary['best_validation_loss']
    
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
                postprocessing_function = getattr(postprocessing,
                                         key)
                if key=='offset' or key=='scaling':
                    self.post_x[0] = self.x[0]
                    for obs_key in list(self.x.keys())[1:]:
                        self.post_x[obs_key] = postprocessing_function(self.nat_thetas[:, par_idx], 
                                                self.obs[obs_key][:,0], self.x[obs_key])
                else:
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

    def plot_corner(self, proposal_id=-1, n_samples=1000, **CORNER_KWARGS):
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



