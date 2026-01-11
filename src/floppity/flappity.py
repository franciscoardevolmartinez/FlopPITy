import numpy as np
import torch
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from sbi.inference import SNPE_C
from sbi.neural_nets import embedding_nets
from typing import List, Optional, Tuple, Union
from floppity import helpers
from floppity import simulators
from sbi.utils import RestrictedPrior, get_density_thresholder
from floppity import postprocessing
import multiprocessing as mp
import cloudpickle as pickle
from corner import corner
from scipy.stats.qmc import LatinHypercube, Sobol
import os
import platform
import time
import copy

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

        print(f'Retrieval object created with simulator {simulator}. You can now add observations and parameters to fit.')

        self.simulator = simulator
        self.parameters = {}
        self.embedding= torch.nn.Identity()

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
        state.pop('augmented_x', None)
        state.pop('x', None)
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
    
    def get_obs(self, observations):#, err_inf=dict()):
        '''
        Read observation(s) to run retrievals on. Needs to be in the 
        format required by the simulator used.

        Parameters
        ----------
        observations : dict 
            dictionary containing files or arrays with the observations.
            The dictionary keys need to identify the observations and are 
            used to retrieve error inflations, offsets, scalings...

        Returns
        -------
        obs : dict
            dictionary with all the observations, keyed 0, 1, 2...
        '''
        self.obs={}

        assert isinstance(observations, dict), "Observations need to be passed in a dictionary"

        for key in observations.keys():
            item = observations[key]

            if isinstance(item, str):
                arr = np.loadtxt(item)
            # If it's already a numpy array, just use it
            elif isinstance(item, np.ndarray):
                arr = item
            else:
                raise TypeError(f"Unsupported type for fnames['{key}']: {type(item)}. Must be str or np.ndarray.")
            
            self.obs[key] = arr

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

    def create_prior(self):
        """
        Constructs a uniform prior distribution based on the min and max
        values of parameters in self.parameters.

        Raises:
            KeyError: If any parameter is missing 'min_value' or 
            'max_value'.
        """
        self.dims = len(self.parameters)
        low = np.empty((self.dims,))
        high=np.empty((self.dims,))
        for i, key in enumerate(self.parameters.keys()):
            low[i] = self.parameters[key]['min']
            high[i] = self.parameters[key]['max']
        low=torch.tensor(low.reshape(1,-1))
        high = torch.tensor(high.reshape(1,-1))

        self.prior = utils.BoxUniform(low=low,
                                      high = high)

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
        self.thetas = helpers.convert_cube(sampler.random(n=n_samples), self.parameters)
    
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

    def add_CNNembedding(self, mode='auto', embed_kwargs:Optional[dict]=None):
        
        '''
        If mode='auto', it builds the embedding automatically. 
        The following are used to build it:
        - The maximum number of convolutional layers is 4.
        - There is one linear layer, which downsizes by 4x at mostn

        - The output size is at least 5 x n_dims. If this is larger than the 
        number of wavelengths, the output size will be n_wvls, but in this case
        you should reconsider using an embedding.
        - The size of the pooling kernel is determined dynamically to achieve the desired 
        dimension reduction. 
        '''

        n_wvl = len(self.default_obs[0])
        n_dims = len(self.parameters)
        
        if mode=='auto':
            output_dim=min(5*n_dims, n_wvl)
            pool_size = 2
            n_layers = int(np.clip( np.log2(n_wvl/output_dim), 1, 4 ))
            channels=[2**(2+i) for i in range(n_layers)]
            linear_in=n_wvl/pool_size**n_layers
            linear_units=int( np.clip(output_dim, linear_in/4, linear_in) )
            print(f'1D CNN embedding with auto settings. Convolutional layers: {n_layers}. Linear units: {linear_units}.')
            embedding_net = embedding_nets.CNNEmbedding(input_shape=(n_wvl,),
                                                         num_conv_layers= n_layers,
                                                      out_channels_per_layer=channels,
                                                      num_linear_layers=1,
                                                      num_linear_units=linear_units, 
                                                      kernel_size=3,
                                                      output_dim=output_dim,
                                                      pool_kernel_size=pool_size)
        elif mode=='custom':
            print(f'1D CNN embedding with custom settings.')
            embedding_net = embedding_nets.CNNEmbedding(input_shape=(n_wvl,),
                                                         **embed_kwargs)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'auto' or 'custom'.")

        self.embedding=embedding_net

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
                                    use_batch_norm=use_batch_norm,
                                    embedding_net = self.embedding)
        
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
        self.default_obs_norm = self.x_transformer.transform(torch.tensor(self.default_obs,dtype=torch.float32 ))

        self.posterior=self.inference.build_posterior(
            self.posterior_estimator)
    
    def generate_training_data(self, proposal, r, n_samples, n_samples_init,
                           sample_prior_method, n_threads, simulator_kwargs, 
                           n_aug, reuse_prior=None):
        
        if (reuse_prior is not None) and (r == 0):
            n_samples_init = n_samples if n_samples_init is None else n_samples_init
            print(f"Reusing prior data from {reuse_prior}")
            prior_data = pickle.load(open(reuse_prior, 'rb'))

            #Are there enough samples to reuse?
            reused_n = min(len(prior_data['par']), n_samples_init)
            remaining_n = n_samples_init - reused_n

            reused_thetas = prior_data['par'][:reused_n]
            reused_x = {key: value[:reused_n] for key, value in prior_data['spec'].items()}

            if remaining_n > 0:
                print(f"Generating {remaining_n} additional samples.")
                self._sample_initial_thetas(sample_prior_method, remaining_n)
                self.sim_pars = sum(1 for key in self.parameters if not self.parameters[key]['post_processing'])

                # Simulate for additional parameters
                if n_threads == 1:
                    new_x = self.simulator(self.obs, self.thetas[:, :self.sim_pars], **simulator_kwargs)
                else:
                    new_x = self.psimulator(self.obs, self.thetas[:, :self.sim_pars], **simulator_kwargs)

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
            self.n_samples = all_thetas.shape[0]
            self.get_x(all_x)
        else:
            print('Generating training examples.')
            # Sample parameters
            if r == 0:
                self._sample_initial_thetas(sample_prior_method, n_samples_init)
            else:
                self.get_thetas(proposal, n_samples)

            # Determine how many parameters to simulate
            self.sim_pars = sum(1 for key in self.parameters if not self.parameters[key]['post_processing'])
            # Simulate
            if n_threads == 1:
                xs = self.simulator(self.obs, self.thetas[:, :self.sim_pars], **simulator_kwargs)
            else:
                xs = self.psimulator(self.obs, self.thetas[:, :self.sim_pars], **simulator_kwargs)

            self.get_x(xs)

        # Postprocessing
        self.do_postprocessing()
        self.augment(n_aug)
        self.add_noise()

        # Convert to tensors
        theta_tensor = torch.tensor(self.augmented_thetas, dtype=torch.float32)
        x_tensor = torch.tensor(np.concatenate(list(self.noisy_x.values()), axis=1), dtype=torch.float32)
        return theta_tensor, x_tensor

    def run(self, n_threads=1, n_samples=1024, n_samples_init=None, log_eps=1e-12,
                        resume=False, n_rounds=3, n_aug=1, flow_kwargs=dict(), 
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
            r0 = len(self.proposals)-1
        else: 
            print('Starting training...')
            self.IS_sampling_efficiency=[]
            self.logZ=[]
            self.dlogZ=[]
            self.create_prior()
            r0 = 0
            self.proposals=[self.prior]
            self.density_builder(**flow_kwargs)
            proposal=self.prior

        for r in range(r0, n_rounds):
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

            # this is a quick hack:
            if self.simulator==simulators.ARCiS_multiple:
                print('Making sure that T1>T2')
                bd1 = theta_tensor[:, :self.dims//2]
                bd2 = theta_tensor[:, self.dims//2:]

                # Determine which rows need to be swapped
                swap_mask = bd1[:, 0] <= bd2[:, 0]

                # Create an output tensor
                theta_new = theta_tensor.clone()

                # Swap rows where swap_mask is True
                theta_new[swap_mask, :self.dims//2] = bd2[swap_mask]
                theta_new[swap_mask, self.dims//2:] = bd1[swap_mask]

                theta_tensor=theta_new
                
            if r==0:
                self.x_transformer = helpers.DataTransformer(eps=log_eps)
                self.x_transformer.fit(x_tensor)

            # Save
            os.makedirs(output_dir, exist_ok=True)
            self.save(f'{output_dir}/retrieval.pkl')
            if save_data==True:
                pickle.dump(dict(par=self.thetas, spec=self.post_x), open(f'{output_dir}/data_{r}.pkl', 'wb'))

            # Do IS stuff
            # if IS:
            #     n_eff, sampling_efficiency, logZ, dlogZ= self.run_IS()
            #     self.IS_sampling_efficiency.append(sampling_efficiency)
            #     self.logZ.append(logZ)
            #     self.dlogZ.append(dlogZ)
            #     print(f'ε = {sampling_efficiency:.3g}')
            #     print(f'log(Z) = {logZ:.3g} +- {dlogZ:.3g}')

            x_std_tensor = self.x_transformer.transform(x_tensor)
                
            self.train(theta_tensor, x_std_tensor, proposal, **training_kwargs)
            
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
            self.noisy_x[key] = np.empty_like(self.augmented_x[key])
            for i in range(len(self.augmented_x[key])):
                self.noisy_x[key][i] = (self.augmented_x[key][i]+self.obs[key][:,2]
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
              key maps to a dictionary containing a `post_processing` key that 
              specifies the name of the post-processing function.
            - `self.x` is expected to be a dictionary of observations to be 
              processed.
            - `self.post_x` is a dictionary where the processed results 
              will be stored.
        """
        self.post_x=copy.deepcopy(self.x)

        for par_idx,key in enumerate(self.parameters.keys()):
            if self.parameters[key]['post_processing']:
                if key.startswith("offset"):
                    postprocessing_function = postprocessing.offset
                    obs_key = (key[len("offset_"):])
                    self.post_x[obs_key] = postprocessing_function(self.thetas[:, par_idx], 
                                            self.obs[obs_key][:,0], self.post_x[obs_key])
                elif key.startswith("scaling"):
                    postprocessing_function = postprocessing.scaling
                    obs_key = (key[len("scaling_"):])
                    self.post_x[obs_key] = postprocessing_function(self.thetas[:, par_idx], 
                                            self.obs[obs_key][:,0], self.post_x[obs_key])                                
                else:
                    postprocessing_function = getattr(postprocessing,
                                                    key)
                    for obs_key in self.x.keys():
                        self.post_x[obs_key] = postprocessing_function(self.thetas[:, par_idx], 
                                                self.obs[obs_key][:,0], self.post_x[obs_key])
            else:
                continue


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
        remainder = n_total % self.n_threads

        chunk_sizes = [chunk_size + 1 if i < remainder else chunk_size for i in range(self.n_threads)]


        chunks = []
        start = 0
        for size in chunk_sizes:
            end = start + size
            chunks.append(parameters[start:end])
            start = end

        args = [
            (self.simulator, self.obs, chunk, i, kwargs)
            for i, chunk in enumerate(chunks)
        ]

        mp_context = "spawn"

        with mp.get_context(mp_context).Pool(processes=self.n_threads) as pool:
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
        fig = corner(samples, 
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
            Default is 1.

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
    time.sleep(thread_idx * 0.05)
    return simulator_func(obs, parameters_chunk, thread_idx, **kwargs)



