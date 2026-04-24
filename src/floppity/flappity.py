import os
import time
import multiprocessing as mp

import cloudpickle as pickle
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube, Sobol

from floppity import helpers
from floppity import postprocessing
from floppity import preprocessing


TRANSIENT_STATE = {
    "noisy_x",
    "post_x",
    "x",
    "nat_thetas",
    "thetas",
    "augmented_x",
    "augmented_thetas",
}


class Retrieval:
    """Simulation-based atmospheric retrieval driver."""

    def __init__(self, simulator, obs_type):
        """
        Parameters
        ----------
        simulator : callable
            Function that accepts ``obs`` and an array of natural parameters
            with shape ``(n_samples, n_dims)`` and returns simulated spectra
            keyed like ``obs``.
        obs_type : str
            Observation type, either ``"emis"`` or ``"trans"``.
        """
        self.simulator = simulator
        self.parameters = {}
        self.preprocessing = None
        self.obs_type = obs_type
        self.completed_rounds = 0

    def save(self, fname, **options):
        """Save this retrieval object with cloudpickle."""
        _ = options
        with open(fname, "wb") as file:
            pickle.dump(self, file)

    def __getstate__(self):
        """Drop large per-round arrays before pickling."""
        state = self.__dict__.copy()
        for key in TRANSIENT_STATE:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Restore a pickled retrieval object."""
        self.__dict__.update(state)
        self.completed_rounds = getattr(
            self, "completed_rounds", max(len(getattr(self, "proposals", [])) - 1, 0)
        )

    @classmethod
    def load(cls, fname):
        """Load a retrieval object from disk."""
        with open(fname, "rb") as file:
            return pickle.load(file)

    def get_obs(self, fnames, error_inflation=1):
        """Read observation files and store them in ``self.obs``."""
        self.obs = {i: np.loadtxt(fname) for i, fname in enumerate(fnames)}
        self.error_inflation = error_inflation

        if self.obs_type == "emis":
            for obs in self.obs.values():
                obs[obs <= 0] = 1e-12

        self.default_obs = self._concat_obs_values(self.obs).reshape(1, -1)

    def add_parameter(
        self,
        parname,
        min_value,
        max_value,
        log_scale=False,
        post_process=False,
        universal=True,
    ):
        """Add one retrieval parameter and its unit-cube transform metadata."""
        self.parameters[parname] = {
            "min": min_value,
            "max": max_value,
            "log": log_scale,
            "post_processing": post_process,
            "universal": universal,
        }

    def get_thetas(self, proposal, n_samples):
        """Sample unit-cube parameters from ``proposal``."""
        self.n_samples = n_samples
        thetas = proposal.sample((self.n_samples,))
        self.thetas = thetas.cpu().detach().numpy().reshape(-1, len(self.parameters))
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)

    def lhs_thetas(self, n_samples):
        """Generate Latin hypercube samples in the unit cube."""
        self.n_samples = n_samples
        sampler = LatinHypercube(d=len(self.parameters), optimization="random-cd")
        self.thetas = sampler.random(n=n_samples)
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)

    def sobol_thetas(self, n_samples):
        """Generate Sobol samples in the unit cube."""
        self.n_samples = n_samples
        sampler = Sobol(d=len(self.parameters), optimization="random-cd")
        self.thetas = sampler.random(n=n_samples)
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)

    def get_x(self, x):
        """Store simulator output using the expected retrieval shapes."""
        self.x = {
            key: value.reshape(self.n_samples, len(self.obs[key][:, 0]))
            for key, value in x.items()
        }

    def create_prior(self):
        """Construct a uniform unit-cube prior."""
        from sbi import utils as sbi_utils

        self.prior = sbi_utils.BoxUniform(
            low=torch.zeros(len(self.parameters)),
            high=torch.ones(len(self.parameters)),
        )

    def density_builder(
        self,
        flow="nsf",
        transforms=10,
        hidden=50,
        blocks=3,
        bins=8,
        dropout=0.05,
        z_score_theta="independent",
        z_score_x="independent",
        use_batch_norm=True,
    ):
        """Build the neural posterior estimator used by SNPE-C."""
        from sbi.inference import SNPE_C
        from sbi.neural_nets import posterior_nn

        self.density = posterior_nn(
            model=flow,
            num_transforms=transforms,
            hidden_features=hidden,
            num_blocks=blocks,
            num_bins=bins,
            dropout_probability=dropout,
            z_score_theta=z_score_theta,
            z_score_x=z_score_x,
            use_batch_norm=use_batch_norm,
        )
        self.inference = SNPE_C(prior=self.prior, density_estimator=self.density)

    def train(self, theta, x, proposal, **kwargs):
        """Append simulations and train the posterior estimator."""
        self.posterior_estimator = (
            self.inference.append_simulations(theta, x, proposal=proposal)
            .train(show_train_summary=True, **kwargs)
        )

    def get_posterior(self):
        """Build the posterior and cache the preprocessed default observation."""
        self.default_obs_norm = self.do_preprocessing(self.default_obs)
        self.posterior = self.inference.build_posterior(self.posterior_estimator)

    def generate_training_data(
        self,
        proposal,
        r,
        n_samples,
        n_samples_init,
        sample_prior_method,
        n_threads,
        simulator_kwargs,
        n_aug,
        reuse_prior=None,
        initial_round=None,
    ):
        """
        Generate one round of simulations and return training tensors.

        ``initial_round`` exists so resumed runs do not accidentally sample
        from the prior just because the local loop counter starts at zero.
        If omitted, the historic ``r == 0`` behavior is preserved.
        """
        use_initial_sampling = (r == 0) if initial_round is None else initial_round

        if reuse_prior is not None and use_initial_sampling:
            self._load_or_extend_reused_prior(
                reuse_prior=reuse_prior,
                n_samples=n_samples,
                sample_prior_method=sample_prior_method,
                n_threads=n_threads,
                simulator_kwargs=simulator_kwargs,
            )
        else:
            print("Generating training examples.")
            self._sample_round_thetas(
                proposal=proposal,
                n_samples=n_samples,
                n_samples_init=n_samples_init,
                sample_prior_method=sample_prior_method,
                initial_round=use_initial_sampling,
            )
            self._run_simulator(n_threads=n_threads, simulator_kwargs=simulator_kwargs)

        self.do_postprocessing()
        self.augment(n_aug)
        self.add_noise()
        self._clip_emission_arrays(self.noisy_x)

        theta_tensor = torch.tensor(self.augmented_thetas, dtype=torch.float32)
        x_tensor = torch.tensor(
            np.concatenate(list(self.noisy_x.values()), axis=1),
            dtype=torch.float32,
        )
        return theta_tensor, x_tensor

    def run(
        self,
        n_threads=1,
        n_samples=100,
        n_samples_init=None,
        n_agg=1,
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
    ):
        """
        Run SNPE-C retrieval rounds.

        Resumed runs continue from the last stored proposal and sample from
        that proposal immediately. Completed checkpoints are written after
        each successful round to ``retrieval.pkl``; pre-round checkpoints are
        also kept as ``retrieval_pre_round_<N>.pkl`` for crash recovery.
        """
        _ = n_agg
        flow_kwargs = {} if flow_kwargs is None else flow_kwargs
        training_kwargs = {} if training_kwargs is None else training_kwargs
        simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
        n_samples_init = n_samples if n_samples_init is None else n_samples_init

        self.n_threads = n_threads
        os.makedirs(output_dir, exist_ok=True)

        proposal = self._prepare_run(resume=resume, flow_kwargs=flow_kwargs)
        start_round = self.completed_rounds

        for local_round in range(n_rounds):
            round_index = start_round + local_round
            initial_round = not resume and round_index == 0
            print(f"Round {round_index + 1}")

            theta_tensor, x_tensor = self.generate_training_data(
                proposal=proposal,
                r=round_index,
                n_samples=n_samples,
                n_samples_init=n_samples_init,
                sample_prior_method=sample_prior_method,
                n_threads=n_threads,
                simulator_kwargs=simulator_kwargs,
                n_aug=n_aug,
                reuse_prior=reuse_prior if initial_round else None,
                initial_round=initial_round,
            )

            self._checkpoint(output_dir, f"retrieval_pre_round_{round_index + 1}.pkl")
            self._save_round_data(output_dir, save_data, round_index)

            x_norm_tensor = self.do_preprocessing(x_tensor)
            self.train(theta_tensor, x_norm_tensor, proposal, **training_kwargs)
            self.get_posterior()

            proposal = self.posterior.set_default_x(self.default_obs_norm)
            self.proposals.append(self.posterior)
            self.completed_rounds = round_index + 1
            self.loss_val = self.inference._summary["best_validation_loss"]
            self._checkpoint(output_dir, "retrieval.pkl")

    def _prepare_run(self, resume, flow_kwargs):
        if resume:
            print("Resuming training...")
            self._validate_resume_state()
            self.completed_rounds = max(
                getattr(self, "completed_rounds", 0),
                len(self.proposals) - 1,
            )
            return self.proposals[-1]

        print("Starting training...")
        self.completed_rounds = 0
        self.create_prior()
        self.proposals = [self.prior]
        self.density_builder(**flow_kwargs)
        return self.prior

    def _validate_resume_state(self):
        required = ["proposals", "prior", "inference", "posterior_estimator"]
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "Cannot resume retrieval; missing saved state: "
                + ", ".join(missing)
            )
        if len(self.proposals) == 0:
            raise RuntimeError("Cannot resume retrieval without at least one proposal.")

    def _checkpoint(self, output_dir, filename):
        self.save(os.path.join(output_dir, filename))

    def _save_round_data(self, output_dir, save_data, round_index):
        if save_data:
            path = os.path.join(output_dir, f"data_{round_index}.pkl")
            with open(path, "wb") as file:
                pickle.dump({"par": self.thetas, "spec": self.x}, file)

    def _sample_round_thetas(
        self,
        proposal,
        n_samples,
        n_samples_init,
        sample_prior_method,
        initial_round,
    ):
        if initial_round:
            self._sample_initial_thetas(sample_prior_method, n_samples_init)
        else:
            self.get_thetas(proposal, n_samples)

    def _load_or_extend_reused_prior(
        self,
        reuse_prior,
        n_samples,
        sample_prior_method,
        n_threads,
        simulator_kwargs,
    ):
        print(f"Reusing prior data from {reuse_prior}")
        with open(reuse_prior, "rb") as file:
            prior_data = pickle.load(file)

        reused_n = min(len(prior_data["par"]), n_samples)
        remaining_n = n_samples - reused_n
        reused_thetas = prior_data["par"][:reused_n]
        reused_x = {
            key: value[:reused_n]
            for key, value in prior_data["spec"].items()
        }

        if remaining_n > 0:
            print(f"Generating {remaining_n} additional samples.")
            self._sample_initial_thetas(sample_prior_method, remaining_n)
            new_x = self._simulate_current_thetas(
                n_threads=n_threads,
                simulator_kwargs=simulator_kwargs,
            )
            all_thetas = np.concatenate([reused_thetas, self.thetas], axis=0)
            all_x = {
                key: np.concatenate([reused_x[key], new_x[key]], axis=0)
                for key in reused_x
            }
        else:
            all_thetas = reused_thetas
            all_x = reused_x

        self.thetas = all_thetas
        self.nat_thetas = helpers.convert_cube(self.thetas, self.parameters)
        self.n_samples = all_thetas.shape[0]
        self.get_x(all_x)

    def _sample_initial_thetas(self, method, n_samples):
        if method == "random":
            self.get_thetas(self.prior, n_samples)
        elif method == "lhs":
            self.lhs_thetas(n_samples)
        elif method == "sobol":
            sobol_n = self._nearest_power_of_two(n_samples)
            if sobol_n != n_samples:
                print(
                    "n_samples must be a power of 2 for Sobol sampling. "
                    f"I will sample the prior with {sobol_n} samples and then "
                    f"go back to {n_samples} for the following rounds."
                )
            self.sobol_thetas(sobol_n)
        else:
            raise ValueError(
                "sample_prior_method must be one of 'random', 'lhs', or 'sobol'."
            )

    @staticmethod
    def _nearest_power_of_two(n_samples):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        return 2 ** int(np.round(np.log2(n_samples)))

    def _run_simulator(self, n_threads, simulator_kwargs):
        xs = self._simulate_current_thetas(
            n_threads=n_threads,
            simulator_kwargs=simulator_kwargs,
        )
        self.get_x(xs)

    def _simulate_current_thetas(self, n_threads, simulator_kwargs):
        self.sim_pars = self._n_simulator_parameters()
        sim_thetas = self.nat_thetas[:, : self.sim_pars]
        if n_threads == 1:
            return self.simulator(self.obs, sim_thetas, **simulator_kwargs)
        return self.psimulator(self.obs, sim_thetas, **simulator_kwargs)

    def _n_simulator_parameters(self):
        return sum(
            1
            for metadata in self.parameters.values()
            if not metadata["post_processing"]
        )

    @staticmethod
    def _concat_obs_values(obs):
        return np.concatenate(list(obs.values()), axis=0)[:, 1]

    def _clip_emission_arrays(self, arrays):
        if self.obs_type == "emis":
            for value in arrays.values():
                value[value <= 0] = 1e-12

    def add_noise(self):
        """Add Gaussian noise to the augmented spectra."""
        self.noisy_x = {}
        for key in self.obs:
            noise = (
                self.error_inflation
                * self.obs[key][:, 2]
                * np.random.standard_normal(len(self.obs[key][:, 1]))
            )
            self.noisy_x[key] = self.augmented_x[key] + noise

    def do_preprocessing(self, x):
        """Apply configured preprocessing functions to an array or tensor."""
        if self.preprocessing is None:
            return x

        xnorm = x
        for function_name in self.preprocessing:
            preprocessing_fun = getattr(preprocessing, function_name)
            if isinstance(xnorm, torch.Tensor):
                xnorm = preprocessing_fun(xnorm.cpu().numpy())
                xnorm = torch.tensor(xnorm, dtype=torch.float32)
            else:
                xnorm = preprocessing_fun(xnorm)
        return xnorm

    def do_postprocessing(self):
        """Apply configured post-processing parameters to simulated spectra."""
        self.post_x = {key: value.copy() for key, value in self.x.items()}

        for par_idx, key in enumerate(self.parameters):
            if not self.parameters[key]["post_processing"]:
                continue

            par_values = self.nat_thetas[:, par_idx]
            if key.startswith("offset"):
                obs_key = int(key[6:])
                self.post_x[obs_key] = postprocessing.offset(
                    par_values,
                    self.obs[obs_key][:, 0],
                    self.post_x[obs_key],
                )
            elif key.startswith("scaling"):
                obs_key = int(key[7:])
                self.post_x[obs_key] = postprocessing.scaling(
                    par_values,
                    self.obs[obs_key][:, 0],
                    self.post_x[obs_key],
                )
            else:
                postprocessing_function = getattr(postprocessing, key)
                for obs_key in self.post_x:
                    self.post_x[obs_key] = postprocessing_function(
                        par_values,
                        self.obs[obs_key][:, 0],
                        self.post_x[obs_key],
                    )

    def psimulator(self, obs, parameters, **kwargs):
        """Run the simulator in parallel and combine chunked spectra."""
        n_total = len(parameters)
        if n_total < self.n_threads:
            self.n_threads = n_total

        chunks = self._parameter_chunks(parameters, self.n_threads)
        args = [
            (self.simulator, self.obs, chunk, i, kwargs)
            for i, chunk in enumerate(chunks)
        ]

        with mp.get_context("spawn").Pool(processes=self.n_threads) as pool:
            spectra_parts = pool.map(_run_single_chunk, args)

        combined_spectra = {key: [] for key in obs}
        for partial in spectra_parts:
            for key, value in partial.items():
                combined_spectra[key].append(value)

        return {
            key: np.vstack(value)
            for key, value in combined_spectra.items()
        }

    @staticmethod
    def _parameter_chunks(parameters, n_threads):
        n_total = len(parameters)
        chunk_size = n_total // n_threads
        remainder = n_total % n_threads
        chunk_sizes = [
            chunk_size + 1 if i < remainder else chunk_size
            for i in range(n_threads)
        ]

        chunks = []
        start = 0
        for size in chunk_sizes:
            end = start + size
            chunks.append(parameters[start:end])
            start = end
        return chunks

    def plot_corner(self, proposal_id=-1, n_samples=1000, **CORNER_KWARGS):
        """Draw a corner plot from one stored proposal."""
        import matplotlib.pyplot as plt
        from corner import corner

        samples = self.proposals[proposal_id].sample((n_samples,)).detach().numpy()
        fig = corner(
            helpers.convert_cube(samples, self.parameters),
            labels=list(self.parameters.keys()),
            **CORNER_KWARGS,
        )
        plt.show()
        return fig

    def augment(self, n_augment=1):
        """Repeat current spectra and theta samples for noise augmentation."""
        self.augmented_x = {
            key: np.vstack([spectrum.copy() for _ in range(n_augment)])
            for key, spectrum in self.post_x.items()
        }
        self.augmented_thetas = np.vstack(
            [self.thetas.copy() for _ in range(n_augment)]
        )


def _run_single_chunk(args):
    """Top-level multiprocessing helper."""
    simulator_func, obs, parameters_chunk, thread_idx, kwargs = args
    time.sleep(thread_idx * 0.05)
    return simulator_func(obs, parameters_chunk, thread_idx, **kwargs)
