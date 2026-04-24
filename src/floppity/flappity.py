import os
import time
import multiprocessing as mp
from datetime import datetime, timezone

import cloudpickle as pickle
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube, Sobol

from floppity import helpers
from floppity.output import RetrievalOutput
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

    def get_obs(self, fnames, error_inflation=1, obs_names=None):
        """Read observation files and store them in ``self.obs``.

        Parameters
        ----------
        fnames : sequence or mapping
            Observation file paths. If a mapping is passed, its keys are used as
            observation names. Otherwise observations are keyed by position.
        error_inflation : float
            Factor multiplying the observational uncertainties when adding
            noise to simulations.
        obs_names : sequence, optional
            Explicit observation names to use when ``fnames`` is a sequence.
        """
        self.obs = self._read_observations(fnames, obs_names=obs_names)
        self.obs_sources = self._observation_sources(fnames, obs_names=obs_names)
        self.error_inflation = error_inflation

        if self.obs_type == "emis":
            for obs in self.obs.values():
                obs[obs <= 0] = 1e-12

        self.default_obs = self._concat_obs_values(self.obs).reshape(1, -1)

    @staticmethod
    def _read_observations(fnames, obs_names=None):
        if hasattr(fnames, "items"):
            if obs_names is not None:
                raise ValueError("obs_names cannot be used when fnames is a mapping.")
            return {
                key: Retrieval._load_observation_file(fname)
                for key, fname in fnames.items()
            }

        if obs_names is None:
            return {
                i: Retrieval._load_observation_file(fname)
                for i, fname in enumerate(fnames)
            }

        if len(obs_names) != len(fnames):
            raise ValueError("obs_names must have the same length as fnames.")
        return {
            name: Retrieval._load_observation_file(fname)
            for name, fname in zip(obs_names, fnames)
        }

    @staticmethod
    def _load_observation_file(fname):
        return np.atleast_2d(np.loadtxt(fname))

    @staticmethod
    def _observation_sources(fnames, obs_names=None):
        if hasattr(fnames, "items"):
            return {key: str(fname) for key, fname in fnames.items()}

        if obs_names is None:
            return {i: str(fname) for i, fname in enumerate(fnames)}

        return {name: str(fname) for name, fname in zip(obs_names, fnames)}

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
        simulator_kwargs = self._simulator_kwargs_for_round(simulator_kwargs, r)
        self._prepare_simulator_round_outputs(simulator_kwargs)

        if reuse_prior is not None and use_initial_sampling:
            self._load_or_extend_reused_prior(
                reuse_prior=reuse_prior,
                n_samples=n_samples,
                sample_prior_method=sample_prior_method,
                n_threads=n_threads,
                simulator_kwargs=simulator_kwargs,
                round_index=r,
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
        self.output = RetrievalOutput(output_dir)

        proposal = self._prepare_run(resume=resume, flow_kwargs=flow_kwargs)
        start_round = self.completed_rounds
        self.write_setup_log(
            output_dir,
            run_config={
                "n_threads": n_threads,
                "n_samples": n_samples,
                "n_samples_init": n_samples_init,
                "n_agg": n_agg,
                "resume": resume,
                "n_rounds": n_rounds,
                "n_aug": n_aug,
                "flow_kwargs": flow_kwargs,
                "training_kwargs": training_kwargs,
                "simulator_kwargs": simulator_kwargs,
                "output_dir": output_dir,
                "save_data": save_data,
                "sample_prior_method": sample_prior_method,
                "reuse_prior": reuse_prior,
                "start_round": start_round,
                "final_round": start_round + n_rounds,
            },
        )

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

            self._checkpoint(
                output_dir,
                self.output.pre_round_checkpoint_name(round_index),
            )
            self._save_round_data(output_dir, save_data, round_index)

            x_norm_tensor = self.do_preprocessing(x_tensor)
            self.train(theta_tensor, x_norm_tensor, proposal, **training_kwargs)
            self.get_posterior()

            proposal = self.posterior.set_default_x(self.default_obs_norm)
            self.proposals.append(self.posterior)
            self.completed_rounds = round_index + 1
            self.loss_val = self.inference._summary["best_validation_loss"]
            self._checkpoint(output_dir, "retrieval.pkl")

    def write_setup_log(self, output_dir, run_config=None, filename="retrieval_setup.json"):
        """Write a JSON setup log describing observations, parameters, and run options."""
        payload = self._setup_log_payload(run_config=run_config)
        return self._output_manager(output_dir).write_setup_log(payload, filename=filename)

    def _setup_log_payload(self, run_config=None):
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "retrieval": {
                "obs_type": self.obs_type,
                "simulator": self._callable_name(self.simulator),
                "preprocessing": self.preprocessing,
                "completed_rounds": self.completed_rounds,
                "error_inflation": getattr(self, "error_inflation", None),
            },
            "observations": self._observation_log_entries(),
            "parameters": self._json_safe(self.parameters),
            "run": self._json_safe(run_config or {}),
            "outputs": {
                "setup_log": "retrieval_setup.json",
                "completed_checkpoint": "retrieval.pkl",
                "pre_round_checkpoint_pattern": "retrieval_pre_round_<N>.pkl",
                "saved_data_pattern": "rounds/round_<NNN>/training_data.npz",
            },
        }

    def _observation_log_entries(self):
        entries = {}
        for key, obs in getattr(self, "obs", {}).items():
            entry = {
                "source": getattr(self, "obs_sources", {}).get(key),
                "shape": list(obs.shape),
            }
            if obs.size and obs.ndim == 2 and obs.shape[1] >= 3:
                entry.update(
                    {
                        "wavelength_min": float(np.nanmin(obs[:, 0])),
                        "wavelength_max": float(np.nanmax(obs[:, 0])),
                        "value_min": float(np.nanmin(obs[:, 1])),
                        "value_max": float(np.nanmax(obs[:, 1])),
                        "uncertainty_min": float(np.nanmin(obs[:, 2])),
                        "uncertainty_max": float(np.nanmax(obs[:, 2])),
                    }
                )
            entries[str(key)] = entry
        return entries

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if callable(value):
            return cls._callable_name(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _callable_name(value):
        module = getattr(value, "__module__", None)
        name = getattr(value, "__name__", None)
        if module and name:
            return f"{module}.{name}"
        return str(value)

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
        self._output_manager(output_dir).write_checkpoint(self, filename=filename)

    def _save_round_data(self, output_dir, save_data, round_index):
        if save_data:
            self._output_manager(output_dir).write_round_data(
                round_index=round_index,
                thetas=self.thetas,
                nat_thetas=getattr(self, "nat_thetas", None),
                spectra=self.x,
            )

    def _output_manager(self, output_dir):
        output = getattr(self, "output", None)
        if output is None or output.output_dir != output_dir:
            output = RetrievalOutput(output_dir)
            self.output = output
        return output

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
        round_index=None,
    ):
        print(f"Reusing prior data from {reuse_prior}")
        prior_data = RetrievalOutput.load_round_data(reuse_prior)

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
                sample_offset=reused_n,
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

    @staticmethod
    def _simulator_kwargs_for_round(simulator_kwargs, round_index):
        round_kwargs = dict(simulator_kwargs)
        round_kwargs["_round_index"] = round_index
        return round_kwargs

    def _prepare_simulator_round_outputs(self, simulator_kwargs):
        if getattr(self.simulator, "__name__", None) != "ARCiS":
            return
        if not simulator_kwargs.get("save_atmosphere", True):
            return
        if "output_dir" not in simulator_kwargs:
            return

        round_index = simulator_kwargs.get("_round_index", 0)
        filename = simulator_kwargs.get(
            "atmosphere_output",
            f"mixingratios_round_{round_index}.dat",
        )
        path = os.path.join(simulator_kwargs["output_dir"], filename)
        if os.path.exists(path):
            os.remove(path)

    def _run_simulator(self, n_threads, simulator_kwargs):
        xs = self._simulate_current_thetas(
            n_threads=n_threads,
            simulator_kwargs=simulator_kwargs,
        )
        self.get_x(xs)

    def _simulate_current_thetas(self, n_threads, simulator_kwargs, sample_offset=0):
        self.sim_pars = self._n_simulator_parameters()
        sim_thetas = self.nat_thetas[:, : self.sim_pars]
        simulator_kwargs = dict(simulator_kwargs)
        simulator_kwargs["_sample_offset"] = sample_offset
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
            if self._is_observation_specific_parameter(key, "offset"):
                obs_key = self._observation_key_from_parameter(key, "offset")
                self.post_x[obs_key] = postprocessing.offset(
                    par_values,
                    self.obs[obs_key][:, 0],
                    self.post_x[obs_key],
                )
            elif self._is_observation_specific_parameter(key, "scaling"):
                obs_key = self._observation_key_from_parameter(key, "scaling")
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

    @staticmethod
    def _is_observation_specific_parameter(parameter_name, prefix):
        return (
            parameter_name == prefix
            or parameter_name.startswith(f"{prefix}:")
            or parameter_name.startswith(f"{prefix}_")
            or parameter_name.startswith(prefix)
        )

    def _observation_key_from_parameter(self, parameter_name, prefix):
        suffix = parameter_name[len(prefix):]
        if suffix.startswith(":") or suffix.startswith("_"):
            suffix = suffix[1:]

        if suffix == "":
            raise ValueError(
                f"Parameter '{parameter_name}' must specify an observation key. "
                f"Use '{prefix}:<obs_key>' or '{prefix}_<obs_key>'."
            )

        candidates = [suffix]
        try:
            candidates.append(int(suffix))
        except ValueError:
            pass

        for candidate in candidates:
            if candidate in self.obs:
                return candidate

        available = ", ".join(repr(key) for key in self.obs)
        raise KeyError(
            f"Parameter '{parameter_name}' refers to observation key {suffix!r}, "
            f"but available keys are: {available}."
        )

    def psimulator(self, obs, parameters, **kwargs):
        """Run the simulator in parallel and combine chunked spectra."""
        n_total = len(parameters)
        if n_total < self.n_threads:
            self.n_threads = n_total

        chunks = self._parameter_chunks(parameters, self.n_threads)
        args = []
        sample_offset = kwargs.get("_sample_offset", 0)
        chunk_start = 0
        for i, chunk in enumerate(chunks):
            chunk_kwargs = dict(kwargs)
            chunk_kwargs["_sample_offset"] = sample_offset + chunk_start
            args.append((self.simulator, self.obs, chunk, i, chunk_kwargs))
            chunk_start += len(chunk)

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
