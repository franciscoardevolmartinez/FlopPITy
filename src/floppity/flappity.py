import os
import sys
import time
import copy
import importlib
import json
import multiprocessing as mp
import warnings
from datetime import datetime, timezone

import cloudpickle as pickle
import numpy as np
import torch
from scipy.stats.qmc import Sobol
from tqdm import tqdm

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
    "theta_sources",
    "best_fit_radius_scales",
    "best_fit_radii",
    "augmented_x",
    "augmented_thetas",
}


class Retrieval:
    """Simulation-based atmospheric retrieval driver."""

    def __init__(self, simulator, obs_type="emis", pca_components=None, do_pca=False):
        """
        Parameters
        ----------
        simulator : callable
            Function that accepts ``obs`` and an array of natural parameters
            with shape ``(n_samples, n_dims)`` and returns simulated spectra
            keyed like ``obs``.
        obs_type : str, optional
            Observation type, either ``"emis"`` or ``"trans"``.
            Defaults to ``"emis"``.
        pca_components : int, optional
            Number of PCA components to train on preprocessed spectra. PCA is
            disabled when omitted.
        do_pca : bool, optional
            Legacy switch for enabling PCA. Prefer ``pca_components`` for new
            code.
        """
        self.simulator = simulator
        self.parameters = {}
        self.preprocessing = ["log_standardize"]
        self.preprocessing_transformers = {}
        self.obs_type = self._normalize_obs_type(obs_type)
        self.completed_rounds = 0
        self.pca_components = (
            100 if do_pca and pca_components is None else pca_components
        )
        self.pca = None
        self.do_pca = self.pca_components is not None

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
        self.obs_type = self._normalize_obs_type(getattr(self, "obs_type", "emis"))
        self.completed_rounds = getattr(
            self, "completed_rounds", max(len(getattr(self, "proposals", [])) - 1, 0)
        )
        self.pca_components = getattr(self, "pca_components", None)
        self.pca = getattr(self, "pca", None)
        self.do_pca = getattr(self, "do_pca", self.pca_components is not None)
        self.preprocessing_transformers = getattr(
            self, "preprocessing_transformers", {}
        )

    @classmethod
    def load(cls, fname):
        """Load a retrieval object from disk."""
        with open(fname, "rb") as file:
            return pickle.load(file)

    @classmethod
    def from_setup(cls, setup_path, simulator=None, run_overrides=None):
        """Rebuild a retrieval object from a ``retrieval_setup.json`` file.

        The setup log stores importable simulator names as strings. If that
        simulator cannot be imported, pass ``simulator=...`` explicitly.
        Missing paths are reported as warnings so callers can edit the setup or
        provide overrides before running.
        """
        setup = cls._load_setup_payload(setup_path)
        run_config = cls._setup_run_config(setup, run_overrides=run_overrides)
        simulator = simulator or cls._import_setup_simulator(setup)
        retrieval = cls(
            simulator,
            obs_type=setup.get("retrieval", {}).get("obs_type") or "emis",
            pca_components=setup.get("retrieval", {}).get("pca_components"),
        )
        retrieval.setup_payload = setup
        retrieval.setup_path = os.fspath(setup_path)
        retrieval.setup_run_config = run_config
        retrieval.parameters = copy.deepcopy(setup.get("parameters", {}))
        retrieval.preprocessing = setup.get("retrieval", {}).get("preprocessing")

        observation_paths = cls._setup_observation_paths(setup)
        cls._warn_missing_setup_paths(observation_paths, "observation")
        retrieval.get_obs(
            observation_paths,
            error_inflation=setup.get("retrieval", {}).get("error_inflation", 1),
        )
        cls._warn_missing_paths_in_value(
            run_config.get("simulator_kwargs", {}),
            context="simulator_kwargs",
        )
        return retrieval

    @staticmethod
    def _normalize_obs_type(obs_type):
        aliases = {
            "emis": "emis",
            "emission": "emis",
            "trans": "trans",
            "transmission": "trans",
        }
        normalized = aliases.get(str(obs_type).lower())
        if normalized is None:
            raise ValueError(
                "obs_type must be one of 'emis', 'emission', 'trans', or "
                "'transmission'."
            )
        return normalized

    @classmethod
    def run_from_setup(
        cls,
        setup_path,
        simulator=None,
        run_overrides=None,
        **overrides,
    ):
        """Rebuild and run a retrieval from a setup log.

        ``run_overrides`` and keyword arguments override values stored in the
        setup log. Nested dictionaries such as ``simulator_kwargs`` are merged.
        """
        run_overrides = cls._merge_setup_overrides(run_overrides, overrides)
        retrieval = cls.from_setup(
            setup_path,
            simulator=simulator,
            run_overrides=run_overrides,
        )
        run_config = dict(retrieval.setup_run_config)
        run_config.pop("start_round", None)
        run_config.pop("final_round", None)
        retrieval.run(**run_config)
        return retrieval

    @staticmethod
    def _load_setup_payload(setup_path):
        with open(setup_path) as file:
            return json.load(file)

    @classmethod
    def _setup_run_config(cls, setup, run_overrides=None):
        run_config = copy.deepcopy(setup.get("run", {}))
        run_config.pop("start_round", None)
        run_config.pop("final_round", None)
        retrieval_config = setup.get("retrieval", {})
        for key in ("fit_radius", "radius_bounds", "radius_reference"):
            if key in retrieval_config and key not in run_config:
                run_config[key] = retrieval_config[key]
        return cls._merge_setup_overrides(run_config, run_overrides or {})

    @staticmethod
    def _merge_setup_overrides(base, overrides):
        merged = copy.deepcopy(base or {})
        for key, value in (overrides or {}).items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = Retrieval._merge_setup_overrides(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def _import_setup_simulator(cls, setup):
        simulator_name = setup.get("retrieval", {}).get("simulator")
        simulator = cls._import_dotted_name(simulator_name)
        if simulator is None:
            warnings.warn(
                f"Could not import simulator {simulator_name!r} from setup log. "
                "Pass simulator=... to Retrieval.from_setup(...) or "
                "Retrieval.run_from_setup(...).",
                RuntimeWarning,
                stacklevel=2,
            )
            return _UnavailableSetupSimulator(simulator_name)
        return simulator

    @staticmethod
    def _import_dotted_name(dotted_name):
        if not dotted_name or "." not in str(dotted_name):
            return None
        module_name, name = str(dotted_name).rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
        return getattr(module, name, None)

    @staticmethod
    def _setup_observation_paths(setup):
        observations = setup.get("observations", {})
        return {
            _deserialize_setup_key(key): entry.get("source")
            for key, entry in observations.items()
            if entry.get("source") is not None
        }

    @classmethod
    def _warn_missing_setup_paths(cls, paths, context):
        for key, path in paths.items():
            if path and not os.path.exists(path):
                warnings.warn(
                    f"Setup {context} path for {key!r} does not exist: {path}",
                    RuntimeWarning,
                    stacklevel=3,
                )

    @classmethod
    def _warn_missing_paths_in_value(cls, value, context):
        for path in cls._path_like_strings(value):
            if not os.path.exists(path):
                warnings.warn(
                    f"Setup path in {context} does not exist: {path}",
                    RuntimeWarning,
                    stacklevel=3,
                )

    @classmethod
    def _path_like_strings(cls, value):
        if isinstance(value, dict):
            paths = []
            for item in value.values():
                paths.extend(cls._path_like_strings(item))
            return paths
        if isinstance(value, (list, tuple)):
            paths = []
            for item in value:
                paths.extend(cls._path_like_strings(item))
            return paths
        if not isinstance(value, str):
            return []
        expanded = os.path.expanduser(value)
        if (
            os.path.isabs(expanded)
            or os.path.sep in expanded
            or (os.path.altsep and os.path.altsep in expanded)
        ):
            return [expanded]
        return []

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
        """Add one retrieval parameter and its prior bounds."""
        self.parameters[parname] = {
            "min": min_value,
            "max": max_value,
            "log": log_scale,
            "post_processing": post_process,
            "universal": universal,
        }

    def get_thetas(self, proposal, n_samples):
        """Sample natural-parameter values from ``proposal``."""
        self.n_samples = n_samples
        thetas = proposal.sample((self.n_samples,))
        self.thetas = thetas.cpu().detach().numpy().reshape(-1, len(self.parameters))
        self.nat_thetas = self.thetas
        self.theta_sources = self._sample_sources_from_proposal(proposal, self.n_samples)

    def sobol_thetas(self, n_samples):
        """Generate Sobol prior samples and immediately scale to natural units."""
        self.n_samples = n_samples
        sampler = Sobol(d=len(self.parameters), optimization="random-cd")
        unit_thetas = sampler.random(n=n_samples)
        self.thetas = helpers.convert_cube(unit_thetas, self.parameters)
        self.nat_thetas = self.thetas
        self.theta_sources = self._sample_sources("prior", self.n_samples)

    def get_x(self, x):
        """Store simulator output using the expected retrieval shapes."""
        self.x = {
            key: value.reshape(self.n_samples, len(self.obs[key][:, 0]))
            for key, value in x.items()
        }

    def create_prior(self):
        """Construct a uniform prior in natural parameter units."""
        from sbi import utils as sbi_utils

        self.dims = len(self.parameters)
        low = np.empty((self.dims,))
        high = np.empty((self.dims,))
        for i, metadata in enumerate(self.parameters.values()):
            low[i] = metadata["min"]
            high[i] = metadata["max"]

        self.prior = sbi_utils.BoxUniform(
            low=torch.as_tensor(low.reshape(1, -1), dtype=torch.float32),
            high=torch.as_tensor(high.reshape(1, -1), dtype=torch.float32),
        )

    def density_builder(
        self,
        flow="nsf",
        transforms=8,
        hidden=64,
        blocks=2,
        bins=5,
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
        n_samples=2048,
        n_samples_init=None,
        n_agg=1,
        resume=False,
        n_rounds=5,
        n_aug=1,
        flow_kwargs=None,
        training_kwargs=None,
        simulator_kwargs=None,
        output_dir="output_FlopPITy",
        save_data=False,
        sample_prior_method="sobol",
        reuse_prior=None,
        alpha=0,
        pca_components=None,
        n_pca=None,
        fit_radius=False,
        radius_bounds=None,
        radius_reference=1.0,
        save_posterior_samples=False,
        save_sbi_data=None,
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
        save_sbi_data = save_data if save_sbi_data is None else bool(save_sbi_data)
        simulator_kwargs = self._freeze_arcis_input_for_run(simulator_kwargs)
        n_samples_init = n_samples if n_samples_init is None else n_samples_init
        self._configure_radius_fit(
            fit_radius=fit_radius,
            radius_bounds=radius_bounds,
            radius_reference=radius_reference,
        )
        self._configure_pca(
            pca_components=pca_components,
            n_pca=n_pca,
            resume=resume,
        )

        self.n_threads = n_threads
        self.alpha = alpha
        self.output = RetrievalOutput(output_dir)
        self.cloned_observation_paths = self.output.clone_observation_files(
            getattr(self, "obs_sources", {})
        )

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
                "alpha": alpha,
                "pca_components": self.pca_components,
                "fit_radius": self.fit_radius,
                "radius_bounds": self.radius_bounds,
                "radius_reference": self.radius_reference,
                "save_posterior_samples": save_posterior_samples,
                "save_sbi_data": save_sbi_data,
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

            pre_round_checkpoint = self.output.pre_round_checkpoint_name(round_index)
            self._checkpoint(output_dir, pre_round_checkpoint)
            self.output.cleanup_pre_round_checkpoints(keep_filename=pre_round_checkpoint)
            self._save_round_data(output_dir, save_data, round_index)

            x_norm_tensor = self.do_preprocessing(
                x_tensor,
                fit_preprocessing=round_index == 0 and not resume,
                fit_pca=self._pca_enabled() and not self._pca_is_fitted(),
            )
            self.default_obs_norm = self.do_preprocessing(self.default_obs)
            self._save_sbi_data(
                output_dir,
                save_sbi_data,
                round_index,
                theta_tensor,
                x_norm_tensor,
                self.default_obs_norm,
            )
            self.train(theta_tensor, x_norm_tensor, proposal, **training_kwargs)
            self.get_posterior()

            posterior = self.posterior.set_default_x(self.default_obs_norm)
            self.posteriors.append(posterior)
            proposal = self._next_proposal(posterior, alpha)
            self.proposals.append(proposal)
            if save_posterior_samples:
                self._save_posterior_samples(
                    output_dir=output_dir,
                    posterior=posterior,
                    round_index=round_index,
                )
            self.completed_rounds = round_index + 1
            self.loss_val = self.inference._summary["best_validation_loss"]
            self._checkpoint(output_dir, "retrieval.pkl")

    def run_ensemble(
        self,
        n_members=3,
        output_dir="output_FlopPITy_ensemble",
        member_prefix="member",
        aggregate=True,
        resume=False,
        add_members=False,
        extend_rounds=False,
        **run_kwargs,
    ):
        """Run the same retrieval several times and aggregate file outputs.

        The first ensemble member generates the prior simulations with
        ``save_data=True``. Later members reuse that first round through
        ``reuse_prior`` so stochastic posterior training is repeated without
        recomputing the prior grid. Use ``resume=True, add_members=True`` to
        append new members, or ``resume=True, extend_rounds=True`` to continue
        every existing member for additional SNPE rounds.
        """
        n_members = int(n_members)
        if n_members <= 0:
            raise ValueError("n_members must be a positive integer.")
        if add_members and extend_rounds:
            raise ValueError("Use either add_members=True or extend_rounds=True, not both.")
        if extend_rounds and not resume:
            raise ValueError("extend_rounds=True requires resume=True.")

        os.makedirs(output_dir, exist_ok=True)
        run_kwargs = dict(run_kwargs)
        run_kwargs["save_data"] = True
        n_rounds = int(run_kwargs.get("n_rounds", 5))
        if n_rounds <= 0:
            raise ValueError("run_ensemble requires n_rounds > 0.")

        existing_member_dirs = self._ensemble_member_dirs(output_dir, member_prefix)
        if existing_member_dirs and not resume:
            raise FileExistsError(
                f"Ensemble output {output_dir!r} already contains member "
                "directories. Use resume=True to add members or choose a new "
                "output_dir."
            )

        extended_member_dirs = []
        if extend_rounds:
            if not existing_member_dirs:
                raise FileNotFoundError(
                    "Cannot extend ensemble rounds because no existing member "
                    f"directories were found in {output_dir!r}."
                )
            for member_dir in existing_member_dirs:
                member_number = self._ensemble_member_number(member_dir, member_prefix)
                member = self._load_ensemble_member(member_dir)
                member_kwargs = self._ensemble_member_resume_kwargs(
                    run_kwargs=run_kwargs,
                    member_dir=member_dir,
                )
                print(f"Extending ensemble member {member_number}")
                member.run(**member_kwargs)
                extended_member_dirs.append(member_dir)

            member_dirs = self._ensemble_member_dirs(output_dir, member_prefix)
            aggregate_rounds = self._ensemble_total_rounds(member_dirs, n_rounds)
            summary = {
                "n_members": len(member_dirs),
                "new_members": [],
                "extended_members": extended_member_dirs,
                "output_dir": output_dir,
                "member_dirs": member_dirs,
                "reuse_prior": self._existing_ensemble_prior_round_path(
                    output_dir,
                    member_prefix,
                ),
                "resume": resume,
                "add_members": add_members,
                "extend_rounds": extend_rounds,
                "aggregated": {},
            }
            if aggregate:
                summary["aggregated"] = self._aggregate_ensemble_outputs(
                    output_dir=output_dir,
                    member_dirs=member_dirs,
                    n_rounds=aggregate_rounds,
                )
            self.ensemble_summary = summary
            return summary

        start_number, stop_number = self._ensemble_member_number_range(
            n_members=n_members,
            existing_member_dirs=existing_member_dirs,
            resume=resume,
            add_members=add_members,
            member_prefix=member_prefix,
        )
        prior_round_path = self._ensemble_prior_round_path(
            output_dir=output_dir,
            member_prefix=member_prefix,
            existing_member_dirs=existing_member_dirs,
            start_number=start_number,
        )
        new_member_dirs = []
        for member_number in range(start_number, stop_number + 1):
            member_dir = os.path.join(
                output_dir,
                f"{member_prefix}_{member_number:03d}",
            )
            new_member_dirs.append(member_dir)
            member = self._fresh_ensemble_member()
            member_kwargs = self._ensemble_member_run_kwargs(
                run_kwargs=run_kwargs,
                member_dir=member_dir,
                reuse_prior=member_number > 1,
                prior_round_path=prior_round_path,
            )

            print(f"Starting ensemble member {member_number}")
            member.run(**member_kwargs)
            if member_number == 1:
                prior_round_path = RetrievalOutput(member_dir).round_data_path(0)

        member_dirs = self._ensemble_member_dirs(output_dir, member_prefix)
        summary = {
            "n_members": len(member_dirs),
            "new_members": new_member_dirs,
            "extended_members": [],
            "output_dir": output_dir,
            "member_dirs": member_dirs,
            "reuse_prior": prior_round_path,
            "resume": resume,
            "add_members": add_members,
            "extend_rounds": extend_rounds,
            "aggregated": {},
        }
        if aggregate:
            aggregate_rounds = self._ensemble_total_rounds(member_dirs, n_rounds)
            summary["aggregated"] = self._aggregate_ensemble_outputs(
                output_dir=output_dir,
                member_dirs=member_dirs,
                n_rounds=aggregate_rounds,
            )
        self.ensemble_summary = summary
        return summary

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
                "pca_components": getattr(self, "pca_components", None),
                "pca_fitted": self._pca_is_fitted(),
                "fit_radius": getattr(self, "fit_radius", False),
                "radius_bounds": getattr(self, "radius_bounds", None),
                "radius_reference": getattr(self, "radius_reference", 1.0),
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
                "sbi_data_pattern": "rounds/round_<NNN>/sbi_data.npz",
                "posterior_samples_pattern": "posterior_samples_round_<N>.txt",
                "observations_dir": "observations",
            },
        }

    def _observation_log_entries(self):
        entries = {}
        for key, obs in getattr(self, "obs", {}).items():
            entry = {
                "source": getattr(self, "obs_sources", {}).get(key),
                "cloned_source": getattr(self, "cloned_observation_paths", {}).get(key),
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
        if name:
            return name
        return str(value)

    def _prepare_run(self, resume, flow_kwargs):
        if resume:
            print("Resuming training...")
            self._validate_resume_state()
            self.completed_rounds = max(
                getattr(self, "completed_rounds", 0),
                len(self.proposals) - 1,
            )
            if not hasattr(self, "posteriors"):
                self.posteriors = []
            return self.proposals[-1]

        print("Starting training...")
        self.completed_rounds = 0
        self.create_prior()
        self.proposals = [self.prior]
        self.posteriors = []
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

    @staticmethod
    def _validate_alpha(alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1.")

    def _configure_radius_fit(
        self,
        fit_radius=False,
        radius_bounds=None,
        radius_reference=1.0,
    ):
        self.fit_radius = bool(fit_radius)
        self.radius_reference = float(radius_reference)
        if self.radius_reference <= 0:
            raise ValueError("radius_reference must be positive.")

        if radius_bounds is None:
            self.radius_bounds = None
        else:
            if len(radius_bounds) != 2:
                raise ValueError("radius_bounds must be a two-value sequence.")
            lower, upper = (float(radius_bounds[0]), float(radius_bounds[1]))
            if lower < 0 or upper <= lower:
                raise ValueError(
                    "radius_bounds must satisfy 0 <= lower < upper."
                )
            self.radius_bounds = (lower, upper)

        if self.fit_radius and self.obs_type != "emis":
            raise ValueError("fit_radius=True is only supported for emission spectra.")
        if self.fit_radius:
            radius_parameters = self._radius_like_sampled_parameters()
            if radius_parameters:
                warnings.warn(
                    "fit_radius=True expects spectra computed at a fixed "
                    "reference radius, but these sampled parameters look "
                    "radius-like: "
                    + ", ".join(radius_parameters)
                    + ". Remove them from the retrieval parameters or make "
                    "sure they are not used by the simulator; otherwise "
                    "FlopPITy will fit an additional radius scale on top of "
                    "the simulator's radius.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not self.fit_radius:
            for attr in ("best_fit_radius_scales", "best_fit_radii"):
                if hasattr(self, attr):
                    delattr(self, attr)

    def _radius_like_sampled_parameters(self):
        return [
            name
            for name, metadata in self.parameters.items()
            if not metadata.get("post_processing", False)
            and self._is_radius_like_parameter_name(name)
        ]

    @staticmethod
    def _is_radius_like_parameter_name(name):
        normalized = str(name).lower().replace("-", "_")
        parts = [part for part in normalized.split("_") if part]
        return normalized in {"r", "rad", "radius"} or "radius" in parts

    def _configure_pca(self, pca_components=None, n_pca=None, resume=False):
        if n_pca is not None:
            if pca_components is not None and int(pca_components) != int(n_pca):
                raise ValueError("pca_components and n_pca specify different values.")
            pca_components = n_pca

        if pca_components is not None:
            pca_components = int(pca_components)
            if pca_components <= 0:
                raise ValueError("pca_components must be a positive integer.")
            fitted_components = getattr(self.pca, "requested_components", None)
            if resume and self._pca_is_fitted() and pca_components != fitted_components:
                raise ValueError(
                    "Cannot change pca_components after PCA has been fitted."
                )
            self.pca_components = pca_components

        self.do_pca = self.pca_components is not None
        if not resume and self._pca_enabled():
            self.pca = None
        if resume and self._pca_enabled() and not self._pca_is_fitted():
            raise RuntimeError(
                "Cannot resume with PCA enabled because the loaded retrieval "
                "does not contain a fitted PCA transformer."
            )

    def _pca_enabled(self):
        return getattr(self, "pca_components", None) is not None

    def _pca_is_fitted(self):
        pca = getattr(self, "pca", None)
        return pca is not None and getattr(pca, "fitted", False)

    def _next_proposal(self, posterior, alpha):
        _ = alpha
        return posterior

    def _sample_sources_from_proposal(self, proposal, n_samples):
        if hasattr(proposal, "last_sample_sources"):
            return proposal.last_sample_sources.copy()
        if proposal is getattr(self, "prior", None):
            return self._sample_sources("prior", n_samples)
        return self._sample_sources("proposal", n_samples)

    @staticmethod
    def _sample_sources(source, n_samples):
        return np.full(n_samples, source, dtype="<U8")

    def _checkpoint(self, output_dir, filename):
        self._output_manager(output_dir).write_checkpoint(self, filename=filename)

    def _save_round_data(self, output_dir, save_data, round_index):
        if save_data:
            self._output_manager(output_dir).write_round_data(
                round_index=round_index,
                thetas=self.thetas,
                spectra=self.x,
                sample_sources=getattr(self, "theta_sources", None),
                processed_spectra=getattr(self, "post_x", None),
                fitted_radii=getattr(self, "best_fit_radii", None),
            )

    def _save_sbi_data(
        self,
        output_dir,
        save_sbi_data,
        round_index,
        theta_tensor,
        x_norm_tensor,
        default_obs_norm,
    ):
        if not save_sbi_data:
            return
        self._output_manager(output_dir).write_sbi_data(
            round_index=round_index,
            theta=self._tensor_to_numpy(theta_tensor),
            x=self._tensor_to_numpy(x_norm_tensor),
            default_x=self._tensor_to_numpy(default_obs_norm),
            parameter_names=self.parameters.keys(),
        )

    @staticmethod
    def _tensor_to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _output_manager(self, output_dir):
        output = getattr(self, "output", None)
        if output is None or output.output_dir != output_dir:
            output = RetrievalOutput(output_dir)
            self.output = output
        return output

    def _fresh_ensemble_member(self):
        member = copy.deepcopy(self)
        for key in TRANSIENT_STATE:
            if hasattr(member, key):
                delattr(member, key)
        for key in (
            "prior",
            "proposals",
            "posteriors",
            "posterior",
            "posterior_estimator",
            "inference",
            "density",
            "loss_val",
            "output",
            "ensemble_summary",
        ):
            if hasattr(member, key):
                delattr(member, key)
        member.completed_rounds = 0
        return member

    def _load_ensemble_member(self, member_dir):
        checkpoint = os.path.join(member_dir, "retrieval.pkl")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                "Cannot extend ensemble member because its completed checkpoint "
                f"was not found: {checkpoint}"
            )
        return self.load(checkpoint)

    def _ensemble_member_dirs(self, output_dir, member_prefix):
        paths = []
        prefix = f"{member_prefix}_"
        if not os.path.exists(output_dir):
            return paths
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if not os.path.isdir(path) or not name.startswith(prefix):
                continue
            if self._ensemble_member_number(path, member_prefix) is not None:
                paths.append(path)
        return sorted(
            paths,
            key=lambda path: self._ensemble_member_number(path, member_prefix),
        )

    @staticmethod
    def _ensemble_member_number(member_dir, member_prefix):
        name = os.path.basename(os.fspath(member_dir))
        prefix = f"{member_prefix}_"
        if not name.startswith(prefix):
            return None
        try:
            return int(name[len(prefix):])
        except ValueError:
            return None

    def _ensemble_member_number_range(
        self,
        n_members,
        existing_member_dirs,
        resume,
        add_members,
        member_prefix,
    ):
        if not existing_member_dirs:
            return 1, n_members

        last_member = max(
            self._ensemble_member_number(path, member_prefix)
            for path in existing_member_dirs
        )
        if add_members:
            return last_member + 1, last_member + n_members
        if n_members <= len(existing_member_dirs):
            return last_member + 1, last_member
        return last_member + 1, n_members

    def _ensemble_prior_round_path(
        self,
        output_dir,
        member_prefix,
        existing_member_dirs,
        start_number,
    ):
        if start_number == 1:
            return None

        first_member = os.path.join(output_dir, f"{member_prefix}_001")
        prior_round_path = RetrievalOutput(first_member).round_data_path(0)
        if not os.path.exists(prior_round_path):
            raise FileNotFoundError(
                "Cannot add ensemble members because the original prior round "
                f"was not found: {prior_round_path}"
            )
        if not existing_member_dirs:
            raise FileNotFoundError(
                "Cannot start a resumed ensemble after member 1 without existing "
                "member directories."
            )
        return prior_round_path

    def _existing_ensemble_prior_round_path(self, output_dir, member_prefix):
        first_member = os.path.join(output_dir, f"{member_prefix}_001")
        prior_round_path = RetrievalOutput(first_member).round_data_path(0)
        if os.path.exists(prior_round_path):
            return prior_round_path
        return None

    def _ensemble_member_run_kwargs(
        self,
        run_kwargs,
        member_dir,
        reuse_prior,
        prior_round_path,
    ):
        member_kwargs = copy.deepcopy(run_kwargs)
        member_kwargs["output_dir"] = member_dir
        member_kwargs["resume"] = False
        if reuse_prior:
            member_kwargs["reuse_prior"] = prior_round_path

        simulator_kwargs = copy.deepcopy(member_kwargs.get("simulator_kwargs", {}))
        if self._uses_arcis_simulator():
            simulator_kwargs["output_dir"] = os.path.join(member_dir, "arcis_outputs")
        member_kwargs["simulator_kwargs"] = simulator_kwargs
        return member_kwargs

    def _ensemble_member_resume_kwargs(self, run_kwargs, member_dir):
        member_kwargs = copy.deepcopy(run_kwargs)
        member_kwargs["output_dir"] = member_dir
        member_kwargs["resume"] = True
        member_kwargs.pop("reuse_prior", None)

        simulator_kwargs = copy.deepcopy(member_kwargs.get("simulator_kwargs", {}))
        if self._uses_arcis_simulator():
            simulator_kwargs["output_dir"] = os.path.join(member_dir, "arcis_outputs")
        member_kwargs["simulator_kwargs"] = simulator_kwargs
        return member_kwargs

    def _ensemble_total_rounds(self, member_dirs, fallback_n_rounds):
        max_rounds = int(fallback_n_rounds)
        for member_dir in member_dirs:
            rounds_dir = os.path.join(member_dir, "rounds")
            if not os.path.isdir(rounds_dir):
                continue
            for name in os.listdir(rounds_dir):
                if not name.startswith("round_"):
                    continue
                try:
                    round_number = int(name[len("round_"):])
                except ValueError:
                    continue
                max_rounds = max(max_rounds, round_number + 1)
        return max_rounds

    def _aggregate_ensemble_outputs(self, output_dir, member_dirs, n_rounds):
        aggregate_dir = os.path.join(output_dir, "aggregated")
        os.makedirs(aggregate_dir, exist_ok=True)
        aggregated = {
            "posterior_samples": [],
            "training_data": [],
            "atmospheres": [],
        }

        for round_index in range(n_rounds):
            posterior_path = self._aggregate_ensemble_posterior_samples(
                aggregate_dir,
                member_dirs,
                round_index,
            )
            if posterior_path is not None:
                aggregated["posterior_samples"].append(posterior_path)

            training_path = self._aggregate_ensemble_round_data(
                aggregate_dir,
                member_dirs,
                round_index,
            )
            if training_path is not None:
                aggregated["training_data"].append(training_path)

            atmosphere_path = self._aggregate_ensemble_atmospheres(
                aggregate_dir,
                member_dirs,
                round_index,
            )
            if atmosphere_path is not None:
                aggregated["atmospheres"].append(atmosphere_path)

        self._write_ensemble_summary(output_dir, member_dirs, aggregated)
        return aggregated

    def _aggregate_ensemble_posterior_samples(
        self,
        aggregate_dir,
        member_dirs,
        round_index,
    ):
        arrays = []
        header = " ".join(str(name) for name in self.parameters.keys())
        for member_dir in member_dirs:
            path = RetrievalOutput(member_dir).posterior_samples_path(round_index + 1)
            if os.path.exists(path):
                arrays.append(self._load_posterior_sample_file(path))
        if not arrays:
            return None

        output = os.path.join(
            aggregate_dir,
            f"posterior_samples_round_{round_index + 1}.txt",
        )
        np.savetxt(output, np.vstack(arrays), header=header)
        return output

    def _load_posterior_sample_file(self, path):
        samples = np.loadtxt(path)
        samples = np.asarray(samples)
        if samples.ndim == 1:
            n_parameters = len(self.parameters)
            if n_parameters == 1:
                return samples.reshape(-1, 1)
            return samples.reshape(1, -1)
        return samples

    def _aggregate_ensemble_round_data(self, aggregate_dir, member_dirs, round_index):
        round_data = []
        for member_dir in member_dirs:
            path = RetrievalOutput(member_dir).round_data_path(round_index)
            if os.path.exists(path):
                round_data.append(RetrievalOutput.load_round_data(path))
        if not round_data:
            return None

        spectra = self._concat_round_dicts(round_data, "spec")
        processed_spectra = self._concat_optional_round_dicts(round_data, "post_spec")
        output = RetrievalOutput(aggregate_dir)
        return output.write_round_data(
            round_index=round_index,
            thetas=np.concatenate([data["par"] for data in round_data], axis=0),
            nat_thetas=self._concat_optional_arrays(round_data, "nat_par"),
            spectra=spectra,
            sample_sources=self._concat_optional_arrays(round_data, "sample_sources"),
            processed_spectra=processed_spectra,
            fitted_radii=self._concat_optional_arrays(round_data, "fitted_radii"),
        )

    def _aggregate_ensemble_atmospheres(self, aggregate_dir, member_dirs, round_index):
        contents = []
        for fallback_index, member_dir in enumerate(member_dirs, start=1):
            path = os.path.join(
                member_dir,
                "arcis_outputs",
                "arcis_files",
                f"mixingratios_round_{round_index}.dat",
            )
            if not os.path.exists(path):
                continue
            with open(path) as file:
                member_number = (
                    self._ensemble_member_number(member_dir, "member")
                    or fallback_index
                )
                contents.append(
                    f"# ensemble_member={member_number} source={path}\n"
                    + file.read().rstrip()
                    + "\n"
                )
        if not contents:
            return None

        output_dir = os.path.join(aggregate_dir, "arcis_files")
        os.makedirs(output_dir, exist_ok=True)
        output = os.path.join(output_dir, f"mixingratios_round_{round_index}.dat")
        with open(output, "w") as file:
            file.write("\n".join(contents))
        return output

    def _write_ensemble_summary(self, output_dir, member_dirs, aggregated):
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_members": len(member_dirs),
            "member_dirs": member_dirs,
            "aggregated": aggregated,
        }
        path = os.path.join(output_dir, "ensemble_summary.json")
        with open(path, "w") as file:
            json.dump(self._json_safe(payload), file, indent=2, sort_keys=True)
            file.write("\n")
        return path

    @staticmethod
    def _concat_round_dicts(round_data, key):
        keys = round_data[0][key].keys()
        return {
            item_key: np.concatenate(
                [data[key][item_key] for data in round_data],
                axis=0,
            )
            for item_key in keys
        }

    @classmethod
    def _concat_optional_round_dicts(cls, round_data, key):
        if not all(key in data for data in round_data):
            return None
        return cls._concat_round_dicts(round_data, key)

    @staticmethod
    def _concat_optional_arrays(round_data, key):
        if not all(key in data for data in round_data):
            return None
        return np.concatenate([data[key] for data in round_data], axis=0)

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
        reused_thetas = prior_data.get("nat_par", prior_data["par"])[:reused_n]
        reused_sources = prior_data.get(
            "sample_sources",
            self._sample_sources("prior", len(prior_data["par"])),
        )[:reused_n]
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
            all_sources = np.concatenate([reused_sources, self.theta_sources], axis=0)
            all_x = {
                key: np.concatenate([reused_x[key], new_x[key]], axis=0)
                for key in reused_x
            }
        else:
            all_thetas = reused_thetas
            all_sources = reused_sources
            all_x = reused_x

        self.thetas = all_thetas
        self.theta_sources = all_sources
        self.nat_thetas = self.thetas
        self._canonicalize_current_thetas()
        self.n_samples = all_thetas.shape[0]
        self.get_x(all_x)

    def _sample_initial_thetas(self, method, n_samples):
        if method == "random":
            self.get_thetas(self.prior, n_samples)
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
                "sample_prior_method must be one of 'random' or 'sobol'."
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
        if not self._uses_arcis_simulator():
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
        path = self._arcis_artifact_path(simulator_kwargs, filename)
        if os.path.exists(path):
            os.remove(path)

    def _freeze_arcis_input_for_run(self, simulator_kwargs):
        simulator_kwargs = dict(simulator_kwargs)
        if not self._uses_arcis_simulator():
            return simulator_kwargs

        input_file = simulator_kwargs.get("input_file", "arcis_input.in")
        output_dir = simulator_kwargs.get("output_dir", "./arcis_outputs")
        frozen_input = self._frozen_arcis_input_path(
            input_file=input_file,
            output_dir=output_dir,
            arcis_file_dir=simulator_kwargs.get("arcis_file_dir", "arcis_files"),
        )
        os.makedirs(os.path.dirname(frozen_input), exist_ok=True)

        with open(input_file, "r") as file:
            lines = file.readlines()
        lines = self._arcis_input_lines_with_makeai(lines)
        with open(frozen_input, "w") as file:
            file.writelines(lines)

        simulator_kwargs.setdefault("original_input_file", input_file)
        simulator_kwargs["input_file"] = frozen_input
        return simulator_kwargs

    @staticmethod
    def _arcis_input_lines_with_makeai(lines):
        lines = list(lines)
        found_makeai = False
        for i, line in enumerate(lines):
            if "makeai=" in line.lower():
                found_makeai = True
                if ".false." in line.lower():
                    print('Warning: Found "makeai=.false." - changing to "makeai=.true."')
                    lines[i] = "makeai=.true.\n"
                break
        if not found_makeai:
            print('Warning: No "makeai=" found - adding "makeai=.true."')
            lines.append("makeai=.true.\n")
        return lines

    @staticmethod
    def _frozen_arcis_input_path(input_file, output_dir, arcis_file_dir):
        if not os.path.isabs(arcis_file_dir):
            arcis_file_dir = os.path.join(output_dir, arcis_file_dir)
        return os.path.join(arcis_file_dir, os.path.basename(input_file))

    def _uses_arcis_simulator(self):
        simulator_name = getattr(self.simulator, "__name__", None)
        base_simulator = getattr(self.simulator, "simulator", None)
        base_simulator_name = getattr(base_simulator, "__name__", None)
        return simulator_name in {
            "ARCiS",
            "ARCiS_binary",
            "ARCiS_multiple",
        } or base_simulator_name == "ARCiS"

    def _save_posterior_samples(self, output_dir, posterior, round_index):
        samples = posterior.sample((1000,))
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        samples = np.asarray(samples)
        self._output_manager(output_dir).write_posterior_samples(
            round_index=round_index + 1,
            samples=samples,
            parameter_names=self.parameters.keys(),
        )

    @staticmethod
    def _arcis_artifact_path(simulator_kwargs, filename):
        if os.path.isabs(filename):
            return filename
        arcis_file_dir = simulator_kwargs.get("arcis_file_dir", "arcis_files")
        output_dir = simulator_kwargs["output_dir"]
        if not os.path.isabs(arcis_file_dir):
            arcis_file_dir = os.path.join(output_dir, arcis_file_dir)
        return os.path.join(arcis_file_dir, filename)

    def _run_simulator(self, n_threads, simulator_kwargs):
        xs = self._simulate_current_thetas(
            n_threads=n_threads,
            simulator_kwargs=simulator_kwargs,
        )
        self.get_x(xs)

    def _simulate_current_thetas(self, n_threads, simulator_kwargs, sample_offset=0):
        self._canonicalize_current_thetas()
        self.sim_pars = self._n_simulator_parameters()
        sim_thetas = self.nat_thetas[:, : self.sim_pars]
        simulator_kwargs = dict(simulator_kwargs)
        simulator_kwargs["_sample_offset"] = sample_offset
        if n_threads == 1:
            return self.simulator(self.obs, sim_thetas, **simulator_kwargs)
        return self.psimulator(self.obs, sim_thetas, **simulator_kwargs)

    def _canonicalize_current_thetas(self):
        canonicalize = getattr(self.simulator, "canonicalize_parameters", None)
        if not callable(canonicalize):
            return
        self.thetas, self.nat_thetas = canonicalize(self.thetas, self.nat_thetas)

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
                * self.obs[key][:, 2][None, :]
                * np.random.standard_normal(self.augmented_x[key].shape)
            )
            self.noisy_x[key] = self.augmented_x[key] + noise

    def do_preprocessing(self, x, fit_preprocessing=False, fit_pca=False):
        """Apply configured preprocessing and optional fitted PCA."""
        xnorm = x
        for function_name in self.preprocessing or []:
            if function_name == "log_standardize":
                transformer = self.preprocessing_transformers.get(function_name)
                if fit_preprocessing:
                    transformer = preprocessing.LogStandardizer()
                    transformer.fit(xnorm)
                    self.preprocessing_transformers[function_name] = transformer
                if transformer is None:
                    raise RuntimeError(
                        "log_standardize preprocessing must be fitted on training "
                        "spectra before it can transform observations."
                    )
                xnorm = transformer.transform(xnorm)
                continue

            preprocessing_fun = getattr(preprocessing, function_name)
            if isinstance(xnorm, torch.Tensor):
                xnorm = preprocessing_fun(xnorm.cpu().numpy())
                xnorm = torch.as_tensor(
                    xnorm,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                xnorm = preprocessing_fun(xnorm)

        if self._pca_enabled():
            if fit_pca:
                self.pca = preprocessing.PCATransformer(self.pca_components)
                self.pca.fit(xnorm)
            if not self._pca_is_fitted():
                raise RuntimeError(
                    "PCA is enabled but has not been fitted. Run preprocessing "
                    "on training spectra with fit_pca=True first."
                )
            xnorm = self.pca.transform(xnorm)
        return xnorm

    def do_postprocessing(self):
        """Apply configured post-processing parameters to simulated spectra."""
        self.post_x = {key: value.copy() for key, value in self.x.items()}

        for par_idx, key in enumerate(self.parameters):
            if not self.parameters[key]["post_processing"]:
                continue
            if self._is_flux_calibration_parameter(key):
                continue

            par_values = self.nat_thetas[:, par_idx]
            postprocessing_function = getattr(postprocessing, key)
            for obs_key in self.post_x:
                self.post_x[obs_key] = postprocessing_function(
                    par_values,
                    self.obs[obs_key][:, 0],
                    self.post_x[obs_key],
                )

        if getattr(self, "fit_radius", False):
            print(
                "Fitting best-fit radius scales for simulated emission spectra.",
                flush=True,
            )
            start_time = time.perf_counter()
            self._fit_and_apply_radius_scale()
            elapsed = time.perf_counter() - start_time
            print(
                "Best-fit radii [R_reference units]: "
                f"min={np.nanmin(self.best_fit_radii):.4g}, "
                f"median={np.nanmedian(self.best_fit_radii):.4g}, "
                f"max={np.nanmax(self.best_fit_radii):.4g} "
                f"computed in {elapsed:.1f} s. "
                f"Flux scale = (R / {self.radius_reference:g})^2."
            )

        for par_idx, key in enumerate(self.parameters):
            if not self.parameters[key]["post_processing"]:
                continue
            if not self._is_flux_calibration_parameter(key):
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

    @staticmethod
    def _is_flux_calibration_parameter(parameter_name):
        return (
            Retrieval._is_observation_specific_parameter(parameter_name, "offset")
            or Retrieval._is_observation_specific_parameter(parameter_name, "scaling")
        )

    def _fit_and_apply_radius_scale(self):
        scales = self._best_fit_radius_scales(self.post_x)
        self.best_fit_radius_scales = scales
        self.best_fit_radii = self.radius_reference * np.sqrt(scales)
        for obs_key in self.post_x:
            self.post_x[obs_key] = self.post_x[obs_key] * scales[:, None]

    def _best_fit_radius_scales(self, spectra):
        numerator = np.zeros(self.n_samples)
        denominator = np.zeros(self.n_samples)
        chunk_size = 1024
        total_chunks = 0
        for obs_key in spectra:
            obs = self.obs[obs_key]
            valid = (
                np.isfinite(obs[:, 1])
                & np.isfinite(obs[:, 2])
                & (obs[:, 2] > 0)
            )
            if np.any(valid):
                total_chunks += int(np.ceil(self.n_samples / chunk_size))

        with tqdm(
            total=total_chunks,
            desc="Fitting radii",
            unit="chunk",
            file=sys.stdout,
        ) as progress:
            for obs_key, model in spectra.items():
                obs = self.obs[obs_key]
                observed = obs[:, 1]
                uncertainty = obs[:, 2]
                valid = (
                    np.isfinite(observed)
                    & np.isfinite(uncertainty)
                    & (uncertainty > 0)
                )
                if not np.any(valid):
                    continue

                weights = 1.0 / uncertainty[valid] ** 2
                model = np.asarray(model)
                for start in range(0, self.n_samples, chunk_size):
                    end = min(start + chunk_size, self.n_samples)
                    model_valid = model[start:end, valid]
                    finite_model = np.isfinite(model_valid)
                    weighted_model = np.where(finite_model, model_valid, 0.0)
                    numerator[start:end] += np.sum(
                        weighted_model * observed[valid] * weights,
                        axis=1,
                    )
                    denominator[start:end] += np.sum(
                        weighted_model * weighted_model * weights,
                        axis=1,
                    )
                    progress.update()

        scales = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        scales = np.clip(scales, 0.0, None)
        if self.radius_bounds is not None:
            lower, upper = self.radius_bounds
            lower_scale = (lower / self.radius_reference) ** 2
            upper_scale = (upper / self.radius_reference) ** 2
            scales = np.clip(scales, lower_scale, upper_scale)
        return scales

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
            samples,
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


def _deserialize_setup_key(key):
    for caster in (int, float):
        try:
            return caster(key)
        except (TypeError, ValueError):
            pass
    if key == "True":
        return True
    if key == "False":
        return False
    return key


class _UnavailableSetupSimulator:
    """Placeholder used when a setup log references an unavailable simulator."""

    def __init__(self, simulator_name):
        self.simulator_name = simulator_name
        self.__name__ = "unavailable_setup_simulator"

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Simulator {self.simulator_name!r} could not be imported. "
            "Pass simulator=... when rebuilding this setup."
        )


def _sample_mixture(prior, posterior, num_samples, alpha):
    n_prior = int(alpha * num_samples)
    n_posterior = num_samples - n_prior
    samples = []
    sources = []

    if n_prior:
        theta_prior = _as_2d_samples(prior.sample((n_prior,)))
        samples.append(theta_prior)
        sources.extend(["prior"] * n_prior)

    if n_posterior:
        theta_posterior = _as_2d_samples(posterior.sample((n_posterior,)))
        samples.append(theta_posterior)
        sources.extend(["proposal"] * n_posterior)

    return torch.cat(samples, dim=0), np.asarray(sources, dtype="<U8")


def _as_2d_samples(samples):
    if samples.ndim == 3 and samples.shape[1] == 1:
        return samples.squeeze(1)
    return samples


class _MixtureProposal:
    """Mixture of the prior and uninflated posterior for posterior inflation."""

    def __init__(self, prior, posterior, alpha):
        self.prior = prior
        self.posterior = posterior
        self.alpha = alpha
        self.last_sample_sources = None

    def sample(self, shape):
        samples, sources = _sample_mixture(
            prior=self.prior,
            posterior=self.posterior,
            num_samples=shape[0],
            alpha=self.alpha,
        )
        self.last_sample_sources = sources
        return samples

    def log_prob(self, theta):
        terms = []
        if self.alpha > 0:
            alpha = torch.as_tensor(self.alpha, dtype=theta.dtype, device=theta.device)
            terms.append(
                self.prior.log_prob(theta) + torch.log(alpha)
            )
        if self.alpha < 1:
            posterior_weight = torch.as_tensor(
                1 - self.alpha,
                dtype=theta.dtype,
                device=theta.device,
            )
            terms.append(
                self.posterior.log_prob(theta) + torch.log(posterior_weight)
            )
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)
