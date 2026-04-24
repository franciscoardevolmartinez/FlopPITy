import unittest
from unittest.mock import patch
import json
import numpy as np
import torch
from torch.distributions import Normal

from floppity import Retrieval
from floppity.helpers import (
    create_obs_file,
    convert_cube,
    compute_moments,
    find_MAP,
    reduced_chi_squared,
)
from floppity.output import RetrievalOutput
from floppity.simulators import (
    ARCiS,
    _append_arcis_atmosphere_structure,
    _arcis_obs_file_name,
    read_ARCiS_input,
)

class DummyProposal:
    def __init__(self, value):
        self.value = value

    def sample(self, shape):
        return torch.full((shape[0], 1), self.value, dtype=torch.float32)

    def log_prob(self, theta):
        return torch.zeros(theta.shape[0])


def flat_simulator(obs, parameters, thread=0, **kwargs):
    return {
        key: np.repeat(parameters[:, :1], len(value[:, 0]), axis=1)
        for key, value in obs.items()
    }


class TestHelpers(unittest.TestCase):

    def test_create_obs_file(self):
        wvl = np.array([1, 2, 3])
        spectrum = np.array([10, 20, 30])
        error = np.array([0.1, 0.2, 0.3])
        extra = np.array([100, 200, 300])
        obs = create_obs_file(wvl, spectrum, error, extra)
        self.assertEqual(obs.shape, (3, 4))
        np.testing.assert_array_equal(obs[:, 0], wvl)
        np.testing.assert_array_equal(obs[:, 1], spectrum)
        np.testing.assert_array_equal(obs[:, 2], error)
        np.testing.assert_array_equal(obs[:, 3], extra)

    def test_convert_cube(self):
        thetas = np.array([[0.5, 0.2], [0.1, 0.9]])
        pars = {
            "param1": {"min": 0, "max": 10},
            "param2": {"min": -5, "max": 5},
        }
        natural = convert_cube(thetas, pars)
        expected = np.array([[5, -3], [1, 4]])
        np.testing.assert_array_almost_equal(natural, expected)

    def test_compute_moments(self):
        torch.manual_seed(0)
        distribution = Normal(0, 1)
        moments = compute_moments(distribution)
        self.assertAlmostEqual(moments["mean"], 0, places=1)
        self.assertAlmostEqual(moments["variance"], 1, places=1)
        self.assertAlmostEqual(moments["skewness"], 0, places=1)
        self.assertAlmostEqual(moments["kurtosis"], 0, places=1)

    def test_find_MAP(self):
        class MockProposal:
            def map(self, **kwargs):
                return "MAP result"

        proposal = MockProposal()
        result = find_MAP(proposal)
        self.assertEqual(result, "MAP result")

    def test_reduced_chi_squared(self):
        obs_dict = {
            "obs1": np.array([[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]])
        }
        sim_dict = {
            "obs1": np.array([[10, 20, 30], [11, 19, 29]])
        }
        chi2_dict = reduced_chi_squared(obs_dict, sim_dict, n_params=1)
        self.assertIn("obs1", chi2_dict)
        self.assertEqual(chi2_dict["obs1"].shape, (2,))

    def test_get_obs_accepts_names(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = os.path.join(tmpdir, "obs.txt")
            np.savetxt(obs_path, np.array([[1.0, 10.0, 0.1]]))

            retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
            retrieval.get_obs([obs_path], obs_names=["prism"])

        self.assertIn("prism", retrieval.obs)
        np.testing.assert_array_equal(
            retrieval.obs["prism"],
            np.array([[1.0, 10.0, 0.1]]),
        )

    def test_get_obs_accepts_mapping_and_validates_names(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            first = os.path.join(tmpdir, "first.txt")
            second = os.path.join(tmpdir, "second.txt")
            np.savetxt(first, np.array([[1.0, 10.0, 0.1]]))
            np.savetxt(second, np.array([[2.0, 20.0, 0.2]]))

            retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
            retrieval.get_obs({"obs1": first, "miri": second})

            with self.assertRaises(ValueError):
                retrieval.get_obs({"obs1": first}, obs_names=["obs1"])

            with self.assertRaises(ValueError):
                retrieval.get_obs([first, second], obs_names=["only_one"])

        self.assertEqual(list(retrieval.obs.keys()), ["obs1", "miri"])

    def test_named_offset_and_scaling_postprocessing(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
        retrieval.obs = {
            "obs1": np.array([[1.0, 10.0, 0.1], [2.0, 20.0, 0.2]]),
            "miri": np.array([[3.0, 30.0, 0.3], [4.0, 40.0, 0.4]]),
        }
        retrieval.x = {
            "obs1": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "miri": np.array([[10.0, 20.0], [30.0, 40.0]]),
        }
        retrieval.nat_thetas = np.array([
            [0.5, 2.0],
            [1.0, 3.0],
        ])
        retrieval.parameters = {
            "offset:obs1": {"post_processing": True},
            "scaling:miri": {"post_processing": True},
        }

        retrieval.do_postprocessing()

        np.testing.assert_array_equal(
            retrieval.post_x["obs1"],
            np.array([[1.5, 2.5], [4.0, 5.0]]),
        )
        np.testing.assert_array_equal(
            retrieval.post_x["miri"],
            np.array([[20.0, 40.0], [90.0, 120.0]]),
        )

    def test_legacy_integer_offset_and_scaling_still_work(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
        retrieval.obs = {
            1: np.array([[1.0, 10.0, 0.1], [2.0, 20.0, 0.2]]),
            2: np.array([[3.0, 30.0, 0.3], [4.0, 40.0, 0.4]]),
        }
        retrieval.x = {
            1: np.array([[1.0, 2.0], [3.0, 4.0]]),
            2: np.array([[10.0, 20.0], [30.0, 40.0]]),
        }
        retrieval.nat_thetas = np.array([
            [0.5, 2.0],
            [1.0, 3.0],
        ])
        retrieval.parameters = {
            "offset1": {"post_processing": True},
            "scaling2": {"post_processing": True},
        }

        retrieval.do_postprocessing()

        np.testing.assert_array_equal(
            retrieval.post_x[1],
            np.array([[1.5, 2.5], [4.0, 5.0]]),
        )
        np.testing.assert_array_equal(
            retrieval.post_x[2],
            np.array([[20.0, 40.0], [90.0, 120.0]]),
        )

    def test_observation_specific_postprocessing_errors_are_clear(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
        retrieval.obs = {"obs1": np.array([[1.0, 10.0, 0.1]])}
        retrieval.x = {"obs1": np.array([[1.0]])}
        retrieval.nat_thetas = np.array([[0.5]])

        retrieval.parameters = {"offset": {"post_processing": True}}
        with self.assertRaises(ValueError):
            retrieval.do_postprocessing()

        retrieval.parameters = {"scaling:missing": {"post_processing": True}}
        with self.assertRaises(KeyError):
            retrieval.do_postprocessing()

    def test_read_ARCiS_input_returns_named_observations(self):
        import tempfile
        import os

        contents = """
        fitpar:keyword="temperature"
        fitpar:min=1000
        fitpar:max=2000
        fitpar:log=.false.
        obs1:file="prism.dat"
        obs2:file="miri.dat"
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "arcis.in")
            with open(input_path, "w") as file:
                file.write(contents)

            pars, obs = read_ARCiS_input(input_path)

        self.assertIn("temperature", pars)
        self.assertEqual(obs, {"obs1": "prism.dat", "obs2": "miri.dat"})

    def test_arcis_atmosphere_structure_appender(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model000001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "mixingratios.dat"), "w") as file:
                file.write("pressure h2o\n1e-3 1e-4\n")

            output_path = os.path.join(tmpdir, "mixingratios_round_0.dat")
            _append_arcis_atmosphere_structure(
                model_dir=model_dir,
                atmosphere_file="mixingratios.dat",
                output_path=output_path,
                round_index=0,
                thread=1,
                local_model_index=2,
                global_model_index=5,
                parameters=np.array([1000.0, -3.0]),
            )

            with open(output_path) as file:
                contents = file.read()

        self.assertIn("# round=0 global_model=5 thread=1 local_model=2", contents)
        self.assertIn("parameters=1000 -3", contents)
        self.assertIn("pressure h2o\n1e-3 1e-4", contents)

    def test_arcis_obs_file_name_mapping(self):
        self.assertEqual(_arcis_obs_file_name(0), "obs001")
        self.assertEqual(_arcis_obs_file_name("obs1"), "obs001")
        self.assertEqual(_arcis_obs_file_name("obs12"), "obs012")
        self.assertEqual(_arcis_obs_file_name("miri"), "miri")

    def test_arcis_wrapper_reads_spectra_collects_atmospheres_and_cleans_models(self):
        import tempfile
        import os

        def fake_run(cmd, check, stdout, stderr, text, env):
            output_base = cmd[3]
            os.makedirs(output_base, exist_ok=True)
            for i in range(2):
                model_dir = os.path.join(output_base, f"model{i + 1:06d}")
                os.makedirs(model_dir)
                np.savetxt(
                    os.path.join(model_dir, "obs001"),
                    np.column_stack([[1.0, 2.0], [10.0 + i, 20.0 + i]]),
                )
                np.savetxt(
                    os.path.join(model_dir, "obs002"),
                    np.column_stack([[3.0, 4.0], [30.0 + i, 40.0 + i]]),
                )
                with open(os.path.join(model_dir, "mixingratios.dat"), "w") as file:
                    file.write(f"model {i + 1}\n")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "arcis.in")
            with open(input_file, "w") as file:
                file.write("makeai=.true.\n")

            obs = {
                "obs1": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
                "obs2": np.array([[3.0, 0.0, 0.1], [4.0, 0.0, 0.1]]),
            }
            parameters = np.array([[1.0, 2.0], [3.0, 4.0]])

            with patch("floppity.simulators.subprocess.run", side_effect=fake_run):
                spectra = ARCiS(
                    obs,
                    parameters,
                    thread=1,
                    input_file=input_file,
                    output_dir=tmpdir,
                    ARCiS_dir="/fake/ARCiS",
                    _round_index=7,
                    _sample_offset=10,
                )

            atmosphere_path = os.path.join(tmpdir, "mixingratios_round_7.dat")
            with open(atmosphere_path) as file:
                atmosphere = file.read()

            output_base = os.path.join(tmpdir, "outputARCiS_1")

        np.testing.assert_array_equal(
            spectra["obs1"],
            np.array([[10.0, 20.0], [11.0, 21.0]]),
        )
        np.testing.assert_array_equal(
            spectra["obs2"],
            np.array([[30.0, 40.0], [31.0, 41.0]]),
        )
        self.assertIn("# round=7 global_model=10 thread=1 local_model=0", atmosphere)
        self.assertIn("# round=7 global_model=11 thread=1 local_model=1", atmosphere)
        self.assertIn("model 1", atmosphere)
        self.assertIn("model 2", atmosphere)
        self.assertFalse(os.path.exists(os.path.join(output_base, "model000001")))

    def test_round_kwargs_and_arcis_output_reset_are_scoped(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "mixingratios_round_2.dat")
            with open(output_path, "w") as file:
                file.write("old")

            retrieval = Retrieval(ARCiS, obs_type="emis")
            kwargs = retrieval._simulator_kwargs_for_round(
                {"output_dir": tmpdir},
                round_index=2,
            )
            retrieval._prepare_simulator_round_outputs(kwargs)
            self.assertFalse(os.path.exists(output_path))

            with open(output_path, "w") as file:
                file.write("old")

            retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
            retrieval._prepare_simulator_round_outputs(kwargs)
            self.assertTrue(os.path.exists(output_path))

    def test_psimulator_passes_global_sample_offsets_to_chunks(self):
        def simulator(obs, parameters, thread=0, **kwargs):
            offsets.append(kwargs["_sample_offset"])
            return {"obs": np.ones((len(parameters), len(obs["obs"]))) * thread}

        offsets = []
        retrieval = Retrieval(simulator, obs_type="trans")
        retrieval.obs = {"obs": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]])}
        retrieval.n_threads = 3

        chunks = retrieval._parameter_chunks(np.arange(10).reshape(5, 2), 3)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])

        args = []
        sample_offset = 7
        chunk_start = 0
        for i, chunk in enumerate(chunks):
            kwargs = {"_sample_offset": sample_offset + chunk_start}
            args.append((simulator, retrieval.obs, chunk, i, kwargs))
            chunk_start += len(chunk)

        for arg in args:
            from floppity.flappity import _run_single_chunk

            _run_single_chunk(arg)

        self.assertEqual(offsets, [7, 9, 11])

    def test_generate_training_data_resume_uses_proposal_not_prior(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.obs = {
            "obs": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        retrieval.default_obs = np.array([[0.0, 0.0]])
        retrieval.error_inflation = 0
        retrieval.parameters = {
            "level": {
                "min": 0,
                "max": 1,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }
        retrieval.prior = DummyProposal(0.1)

        theta_tensor, x_tensor = retrieval.generate_training_data(
            proposal=DummyProposal(0.8),
            r=1,
            n_samples=3,
            n_samples_init=3,
            sample_prior_method="random",
            n_threads=1,
            simulator_kwargs={},
            n_aug=1,
            initial_round=False,
        )

        np.testing.assert_allclose(theta_tensor.numpy(), np.full((3, 1), 0.8))
        np.testing.assert_allclose(x_tensor.numpy(), np.full((3, 2), 0.8))

    def test_generate_training_data_initial_round_uses_prior_method(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.obs = {
            "obs": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        retrieval.error_inflation = 0
        retrieval.parameters = {
            "level": {
                "min": 0,
                "max": 1,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }
        retrieval.prior = DummyProposal(0.2)

        theta_tensor, _ = retrieval.generate_training_data(
            proposal=DummyProposal(0.8),
            r=0,
            n_samples=3,
            n_samples_init=3,
            sample_prior_method="random",
            n_threads=1,
            simulator_kwargs={},
            n_aug=1,
            initial_round=True,
        )

        np.testing.assert_allclose(theta_tensor.numpy(), np.full((3, 1), 0.2))

    def test_mixture_proposal_tracks_per_sample_sources(self):
        from floppity.flappity import _MixtureProposal

        mixture = _MixtureProposal(
            prior=DummyProposal(0.1),
            posterior=DummyProposal(0.8),
            alpha=0.4,
        )

        samples = mixture.sample((5,))

        np.testing.assert_array_equal(
            samples.numpy(),
            np.array([[0.1], [0.1], [0.8], [0.8], [0.8]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mixture.last_sample_sources,
            np.array(["prior", "prior", "proposal", "proposal", "proposal"]),
        )

    def test_get_thetas_preserves_mixture_sample_sources(self):
        from floppity.flappity import _MixtureProposal

        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.parameters = {
            "level": {
                "min": 0,
                "max": 1,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }
        proposal = _MixtureProposal(
            prior=DummyProposal(0.1),
            posterior=DummyProposal(0.8),
            alpha=0.5,
        )

        retrieval.get_thetas(proposal, 4)

        np.testing.assert_array_equal(
            retrieval.theta_sources,
            np.array(["prior", "prior", "proposal", "proposal"]),
        )

    def test_resume_state_validation(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        with self.assertRaises(RuntimeError):
            retrieval._prepare_run(resume=True, flow_kwargs={})

        retrieval.proposals = []
        retrieval.prior = DummyProposal(0.1)
        retrieval.inference = object()
        retrieval.posterior_estimator = object()
        with self.assertRaises(RuntimeError):
            retrieval._prepare_run(resume=True, flow_kwargs={})

    def test_resume_initializes_missing_posteriors_list(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.proposals = [DummyProposal(0.1)]
        retrieval.prior = DummyProposal(0.1)
        retrieval.inference = object()
        retrieval.posterior_estimator = object()

        retrieval._prepare_run(resume=True, flow_kwargs={})

        self.assertEqual(retrieval.posteriors, [])

    def test_alpha_validation_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            Retrieval._validate_alpha(-0.1)
        with self.assertRaises(ValueError):
            Retrieval._validate_alpha(1.1)

    def test_save_and_load_drop_transient_arrays(self):
        import tempfile
        import os

        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.x = {"obs": np.array([[1.0]])}
        retrieval.noisy_x = {"obs": np.array([[2.0]])}
        retrieval.thetas = np.array([[0.1]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "retrieval.pkl")
            retrieval.save(path)
            loaded = Retrieval.load(path)

        self.assertFalse(hasattr(loaded, "x"))
        self.assertFalse(hasattr(loaded, "noisy_x"))
        self.assertFalse(hasattr(loaded, "thetas"))
        self.assertEqual(loaded.completed_rounds, 0)

    def test_output_manager_round_data_npz_round_trip_preserves_keys(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output = RetrievalOutput(tmpdir)
            path = output.write_round_data(
                round_index=2,
                thetas=np.array([[0.1, 0.2], [0.3, 0.4]]),
                nat_thetas=np.array([[1.0, 2.0], [3.0, 4.0]]),
                spectra={
                    "obs1": np.array([[10.0, 20.0], [30.0, 40.0]]),
                    2: np.array([[50.0], [60.0]]),
                },
                processed_spectra={
                    "obs1": np.array([[11.0, 21.0], [31.0, 41.0]]),
                    2: np.array([[51.0], [61.0]]),
                },
                sample_sources=np.array(["prior", "proposal"]),
            )
            loaded = RetrievalOutput.load_round_data(path)

        self.assertTrue(path.endswith(os.path.join("round_002", "training_data.npz")))
        np.testing.assert_array_equal(
            loaded["par"],
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        np.testing.assert_array_equal(
            loaded["nat_par"],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        self.assertEqual(list(loaded["spec"].keys()), ["obs1", 2])
        np.testing.assert_array_equal(
            loaded["sample_sources"],
            np.array(["prior", "proposal"]),
        )
        np.testing.assert_array_equal(
            loaded["spec"]["obs1"],
            np.array([[10.0, 20.0], [30.0, 40.0]]),
        )
        np.testing.assert_array_equal(
            loaded["post_spec"]["obs1"],
            np.array([[11.0, 21.0], [31.0, 41.0]]),
        )

    def test_save_round_data_writes_npz_archive(self):
        import tempfile
        import os

        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.thetas = np.array([[0.1]])
        retrieval.nat_thetas = np.array([[1.0]])
        retrieval.x = {"obs": np.array([[2.0]])}

        with tempfile.TemporaryDirectory() as tmpdir:
            retrieval._save_round_data(tmpdir, save_data=True, round_index=1)

            npz_path = os.path.join(tmpdir, "rounds", "round_001", "training_data.npz")
            pkl_path = os.path.join(tmpdir, "data_1.pkl")

            self.assertTrue(os.path.exists(npz_path))
            self.assertFalse(os.path.exists(pkl_path))

    def test_reuse_prior_loads_npz_training_data(self):
        import tempfile

        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.obs = {
            "obs": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        retrieval.parameters = {
            "level": {
                "min": 0,
                "max": 10,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = RetrievalOutput(tmpdir).write_round_data(
                round_index=0,
                thetas=np.array([[0.2], [0.4]]),
                spectra={"obs": np.array([[2.0, 2.0], [4.0, 4.0]])},
            )
            retrieval._load_or_extend_reused_prior(
                reuse_prior=path,
                n_samples=2,
                sample_prior_method="random",
                n_threads=1,
                simulator_kwargs={},
            )

        np.testing.assert_array_equal(retrieval.thetas, np.array([[0.2], [0.4]]))
        np.testing.assert_array_equal(retrieval.nat_thetas, np.array([[2.0], [4.0]]))
        np.testing.assert_array_equal(
            retrieval.x["obs"],
            np.array([[2.0, 2.0], [4.0, 4.0]]),
        )

    def test_preprocessing_chain_preserves_tensor_output(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.preprocessing = ["log"]
        result = retrieval.do_preprocessing(torch.tensor([[1.0, 10.0]]))

        self.assertIsInstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), np.array([[0.0, 1.0]]))

    def test_emission_observations_and_noisy_spectra_are_clipped(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = os.path.join(tmpdir, "obs.txt")
            np.savetxt(obs_path, np.array([1.0, -5.0, 0.1]))

            retrieval = Retrieval(flat_simulator, obs_type="emis")
            retrieval.get_obs([obs_path])

        self.assertEqual(retrieval.obs[0][0, 1], 1e-12)

        arrays = {"obs": np.array([[-1.0, 0.0, 1.0]])}
        retrieval._clip_emission_arrays(arrays)
        np.testing.assert_array_equal(arrays["obs"], np.array([[1e-12, 1e-12, 1.0]]))

    def test_nearest_power_of_two_rejects_non_positive_samples(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        with self.assertRaises(ValueError):
            retrieval._nearest_power_of_two(0)

    def test_setup_log_records_inputs_and_run_configuration(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = os.path.join(tmpdir, "obs.txt")
            np.savetxt(
                obs_path,
                np.array([
                    [1.0, 10.0, 0.1],
                    [2.0, 20.0, 0.2],
                ]),
            )

            retrieval = Retrieval(flat_simulator, obs_type="trans")
            retrieval.get_obs({"obs1": obs_path}, error_inflation=2)
            retrieval.add_parameter("level", 0.0, 1.0)
            retrieval.preprocessing = ["log"]
            log_path = retrieval.write_setup_log(
                tmpdir,
                run_config={
                    "n_rounds": 3,
                    "flow_kwargs": {"hidden": np.int64(16)},
                    "simulator_kwargs": {"callback": flat_simulator},
                },
            )

            with open(log_path) as file:
                payload = json.load(file)

        self.assertEqual(payload["retrieval"]["obs_type"], "trans")
        self.assertEqual(payload["retrieval"]["simulator"], "test_core.flat_simulator")
        self.assertEqual(payload["retrieval"]["preprocessing"], ["log"])
        self.assertEqual(payload["retrieval"]["error_inflation"], 2)
        self.assertEqual(payload["observations"]["obs1"]["source"], obs_path)
        self.assertEqual(payload["observations"]["obs1"]["shape"], [2, 3])
        self.assertEqual(payload["observations"]["obs1"]["wavelength_min"], 1.0)
        self.assertEqual(payload["observations"]["obs1"]["wavelength_max"], 2.0)
        self.assertEqual(payload["parameters"]["level"]["min"], 0.0)
        self.assertEqual(payload["parameters"]["level"]["max"], 1.0)
        self.assertEqual(payload["run"]["n_rounds"], 3)
        self.assertEqual(payload["run"]["flow_kwargs"]["hidden"], 16)
        self.assertEqual(
            payload["run"]["simulator_kwargs"]["callback"],
            "test_core.flat_simulator",
        )
        self.assertEqual(payload["outputs"]["completed_checkpoint"], "retrieval.pkl")

if __name__ == "__main__":
    unittest.main()
