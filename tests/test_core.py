import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
import json
import inspect
import sys
import types
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
from floppity.preprocessing import PCATransformer
from floppity.simulators import (
    ARCiS,
    ARCiS_binary,
    ARCiS_multiple,
    _arcis_atmosphere_columns,
    _arcis_atmosphere_numeric_contents,
    _append_arcis_atmosphere_structure,
    _arcis_obs_file_name,
    _remove_arcis_output,
    make_binary_simulator,
    make_multi_component_parameters,
    make_multi_component_simulator,
    read_ARCiS_input,
)

class DummyProposal:
    def __init__(self, value):
        self.value = value

    def sample(self, shape):
        return torch.full((shape[0], 1), self.value, dtype=torch.float32)

    def log_prob(self, theta):
        return torch.zeros(theta.shape[0])

    def set_default_x(self, x):
        return self


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

    def test_retrieval_defaults_match_quickstart_settings(self):
        retrieval = Retrieval(lambda obs, pars: {})
        self.assertEqual(retrieval.obs_type, "emis")

        run_signature = inspect.signature(Retrieval.run)
        self.assertEqual(run_signature.parameters["n_samples"].default, 2048)
        self.assertEqual(run_signature.parameters["n_rounds"].default, 5)

        density_signature = inspect.signature(Retrieval.density_builder)
        self.assertEqual(density_signature.parameters["flow"].default, "nsf")
        self.assertEqual(density_signature.parameters["bins"].default, 5)
        self.assertEqual(density_signature.parameters["transforms"].default, 8)
        self.assertEqual(density_signature.parameters["blocks"].default, 2)
        self.assertEqual(density_signature.parameters["hidden"].default, 64)
        self.assertEqual(density_signature.parameters["dropout"].default, 0.05)

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

    def test_best_fit_radius_scales_emission_spectra(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="emis")
        retrieval.obs = {
            "obs1": np.array([[1.0, 2.0, 0.1], [2.0, 4.0, 0.1]]),
            "obs2": np.array([[3.0, 6.0, 0.2]]),
        }
        retrieval.x = {
            "obs1": np.array([[1.0, 2.0], [2.0, 2.0]]),
            "obs2": np.array([[3.0], [2.0]]),
        }
        retrieval.n_samples = 2
        retrieval.parameters = {}
        retrieval._configure_radius_fit(fit_radius=True, radius_reference=1.0)

        stdout = StringIO()
        with redirect_stdout(stdout):
            retrieval.do_postprocessing()

        np.testing.assert_allclose(retrieval.best_fit_radius_scales[0], 2.0)
        np.testing.assert_allclose(retrieval.best_fit_radii[0], np.sqrt(2.0))
        np.testing.assert_allclose(retrieval.post_x["obs1"][0], np.array([2.0, 4.0]))
        np.testing.assert_allclose(retrieval.post_x["obs2"][0], np.array([6.0]))
        self.assertIn("Fitting best-fit radius scales", stdout.getvalue())
        self.assertIn("Best-fit radii", stdout.getvalue())

    def test_best_fit_radius_respects_bounds(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="emis")
        retrieval.obs = {
            "obs": np.array([[1.0, 100.0, 1.0]]),
        }
        retrieval.x = {"obs": np.array([[1.0]])}
        retrieval.n_samples = 1
        retrieval.parameters = {}
        retrieval._configure_radius_fit(
            fit_radius=True,
            radius_bounds=(0.5, 2.0),
            radius_reference=1.0,
        )

        retrieval.do_postprocessing()

        np.testing.assert_allclose(retrieval.best_fit_radius_scales, np.array([4.0]))
        np.testing.assert_allclose(retrieval.best_fit_radii, np.array([2.0]))
        np.testing.assert_allclose(retrieval.post_x["obs"], np.array([[4.0]]))

    def test_radius_fit_is_emission_only(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
        with self.assertRaises(ValueError):
            retrieval._configure_radius_fit(fit_radius=True)

    def test_radius_fit_warns_when_radius_is_still_sampled(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="emis")
        retrieval.parameters = {
            "radius": {
                "min": 0.5,
                "max": 2.0,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }

        with self.assertWarnsRegex(RuntimeWarning, "radius-like"):
            retrieval._configure_radius_fit(fit_radius=True)

    def test_radius_fit_happens_before_flux_offsets(self):
        retrieval = Retrieval(lambda obs, pars: {}, obs_type="emis")
        retrieval.obs = {"obs": np.array([[1.0, 5.0, 1.0]])}
        retrieval.x = {"obs": np.array([[2.0]])}
        retrieval.n_samples = 1
        retrieval.nat_thetas = np.array([[1.0]])
        retrieval.parameters = {
            "offset:obs": {
                "min": 0,
                "max": 2,
                "log": False,
                "post_processing": True,
                "universal": True,
            },
        }
        retrieval._configure_radius_fit(fit_radius=True)

        retrieval.do_postprocessing()

        np.testing.assert_allclose(retrieval.best_fit_radius_scales, np.array([2.5]))
        np.testing.assert_allclose(retrieval.post_x["obs"], np.array([[6.0]]))

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
        self.assertIn("# columns=pressure h2o", contents)
        self.assertIn("parameters=1000 -3", contents)
        self.assertIn("1e-3 1e-4", contents)
        self.assertNotIn("\npressure h2o\n", contents)

    def test_arcis_atmosphere_columns_parse_header_units(self):
        contents = "#    T [K]      P [bar]     H2O            CH4\n1000 1e-3 1e-4 1e-6\n"
        self.assertEqual(
            _arcis_atmosphere_columns(contents),
            ["T", "P", "H2O", "CH4"],
        )

    def test_arcis_atmosphere_numeric_contents_drops_header(self):
        contents = "#    T [K]      P [bar]     H2O\n1000 1e-3 1e-4\n"
        self.assertEqual(
            _arcis_atmosphere_numeric_contents(contents),
            "1000 1e-3 1e-4",
        )

    def test_arcis_obs_file_name_mapping(self):
        self.assertEqual(_arcis_obs_file_name(0), "obs001")
        self.assertEqual(_arcis_obs_file_name("obs1"), "obs001")
        self.assertEqual(_arcis_obs_file_name("obs12"), "obs012")
        self.assertEqual(_arcis_obs_file_name("miri"), "miri")

    def test_remove_arcis_output_removes_top_level_directory(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = os.path.join(tmpdir, "outputARCiS_0_component1")
            model_dir = os.path.join(output_base, "model000001")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "obs001"), "w") as file:
                file.write("1 2\n")

            _remove_arcis_output(output_base)

            self.assertFalse(os.path.exists(output_base))

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

            arcis_file_dir = os.path.join(tmpdir, "arcis_files")
            atmosphere_path = os.path.join(arcis_file_dir, "mixingratios_round_7.dat")
            with open(atmosphere_path) as file:
                atmosphere = file.read()

            output_base = os.path.join(tmpdir, "outputARCiS_1")
            log_dir = os.path.join(arcis_file_dir, "logs")
            parameter_grid_dir = os.path.join(arcis_file_dir, "parameter_grids")
            output_base_exists = os.path.exists(output_base)
            log_exists = os.path.exists(os.path.join(log_dir, "arcis_run_1_1.log"))
            root_log_exists = os.path.exists(os.path.join(tmpdir, "arcis_run_1_1.log"))
            input_copy_exists = os.path.exists(os.path.join(arcis_file_dir, "arcis.in"))
            root_input_copy_exists = os.path.exists(os.path.join(tmpdir, "arcis.in"))
            parameter_grid_exists = os.path.exists(
                os.path.join(parameter_grid_dir, "parametergridfile_1.dat")
            )
            root_parameter_grid_exists = os.path.exists(
                os.path.join(tmpdir, "parametergridfile_1.dat")
            )

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
        self.assertFalse(output_base_exists)
        self.assertTrue(input_copy_exists)
        self.assertTrue(root_input_copy_exists)
        self.assertTrue(log_exists)
        self.assertFalse(root_log_exists)
        self.assertTrue(parameter_grid_exists)
        self.assertFalse(root_parameter_grid_exists)

    def test_arcis_binary_splits_parameters_and_sums_components(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        parameters = np.array([
            [1.0, 2.0, 10.0, 20.0],
            [3.0, 4.0, 30.0, 40.0],
        ])
        calls = []

        def fake_arcis(obs_arg, component_parameters, thread=0, **kwargs):
            calls.append((component_parameters.copy(), thread))
            return {
                "obs1": np.repeat(
                    component_parameters.sum(axis=1, keepdims=True),
                    len(obs_arg["obs1"]),
                    axis=1,
                )
            }

        with patch("floppity.simulators.ARCiS", side_effect=fake_arcis):
            spectra = ARCiS_binary(obs, parameters, thread=5)

        np.testing.assert_array_equal(calls[0][0], parameters[:, 2:])
        np.testing.assert_array_equal(calls[1][0], parameters[:, :2])
        self.assertEqual(calls[0][1], "5_component1")
        self.assertEqual(calls[1][1], "5_component2")
        np.testing.assert_array_equal(
            spectra["obs1"],
            np.array([[33.0, 33.0], [77.0, 77.0]]),
        )

        with self.assertRaises(ValueError):
            ARCiS_multiple(obs, np.ones((2, 5)), n_components=2)

    def test_generic_binary_wrapper_supports_shared_parameters(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        base_parameters = {
            "temperature": {"min": 500, "max": 2500, "post_processing": False},
            "gravity": {"min": 3, "max": 5, "post_processing": False},
            "log_h2o": {"min": -12, "max": -1, "post_processing": False},
        }
        calls = []

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            calls.append((parameters.copy(), thread))
            return {
                "obs1": np.repeat(
                    parameters.sum(axis=1, keepdims=True),
                    len(obs_arg["obs1"]),
                    axis=1,
                )
            }

        binary_simulator, binary_parameters = make_binary_simulator(
            simulator,
            base_parameters,
            shared_parameters=["log_h2o"],
        )
        self.assertEqual(
            list(binary_parameters),
            ["temperature_1", "temperature_2", "gravity_1", "gravity_2", "log_h2o"],
        )

        theta = np.array([[1000.0, 1200.0, 4.0, 4.5, -4.0]])
        spectra = binary_simulator(obs, theta, thread=3)

        np.testing.assert_array_equal(
            calls[0][0],
            np.array([[1200.0, 4.5, -4.0]]),
        )
        np.testing.assert_array_equal(
            calls[1][0],
            np.array([[1000.0, 4.0, -4.0]]),
        )
        self.assertEqual(calls[0][1], "3_component1")
        self.assertEqual(calls[1][1], "3_component2")
        np.testing.assert_array_equal(spectra["obs1"], np.array([[2200.5, 2200.5]]))

    def test_multi_component_wrapper_supports_fixed_weights(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        base_parameters = {
            "level": {"min": 0, "max": 10, "post_processing": False},
        }

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            return {
                "obs1": np.repeat(parameters, len(obs_arg["obs1"]), axis=1)
            }

        wrapped_simulator, parameters = make_binary_simulator(
            simulator,
            base_parameters,
            component_weights=[0.25, 0.75],
        )

        self.assertEqual(list(parameters), ["level_1", "level_2"])
        spectra = wrapped_simulator(obs, np.array([[4.0, 8.0]]))
        np.testing.assert_array_equal(spectra["obs1"], np.array([[7.0, 7.0]]))

    def test_multi_component_wrapper_supports_sampled_binary_fraction(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        base_parameters = {
            "level": {"min": 0, "max": 10, "post_processing": False},
        }

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            return {
                "obs1": np.repeat(parameters, len(obs_arg["obs1"]), axis=1)
            }

        wrapped_simulator, parameters = make_binary_simulator(
            simulator,
            base_parameters,
            weight_parameters={"column_fraction": (0, 1)},
        )

        self.assertEqual(list(parameters), ["level_1", "level_2", "column_fraction"])
        spectra = wrapped_simulator(obs, np.array([[10.0, 20.0, 0.25]]))
        np.testing.assert_array_equal(spectra["obs1"], np.array([[17.5, 17.5]]))

    def test_multi_component_wrapper_sorts_each_sample_by_first_free_parameter(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1]]),
        }
        base_parameters = {
            "temperature": {"min": 0, "max": 3000, "post_processing": False},
            "gravity": {"min": 3, "max": 5, "post_processing": False},
            "chemistry": {"min": -12, "max": -1, "post_processing": False},
        }
        calls = []

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            calls.append(parameters.copy())
            return {"obs1": parameters[:, :1]}

        wrapped_simulator, _ = make_binary_simulator(
            simulator,
            base_parameters,
            shared_parameters=["chemistry"],
            weight_parameters={"column_fraction": (0, 1)},
        )
        theta = np.array([
            [500.0, 1000.0, 3.5, 4.0, -4.0, 0.25],
            [1500.0, 700.0, 4.5, 3.0, -5.0, 0.75],
        ])

        wrapped_simulator(obs, theta)

        np.testing.assert_array_equal(
            calls[0],
            np.array([
                [1000.0, 4.0, -4.0],
                [1500.0, 4.5, -5.0],
            ]),
        )
        np.testing.assert_array_equal(
            calls[1],
            np.array([
                [500.0, 3.5, -4.0],
                [700.0, 3.0, -5.0],
            ]),
        )

    def test_multi_component_wrapper_canonicalizes_training_parameters(self):
        base_parameters = {
            "temperature": {"min": 0, "max": 3000, "post_processing": False},
            "gravity": {"min": 3, "max": 5, "post_processing": False},
            "chemistry": {"min": -12, "max": -1, "post_processing": False},
        }

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            return {"obs1": parameters[:, :1]}

        wrapped_simulator, _ = make_binary_simulator(
            simulator,
            base_parameters,
            shared_parameters=["chemistry"],
            weight_parameters={"column_fraction": (0, 1)},
        )
        theta = np.array([
            [0.2, 0.8, 0.1, 0.9, 0.5, 0.25],
            [0.9, 0.3, 0.7, 0.2, 0.4, 0.75],
        ])
        nat_theta = np.array([
            [500.0, 1000.0, 3.5, 4.0, -4.0, 0.25],
            [1500.0, 700.0, 4.5, 3.0, -5.0, 0.75],
        ])

        sorted_theta, sorted_nat_theta = wrapped_simulator.canonicalize_parameters(
            theta,
            nat_theta,
        )

        np.testing.assert_array_equal(
            sorted_nat_theta,
            np.array([
                [1000.0, 500.0, 4.0, 3.5, -4.0, 0.75],
                [1500.0, 700.0, 4.5, 3.0, -5.0, 0.75],
            ]),
        )
        np.testing.assert_array_equal(
            sorted_theta,
            np.array([
                [0.8, 0.2, 0.9, 0.1, 0.5, 0.75],
                [0.9, 0.3, 0.7, 0.2, 0.4, 0.75],
            ]),
        )

    def test_retrieval_canonicalizes_current_thetas_before_simulation(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1]]),
        }
        base_parameters = {
            "temperature": {"min": 0, "max": 3000, "post_processing": False},
            "gravity": {"min": 3, "max": 5, "post_processing": False},
        }

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            return {"obs1": parameters[:, :1]}

        wrapped_simulator, parameters = make_binary_simulator(
            simulator,
            base_parameters,
        )
        retrieval = Retrieval(wrapped_simulator, obs_type="emis")
        retrieval.obs = obs
        retrieval.parameters = parameters
        retrieval.thetas = np.array([[500.0, 1000.0, 3.5, 4.0]])
        retrieval.nat_thetas = retrieval.thetas

        retrieval._canonicalize_current_thetas()

        np.testing.assert_array_equal(
            retrieval.nat_thetas,
            np.array([[1000.0, 500.0, 4.0, 3.5]]),
        )
        np.testing.assert_array_equal(
            retrieval.thetas,
            np.array([[1000.0, 500.0, 4.0, 3.5]]),
        )

    def test_multi_component_wrapper_supports_arbitrary_weighted_components(self):
        obs = {
            "obs1": np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]),
        }
        base_parameters = {
            "level": {"min": 0, "max": 10, "post_processing": False},
        }

        def simulator(obs_arg, parameters, thread=0, **kwargs):
            return {
                "obs1": np.repeat(parameters, len(obs_arg["obs1"]), axis=1)
            }

        wrapped_simulator, parameters = make_multi_component_simulator(
            simulator,
            base_parameters,
            n_components=3,
            weight_parameters={
                "weight_1": (0, 1),
                "weight_2": (0, 1),
                "weight_3": (0, 1),
            },
        )

        self.assertEqual(
            list(parameters),
            ["level_1", "level_2", "level_3", "weight_1", "weight_2", "weight_3"],
        )
        spectra = wrapped_simulator(
            obs,
            np.array([[10.0, 20.0, 30.0, 1.0, 1.0, 2.0]]),
        )
        np.testing.assert_array_equal(spectra["obs1"], np.array([[22.5, 22.5]]))

    def test_multi_component_parameter_builder_validates_shared_names(self):
        with self.assertRaises(ValueError):
            make_multi_component_parameters(
                {"temperature": {"min": 0, "max": 1}},
                shared_parameters=["missing"],
            )

    def test_round_kwargs_and_arcis_output_reset_are_scoped(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(
                tmpdir,
                "arcis_files",
                "mixingratios_round_2.dat",
            )
            os.makedirs(os.path.dirname(output_path))
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
            retrieval = Retrieval(ARCiS_binary, obs_type="emis")
            retrieval._prepare_simulator_round_outputs(kwargs)
            self.assertFalse(os.path.exists(output_path))

            with open(output_path, "w") as file:
                file.write("old")
            wrapped_arcis, _ = make_binary_simulator(
                ARCiS,
                {"temperature": {"min": 0, "max": 1}},
            )
            retrieval = Retrieval(wrapped_arcis, obs_type="emis")
            retrieval._prepare_simulator_round_outputs(kwargs)
            self.assertFalse(os.path.exists(output_path))

            with open(output_path, "w") as file:
                file.write("old")

            retrieval = Retrieval(lambda obs, pars: {}, obs_type="trans")
            retrieval._prepare_simulator_round_outputs(kwargs)
            self.assertTrue(os.path.exists(output_path))

    def test_arcis_input_is_frozen_once_for_run(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "arcis.in")
            arcis_output = os.path.join(tmpdir, "arcis_output")
            with open(input_path, "w") as file:
                file.write("makeai=.false.\noriginal=1\n")

            retrieval = Retrieval(ARCiS, obs_type="emis")
            simulator_kwargs = retrieval._freeze_arcis_input_for_run(
                {
                    "input_file": input_path,
                    "output_dir": arcis_output,
                }
            )

            frozen_path = os.path.join(arcis_output, "arcis_files", "arcis.in")
            with open(input_path, "w") as file:
                file.write("makeai=.true.\nmodified=1\n")
            with open(frozen_path) as file:
                frozen_contents = file.read()

        self.assertEqual(simulator_kwargs["input_file"], frozen_path)
        self.assertEqual(simulator_kwargs["original_input_file"], input_path)
        self.assertIn("makeai=.true.", frozen_contents)
        self.assertIn("original=1", frozen_contents)
        self.assertNotIn("modified=1", frozen_contents)

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

    def test_create_prior_uses_natural_parameter_bounds(self):
        class FakeBoxUniform:
            def __init__(self, low, high):
                self.low = low
                self.high = high

        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.parameters = {
            "temperature": {
                "min": 500,
                "max": 2500,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
            "log_h2o": {
                "min": -12,
                "max": -1,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }

        fake_sbi = types.SimpleNamespace(
            utils=types.SimpleNamespace(BoxUniform=FakeBoxUniform)
        )
        with patch.dict(sys.modules, {"sbi": fake_sbi}):
            retrieval.create_prior()

        np.testing.assert_allclose(retrieval.prior.low.numpy(), [500, -12])
        np.testing.assert_allclose(retrieval.prior.high.numpy(), [2500, -1])

    def test_sobol_initial_samples_are_immediately_scaled_to_natural_units(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.parameters = {
            "level": {
                "min": 10,
                "max": 20,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }

        retrieval.sobol_thetas(4)

        self.assertTrue(np.all(retrieval.thetas >= 10))
        self.assertTrue(np.all(retrieval.thetas <= 20))
        np.testing.assert_array_equal(retrieval.nat_thetas, retrieval.thetas)

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
                fitted_radii=np.array([1.0, 1.5]),
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
            loaded["fitted_radii"],
            np.array([1.0, 1.5]),
        )
        np.testing.assert_array_equal(
            loaded["spec"]["obs1"],
            np.array([[10.0, 20.0], [30.0, 40.0]]),
        )
        np.testing.assert_array_equal(
            loaded["post_spec"]["obs1"],
            np.array([[11.0, 21.0], [31.0, 41.0]]),
        )

    def test_output_manager_cleans_old_pre_round_checkpoints(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output = RetrievalOutput(tmpdir)
            for name in [
                "retrieval_pre_round_1.pkl",
                "retrieval_pre_round_2.pkl",
                "retrieval_pre_round_3.pkl",
            ]:
                with open(os.path.join(tmpdir, name), "w") as file:
                    file.write("checkpoint")

            output.cleanup_pre_round_checkpoints(
                keep_filename="retrieval_pre_round_3.pkl"
            )

            self.assertFalse(
                os.path.exists(os.path.join(tmpdir, "retrieval_pre_round_1.pkl"))
            )
            self.assertFalse(
                os.path.exists(os.path.join(tmpdir, "retrieval_pre_round_2.pkl"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "retrieval_pre_round_3.pkl"))
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
                nat_thetas=np.array([[2.0], [4.0]]),
                spectra={"obs": np.array([[2.0, 2.0], [4.0, 4.0]])},
            )
            retrieval._load_or_extend_reused_prior(
                reuse_prior=path,
                n_samples=2,
                sample_prior_method="random",
                n_threads=1,
                simulator_kwargs={},
            )

        np.testing.assert_array_equal(retrieval.thetas, np.array([[2.0], [4.0]]))
        np.testing.assert_array_equal(retrieval.nat_thetas, retrieval.thetas)
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

    def test_pca_transformer_reduces_arrays_and_tensors(self):
        x = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ])

        transformer = PCATransformer(2).fit(x)
        transformed = transformer.transform(x)
        reconstructed = transformer.inverse_transform(transformed)

        self.assertEqual(transformed.shape, (4, 2))
        np.testing.assert_allclose(reconstructed, x, atol=1e-12)
        self.assertEqual(transformer.requested_components, 2)
        self.assertEqual(transformer.n_components, 2)

        tensor_result = transformer.transform(torch.tensor(x, dtype=torch.float32))
        self.assertIsInstance(tensor_result, torch.Tensor)
        self.assertEqual(tuple(tensor_result.shape), (4, 2))

    def test_retrieval_pca_fits_once_after_preprocessing(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans", pca_components=2)
        retrieval.preprocessing = ["log"]
        x = torch.tensor([
            [1.0, 10.0, 100.0],
            [10.0, 100.0, 1000.0],
            [100.0, 1000.0, 10000.0],
        ])

        fitted = retrieval.do_preprocessing(x, fit_pca=True)
        default = retrieval.do_preprocessing(np.array([[10.0, 100.0, 1000.0]]))

        self.assertIsInstance(fitted, torch.Tensor)
        self.assertEqual(tuple(fitted.shape), (3, 2))
        self.assertEqual(default.shape, (1, 2))
        self.assertTrue(retrieval._pca_is_fitted())
        np.testing.assert_array_equal(retrieval.pca.mean_, np.array([[1.0, 2.0, 3.0]]))

    def test_pca_resume_requires_saved_transformer(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans", pca_components=2)

        with self.assertRaises(RuntimeError):
            retrieval._configure_pca(resume=True)

    def test_legacy_n_pca_configures_pca_components(self):
        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval._configure_pca(n_pca=3)

        self.assertEqual(retrieval.pca_components, 3)
        self.assertTrue(retrieval.do_pca)

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
        self.assertIsNone(payload["retrieval"]["pca_components"])
        self.assertFalse(payload["retrieval"]["pca_fitted"])
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

    def test_run_logs_default_sample_and_training_settings(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = os.path.join(tmpdir, "obs.txt")
            np.savetxt(obs_path, np.array([[1.0, 10.0, 0.1]]))

            retrieval = Retrieval(flat_simulator)
            retrieval.get_obs({"obs1": obs_path})
            retrieval.add_parameter("level", 0.0, 1.0)
            with patch.object(retrieval, "create_prior") as create_prior, patch.object(
                retrieval, "density_builder"
            ):
                create_prior.side_effect = lambda: setattr(
                    retrieval, "prior", DummyProposal(0.1)
                )
                retrieval.run(n_rounds=0, output_dir=tmpdir)

            with open(os.path.join(tmpdir, "retrieval_setup.json")) as file:
                payload = json.load(file)
            cloned_obs_path = os.path.join(tmpdir, "observations", "obs1_obs.txt")
            cloned_obs_exists = os.path.exists(cloned_obs_path)

        self.assertEqual(payload["run"]["n_samples"], 2048)
        self.assertEqual(payload["run"]["training_kwargs"]["learning_rate"], 1e-3)
        self.assertEqual(payload["run"]["training_kwargs"]["stop_after_epochs"], 20)
        self.assertEqual(payload["run"]["training_kwargs"]["num_atoms"], 20)
        self.assertTrue(cloned_obs_exists)
        self.assertEqual(payload["observations"]["obs1"]["cloned_source"], cloned_obs_path)

    def test_posterior_samples_are_saved_without_unit_cube_conversion(self):
        import tempfile
        import os

        retrieval = Retrieval(flat_simulator, obs_type="trans")
        retrieval.parameters = {
            "level": {
                "min": 10,
                "max": 20,
                "log": False,
                "post_processing": False,
                "universal": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            retrieval._save_posterior_samples(
                output_dir=tmpdir,
                posterior=DummyProposal(0.25),
                round_index=0,
            )
            path = os.path.join(tmpdir, "posterior_samples_round_1.txt")
            samples = np.loadtxt(path)
            with open(path) as file:
                header = file.readline()

        self.assertIn("level", header)
        self.assertEqual(samples.shape, (1000,))
        np.testing.assert_allclose(samples, np.full(1000, 0.25))

    def test_run_ensemble_reuses_prior_and_aggregates_outputs(self):
        import tempfile
        import os

        calls = []

        def fake_run(member, **kwargs):
            calls.append(kwargs)
            output = RetrievalOutput(kwargs["output_dir"])
            sim_output = kwargs["simulator_kwargs"]["output_dir"]
            atmosphere_dir = os.path.join(sim_output, "arcis_files")
            os.makedirs(atmosphere_dir, exist_ok=True)

            for round_index in range(kwargs["n_rounds"]):
                output.write_round_data(
                    round_index=round_index,
                    thetas=np.full((2, 1), round_index + len(calls)),
                    nat_thetas=np.full((2, 1), 10 + round_index + len(calls)),
                    spectra={"obs1": np.full((2, 2), round_index + len(calls))},
                    sample_sources=np.array(["prior", "proposal"]),
                )
                output.write_posterior_samples(
                    round_index + 1,
                    np.full((3, 1), round_index + len(calls)),
                    ["level"],
                )
                with open(
                    os.path.join(atmosphere_dir, f"mixingratios_round_{round_index}.dat"),
                    "w",
                ) as file:
                    file.write(f"member={len(calls)} round={round_index}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            retrieval = Retrieval(ARCiS)
            retrieval.parameters = {
                "level": {
                    "min": 0,
                    "max": 1,
                    "log": False,
                    "post_processing": False,
                    "universal": True,
                },
            }
            with patch.object(Retrieval, "run", fake_run):
                summary = retrieval.run_ensemble(
                    n_members=2,
                    n_rounds=2,
                    output_dir=tmpdir,
                    simulator_kwargs={"output_dir": "should_be_scoped"},
                )

            expected_prior = os.path.join(
                tmpdir,
                "member_001",
                "rounds",
                "round_000",
                "training_data.npz",
            )
            aggregate_samples = np.loadtxt(
                os.path.join(tmpdir, "aggregated", "posterior_samples_round_1.txt")
            )
            aggregate_data = RetrievalOutput.load_round_data(
                os.path.join(
                    tmpdir,
                    "aggregated",
                    "rounds",
                    "round_000",
                    "training_data.npz",
                )
            )
            with open(
                os.path.join(
                    tmpdir,
                    "aggregated",
                    "arcis_files",
                    "mixingratios_round_0.dat",
                )
            ) as file:
                atmosphere = file.read()

        self.assertIsNone(calls[0].get("reuse_prior"))
        self.assertEqual(calls[1]["reuse_prior"], expected_prior)
        self.assertEqual(
            calls[0]["simulator_kwargs"]["output_dir"],
            os.path.join(tmpdir, "member_001", "arcis_outputs"),
        )
        self.assertEqual(aggregate_samples.shape, (6,))
        self.assertEqual(aggregate_data["par"].shape, (4, 1))
        self.assertIn("ensemble_member=1", atmosphere)
        self.assertIn("ensemble_member=2", atmosphere)
        self.assertEqual(len(summary["aggregated"]["posterior_samples"]), 2)

    def test_run_ensemble_can_append_members(self):
        import tempfile
        import os

        calls = []

        def fake_run(member, **kwargs):
            calls.append(kwargs)
            output = RetrievalOutput(kwargs["output_dir"])
            sim_output = kwargs["simulator_kwargs"]["output_dir"]
            atmosphere_dir = os.path.join(sim_output, "arcis_files")
            os.makedirs(atmosphere_dir, exist_ok=True)
            output.write_round_data(
                round_index=0,
                thetas=np.full((2, 1), len(calls)),
                nat_thetas=np.full((2, 1), 10 + len(calls)),
                spectra={"obs1": np.full((2, 2), len(calls))},
                sample_sources=np.array(["prior", "proposal"]),
            )
            output.write_posterior_samples(1, np.full((3, 1), len(calls)), ["level"])
            with open(os.path.join(atmosphere_dir, "mixingratios_round_0.dat"), "w") as file:
                file.write(f"member={len(calls)}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            retrieval = Retrieval(ARCiS)
            retrieval.parameters = {
                "level": {
                    "min": 0,
                    "max": 1,
                    "log": False,
                    "post_processing": False,
                    "universal": True,
                },
            }
            with patch.object(Retrieval, "run", fake_run):
                retrieval.run_ensemble(
                    n_members=2,
                    n_rounds=1,
                    output_dir=tmpdir,
                    simulator_kwargs={},
                )
                summary = retrieval.run_ensemble(
                    n_members=2,
                    n_rounds=1,
                    output_dir=tmpdir,
                    resume=True,
                    add_members=True,
                    simulator_kwargs={},
                )

            expected_prior = os.path.join(
                tmpdir,
                "member_001",
                "rounds",
                "round_000",
                "training_data.npz",
            )
            aggregate_samples = np.loadtxt(
                os.path.join(tmpdir, "aggregated", "posterior_samples_round_1.txt")
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[2]["reuse_prior"], expected_prior)
        self.assertEqual(calls[3]["reuse_prior"], expected_prior)
        self.assertTrue(calls[2]["output_dir"].endswith("member_003"))
        self.assertTrue(calls[3]["output_dir"].endswith("member_004"))
        self.assertEqual(summary["n_members"], 4)
        self.assertEqual(len(summary["new_members"]), 2)
        self.assertEqual(aggregate_samples.shape, (12,))

    def test_from_setup_rebuilds_retrieval_and_run_config(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = os.path.join(tmpdir, "obs.txt")
            np.savetxt(obs_path, np.array([[1.0, 10.0, 0.1]]))
            setup_path = os.path.join(tmpdir, "retrieval_setup.json")
            payload = {
                "retrieval": {
                    "obs_type": "trans",
                    "simulator": "test_core.flat_simulator",
                    "preprocessing": ["log"],
                    "pca_components": None,
                    "error_inflation": 2,
                    "fit_radius": False,
                    "radius_reference": 1.0,
                },
                "observations": {
                    "obs1": {"source": obs_path},
                },
                "parameters": {
                    "level": {
                        "min": 0.0,
                        "max": 1.0,
                        "log": False,
                        "post_processing": False,
                        "universal": True,
                    },
                },
                "run": {
                    "n_rounds": 2,
                    "n_samples": 8,
                    "start_round": 0,
                    "final_round": 2,
                    "flow_kwargs": {"hidden": 16},
                    "simulator_kwargs": {"cache_dir": os.path.join(tmpdir, "missing")},
                },
            }
            with open(setup_path, "w") as file:
                json.dump(payload, file)

            with self.assertWarnsRegex(RuntimeWarning, "simulator_kwargs"):
                retrieval = Retrieval.from_setup(
                    setup_path,
                    run_overrides={
                        "n_rounds": 1,
                        "flow_kwargs": {"dropout": 0.1},
                    },
                )

        self.assertIs(retrieval.simulator, flat_simulator)
        self.assertEqual(retrieval.obs_type, "trans")
        self.assertEqual(retrieval.preprocessing, ["log"])
        self.assertEqual(retrieval.error_inflation, 2)
        self.assertEqual(list(retrieval.obs), ["obs1"])
        self.assertEqual(retrieval.parameters["level"]["max"], 1.0)
        self.assertEqual(retrieval.setup_run_config["n_rounds"], 1)
        self.assertNotIn("start_round", retrieval.setup_run_config)
        self.assertNotIn("final_round", retrieval.setup_run_config)
        self.assertEqual(retrieval.setup_run_config["flow_kwargs"]["hidden"], 16)
        self.assertEqual(retrieval.setup_run_config["flow_kwargs"]["dropout"], 0.1)

    def test_from_setup_warns_for_unavailable_simulator_and_missing_observation(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_path = os.path.join(tmpdir, "retrieval_setup.json")
            payload = {
                "retrieval": {
                    "obs_type": "trans",
                    "simulator": "missing.module.simulator",
                    "error_inflation": 1,
                },
                "observations": {
                    "obs": {"source": os.path.join(tmpdir, "missing_obs.txt")},
                },
                "parameters": {},
                "run": {},
            }
            with open(setup_path, "w") as file:
                json.dump(payload, file)

            with self.assertWarns(RuntimeWarning) as caught:
                with self.assertRaises(FileNotFoundError):
                    Retrieval.from_setup(setup_path)

        messages = "\n".join(str(item.message) for item in caught.warnings)
        self.assertIn("Could not import simulator", messages)
        self.assertIn("observation path", messages)

    def test_run_from_setup_applies_overrides(self):
        import tempfile
        import os

        calls = []

        def fake_run(self, **kwargs):
            calls.append(kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = os.path.join(tmpdir, "obs.txt")
            np.savetxt(obs_path, np.array([[1.0, 10.0, 0.1]]))
            setup_path = os.path.join(tmpdir, "retrieval_setup.json")
            payload = {
                "retrieval": {
                    "obs_type": "trans",
                    "simulator": "test_core.flat_simulator",
                    "error_inflation": 1,
                },
                "observations": {"obs": {"source": obs_path}},
                "parameters": {},
                "run": {
                    "n_rounds": 5,
                    "simulator_kwargs": {"output_dir": "old"},
                },
            }
            with open(setup_path, "w") as file:
                json.dump(payload, file)

            with patch.object(Retrieval, "run", fake_run):
                retrieval = Retrieval.run_from_setup(
                    setup_path,
                    n_rounds=1,
                    simulator_kwargs={"output_dir": "new"},
                )

        self.assertIsInstance(retrieval, Retrieval)
        self.assertEqual(calls[0]["n_rounds"], 1)
        self.assertEqual(calls[0]["simulator_kwargs"]["output_dir"], "new")

if __name__ == "__main__":
    unittest.main()
