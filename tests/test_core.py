import unittest
import numpy as np
import torch
from torch.distributions import Normal

from floppity.helpers import (
    create_obs_file,
    convert_cube,
    compute_moments,
    find_MAP,
    reduced_chi_squared,
    W2_distance,
)

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

    def test_W2_distance(self):
        class MockProposal:
            def sample(self, shape):
                return torch.randn(*shape)

        proposals = [MockProposal(), MockProposal()]
        w2, w2_err = W2_distance(proposals, n_mc=10, n_draws=100)
        self.assertGreaterEqual(w2, 0)
        self.assertGreaterEqual(w2_err, 0)

if __name__ == "__main__":
    unittest.main()