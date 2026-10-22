"""Lightweight smoke tests for the public research-code sample.

These tests do not train models or require generated datasets. They verify the
main sampling/model interfaces and the exact Dirichlet endpoint enforcement.
Run from the repository root with:

    python -m unittest tests.test_smoke
"""

import unittest

import numpy as np
import torch

from data.sample_c import sample_c
from data.sample_u0 import sample_u0
from models.deeponet import create_deeponet1d
from models.fno1d import create_fno1d


class SamplingSmokeTest(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0.0, 1.0, 128)

    def test_initial_condition_sampling_is_finite_and_respects_boundaries(self):
        for distribution in ("train", "ood_frequency"):
            u0 = sample_u0(self.x, distribution=distribution, seed=2026)
            self.assertEqual(u0.shape, self.x.shape)
            self.assertTrue(np.all(np.isfinite(u0)))
            self.assertAlmostEqual(float(u0[0]), 0.0, places=12)
            self.assertAlmostEqual(float(u0[-1]), 0.0, places=12)

    def test_coefficient_sampling_is_finite_and_positive(self):
        for distribution in ("train", "ood_smooth", "ood_rough"):
            c = sample_c(self.x, distribution=distribution, seed=2026)
            self.assertEqual(c.shape, self.x.shape)
            self.assertTrue(np.all(np.isfinite(c)))
            self.assertTrue(np.all(c > 0.0))


class ModelSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.batch = 2
        self.n = 128
        self.grid = torch.linspace(0.0, 1.0, self.n).repeat(self.batch, 1)
        self.u0 = torch.randn(self.batch, self.n)
        self.c = 1.0 + 0.1 * torch.randn(self.batch, self.n)

    def test_fno_forward_shape_and_dirichlet_endpoints(self):
        model = create_fno1d(modes=16, width=16, num_layers=2, enforce_dirichlet=True)
        x = torch.stack([self.u0, self.c], dim=-1)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (self.batch, self.n, 1))
        self.assertTrue(torch.allclose(y[:, 0, 0], torch.zeros(self.batch), atol=1e-7))
        self.assertTrue(torch.allclose(y[:, -1, 0], torch.zeros(self.batch), atol=1e-7))

    def test_deeponet_forward_shape_and_dirichlet_endpoints(self):
        model = create_deeponet1d(
            N=self.n,
            branch_hidden=16,
            trunk_hidden=16,
            num_layers=2,
            enforce_dirichlet=True,
        )
        with torch.no_grad():
            y = model(self.u0, self.c, self.grid)
        self.assertEqual(tuple(y.shape), (self.batch, self.n))
        self.assertTrue(torch.allclose(y[:, 0], torch.zeros(self.batch), atol=1e-7))
        self.assertTrue(torch.allclose(y[:, -1], torch.zeros(self.batch), atol=1e-7))


if __name__ == "__main__":
    unittest.main()
