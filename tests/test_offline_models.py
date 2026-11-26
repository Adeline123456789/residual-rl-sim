import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(package) for package in ("torch", "numpy"))
if HAS_RUNTIME:
    import numpy as np
    import torch

    from cf_resrl.models import fit_probe, load_probe, predict_probe, save_probe


@unittest.skipUnless(HAS_RUNTIME, "offline probe requires NumPy and PyTorch")
class OfflineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = np.random.default_rng(42)
        self.context = rng.normal(size=(64, 3)).astype(np.float32)
        self.residual = rng.normal(size=(64, 2)).astype(np.float32) * 0.1 + 0.2
        self.zero = self.context[:, 0] * 0.1
        self.delta = self.zero + self.residual[:, 0] - 2 * self.residual[:, 1]
        self.options = {"steps": 120, "batch_size": 16, "learning_rate": 0.01, "seed": 1234, "width": 12}

    def fit(self, method):
        return fit_probe(method, self.context, self.residual, self.zero, self.delta, **self.options)

    def test_b5_and_p1_match_capacity_and_label_scaling(self):
        b5, p1 = (self.fit(method) for method in ("B5", "P1"))
        self.assertEqual(b5.parameter_count, p1.parameter_count)
        for key in ("target_mean", "target_scale", "residual_mean", "residual_scale"):
            self.assertTrue(torch.equal(getattr(b5, key), getattr(p1, key)))
        for probe in (b5, p1):
            predicted = predict_probe(probe, self.context, self.residual)
            np.testing.assert_allclose(
                predicted["g_zero"] + predicted["interaction"], predicted["g_delta"], atol=1e-7,
            )
            self.assertLess(np.mean(np.abs(predicted["g_delta"] - self.delta)), 0.06)
            self.assertLess(np.mean(np.abs(predicted["g_zero"] - self.zero)), 0.06)

    def test_p1_interaction_is_exactly_zero_at_raw_zero_residual(self):
        probe = self.fit("P1")
        prediction = predict_probe(probe, self.context, np.zeros_like(self.residual))
        np.testing.assert_array_equal(prediction["interaction"], np.zeros(len(self.context)))
        np.testing.assert_array_equal(prediction["g_delta"], prediction["g_zero"])

    def test_pair_only_probe_ignores_zero_residual_labels(self):
        without_g0 = fit_probe("B6", self.context, self.residual, None, self.delta, **self.options)
        for probe in (self.fit("B6"), without_g0):
            self.assertEqual(probe.parameter_count, without_g0.parameter_count)
            result = predict_probe(probe, self.context, self.residual)
            self.assertEqual(set(result), {"g_delta"})
            np.testing.assert_array_equal(
                result["g_delta"], predict_probe(without_g0, self.context, self.residual)["g_delta"],
            )

    def test_norm_omits_direction_and_checkpoint_roundtrip(self):
        probe = self.fit("Norm")
        positive = predict_probe(probe, self.context, self.residual)
        negative = predict_probe(probe, self.context, -self.residual)
        np.testing.assert_array_equal(positive["g_delta"], negative["g_delta"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            save_probe(probe, path)
            restored = load_probe(path)
        self.assertEqual(restored.method, "Norm")
        self.assertEqual(restored.parameter_count, probe.parameter_count)
        np.testing.assert_array_equal(
            predict_probe(restored, self.context, self.residual)["g_delta"], positive["g_delta"],
        )

    def test_linear_full_fits_directional_gain(self):
        probe = self.fit("Linear-full")
        gains = predict_probe(probe, self.context, self.residual)
        self.assertEqual(set(gains), {"g_delta"})
        self.assertLess(np.mean(np.abs(gains["g_delta"] - self.delta)), 0.05)
