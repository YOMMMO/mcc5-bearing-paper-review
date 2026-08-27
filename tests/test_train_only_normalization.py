"""Tests for train-only sequence and scalar normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.train_only_normalization import (
    fit_sequence_stats,
    scalar_training_audit,
    sequence_training_audit,
    standardize_scalar_frames,
    standardize_sequence_splits,
    transform_sequences,
)


class SequenceNormalizationTests(unittest.TestCase):
    def test_train_only_channel_zscore(self) -> None:
        rng = np.random.default_rng(7)
        train = np.stack(
            [rng.normal(4.0, 2.0, size=(12, 40)), rng.normal(-8.0, 11.0, size=(12, 40))],
            axis=1,
        ).astype(np.float32)
        val = (train[:3] + 100.0).copy()
        test = (train[3:6] - 50.0).copy()
        x_train, x_val, x_test, stats = standardize_sequence_splits(train, val, test)

        audit = sequence_training_audit(x_train, stats)
        self.assertTrue(np.all(np.abs(audit["transformed_mean"]) < 1e-5))
        self.assertTrue(np.all(np.abs(audit["transformed_std"] - 1.0) < 1e-5))
        np.testing.assert_allclose(x_val, (val - stats.mean) / stats.scale, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x_test, (test - stats.mean) / stats.scale, rtol=1e-6, atol=1e-6)

    def test_constant_channel_and_serialized_stats(self) -> None:
        train = np.zeros((5, 2, 8), dtype=np.float32)
        train[:, 1, :] = np.arange(8, dtype=np.float32)
        stats = fit_sequence_stats(train)
        transformed = transform_sequences(train, stats)
        self.assertTrue(bool(stats.constant_mask.reshape(-1)[0]))
        self.assertTrue(np.allclose(transformed[:, 0, :], 0.0))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.npz"
            np.savez(path, mean=stats.mean, scale=stats.scale)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["mean"], stats.mean)
                np.testing.assert_array_equal(saved["scale"], stats.scale)


class ScalarNormalizationTests(unittest.TestCase):
    def test_imputation_and_feature_zscore(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
        val = pd.DataFrame({"a": [100.0, np.nan], "b": [5.0, 5.0]})
        test = pd.DataFrame({"a": [-10.0], "b": [5.0]})
        x_train, x_val, x_test, stats = standardize_scalar_frames(train, val, test, ["a", "b"])
        audit = scalar_training_audit(x_train, stats)

        nonconstant = ~audit["constant"]
        self.assertTrue(np.all(np.abs(audit.loc[nonconstant, "transformed_mean"]) < 1e-6))
        self.assertTrue(np.all(np.abs(audit.loc[nonconstant, "transformed_std"] - 1.0) < 1e-6))
        self.assertTrue(bool(audit.loc[audit["feature"] == "b", "constant"].iloc[0]))
        self.assertTrue(np.isfinite(x_val).all())
        self.assertTrue(np.isfinite(x_test).all())
        self.assertEqual(float(stats.median["a"]), 2.0)


if __name__ == "__main__":
    unittest.main()
