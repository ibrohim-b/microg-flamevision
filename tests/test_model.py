"""
Unit tests for ml_model.py module.

Tests cover:
- Fuel type encoding and decoding
- Feature matrix preparation and validation
- Synthetic training data generation
- FlameHazardPredictor training and cross-validation
- Hazard level computation and assessment generation
- Model serialization (save/load roundtrip)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.ml_model import (
    FlameHazardPredictor,
    HazardAssessment,
    TrainingMetrics,
    FUEL_CODES,
    FUEL_NAMES,
    HAZARD_LEVELS,
    generate_synthetic_training_data,
)


class TestFuelEncoding(unittest.TestCase):
    """Test fuel string/integer mapping."""

    def setUp(self):
        self.predictor = FlameHazardPredictor()

    def test_fuel_encoding_consistency(self):
        for name, code in FUEL_CODES.items():
            encoded = self.predictor._encode_fuel(name)
            self.assertEqual(encoded, code)
            decoded = self.predictor._decode_fuel(encoded)
            self.assertEqual(decoded, name)

    def test_unknown_fuel_defaults(self):
        # Unknown fuel should default to heptane (0)
        self.assertEqual(self.predictor._encode_fuel("unknown_fuel"), 0)
        self.assertEqual(self.predictor._decode_fuel(999), "heptane")


class TestFeaturePreparation(unittest.TestCase):
    """Test feature scaling and shape verification."""

    def setUp(self):
        self.predictor = FlameHazardPredictor()

    def test_scalar_features(self):
        X = self.predictor._prepare_features(
            pressure_kpa=101.3,
            o2_percent=21.0,
            flow_velocity_mps=0.05,
            initial_diameter_mm=8.0,
            fuel_type="heptane",
        )
        self.assertEqual(X.shape, (1, 5))
        self.assertEqual(X[0, 4], FUEL_CODES["heptane"])

    def test_array_features(self):
        pressures = np.array([101.3, 80.0, 120.0])
        o2 = np.array([21.0, 18.0, 30.0])
        flow = np.array([0.05, 0.1, 0.2])
        diam = np.array([8.0, 6.0, 10.0])
        fuels = ["heptane", "methanol", "pmma"]

        X = self.predictor._prepare_features(pressures, o2, flow, diam, fuels)
        self.assertEqual(X.shape, (3, 5))
        self.assertEqual(X[1, 4], FUEL_CODES["methanol"])

    def test_mismatched_lengths_raise_error(self):
        with self.assertRaises(ValueError):
            self.predictor._prepare_features(
                pressure_kpa=[101.3, 80.0],
                o2_percent=[21.0],
                flow_velocity_mps=[0.05, 0.1],
                initial_diameter_mm=[8.0, 6.0],
                fuel_type=["heptane", "methanol"],
            )


class TestSyntheticDataAndTraining(unittest.TestCase):
    """Test data generation, model fitting, and serialization."""

    @classmethod
    def setUpClass(cls):
        # Use small sample for fast unit testing
        cls.X_train, cls.y_burn, cls.y_cool = generate_synthetic_training_data(n_samples=150, seed=42)

    def test_synthetic_data_properties(self):
        self.assertEqual(self.X_train.shape, (150, 5))
        self.assertEqual(len(self.y_burn), 150)
        self.assertEqual(len(self.y_cool), 150)
        # Burn time should be positive
        self.assertTrue(np.all(self.y_burn > 0))
        # Cool flame labels should be binary {0, 1}
        self.assertTrue(set(np.unique(self.y_cool)).issubset({0, 1}))

    def test_untrained_model_raises_error(self):
        predictor = FlameHazardPredictor()
        with self.assertRaises(RuntimeError):
            predictor.predict(self.X_train[:1])

    def test_train_and_predict(self):
        predictor = FlameHazardPredictor(n_estimators=20, max_depth=5, random_state=42)
        metrics = predictor.train(self.X_train, self.y_burn, self.y_cool, validate=True)

        self.assertTrue(predictor.is_trained)
        self.assertIsInstance(metrics, TrainingMetrics)
        self.assertGreater(metrics.classifier_accuracy, 0.5)

        burn_time, cool_prob, hazard_idx = predictor.predict(self.X_train[:5])
        self.assertEqual(len(burn_time), 5)
        self.assertEqual(len(cool_prob), 5)
        self.assertEqual(len(hazard_idx), 5)
        self.assertTrue(np.all(burn_time > 0))
        self.assertTrue(np.all((cool_prob >= 0.0) & (cool_prob <= 1.0)))
        self.assertTrue(np.all(np.isin(hazard_idx, [0, 1, 2, 3])))

    def test_assess_hazard_output_contract(self):
        predictor = FlameHazardPredictor(n_estimators=20, max_depth=5, random_state=42)
        predictor.train(self.X_train, self.y_burn, self.y_cool, validate=False)

        assessment = predictor.assess_hazard(
            pressure_kpa=101.3,
            o2_percent=21.0,
            flow_velocity_mps=0.05,
            initial_diameter_mm=8.0,
            fuel_type="heptane",
        )

        self.assertIsInstance(assessment, HazardAssessment)
        self.assertIn(assessment.hazard_level, HAZARD_LEVELS)
        self.assertGreater(assessment.predicted_time_to_extinction_s, 0)
        self.assertTrue(0 <= assessment.cool_flame_risk_pct <= 100)
        self.assertGreater(assessment.minimum_quench_oxygen_pct, 5.0)
        self.assertIsInstance(assessment.ventilation_recommendation, str)
        self.assertTrue(len(assessment.ventilation_recommendation) > 10)

    def test_model_persistence_roundtrip(self):
        predictor = FlameHazardPredictor(n_estimators=20, max_depth=5, random_state=42)
        predictor.train(self.X_train, self.y_burn, self.y_cool, validate=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.joblib"
            predictor.save(save_path)
            self.assertTrue(save_path.exists())

            loaded_predictor = FlameHazardPredictor()
            loaded_predictor.load(save_path)
            self.assertTrue(loaded_predictor.is_trained)

            # Assert outputs match exactly
            orig_burn, orig_cool, orig_haz = predictor.predict(self.X_train[:3])
            load_burn, load_cool, load_haz = loaded_predictor.predict(self.X_train[:3])

            np.testing.assert_array_almost_equal(orig_burn, load_burn)
            np.testing.assert_array_almost_equal(orig_cool, load_cool)
            np.testing.assert_array_equal(orig_haz, load_haz)


if __name__ == "__main__":
    unittest.main()
