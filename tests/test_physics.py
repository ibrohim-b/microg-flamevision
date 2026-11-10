"""
Unit tests for physics_engine.py module.

Tests cover:
- Savitzky-Golay smoothing on D² signals
- d²-law regression and burning rate (K) calculation
- Extinction event detection and classification
- Grashof number analysis (microgravity vs terrestrial regimes)
- Cool flame secondary regime detection
- Minimum oxygen index (MOI) calculations
- Full FlameMetrics physics analysis pipeline
"""

from __future__ import annotations

import unittest
import numpy as np

from modules.flame_cv import FlameMetrics
from modules.physics_engine import (
    smooth_d2_signal,
    compute_d2_burn_rate,
    detect_extinction_event,
    compute_grashof_number,
    detect_cool_flame_transition,
    predict_extinction_time,
    compute_minimum_oxygen_index,
    analyze_flame_metrics,
    D2LawResult,
    ExtinctionEvent,
    GrashofAnalysis,
)


class TestD2SignalSmoothing(unittest.TestCase):
    """Test Savitzky-Golay filtering and bounds enforcement on D² signals."""

    def test_smooth_d2_signal_preserves_shape(self):
        t = np.linspace(0, 10, 50)
        d = np.linspace(8, 2, 50)
        smoothed = smooth_d2_signal(t, d, window_length=11, polyorder=3)
        self.assertEqual(len(smoothed), len(t))
        self.assertTrue(np.all(smoothed >= 0))

    def test_smooth_d2_small_dataset_fallback(self):
        t = np.linspace(0, 1, 4)
        d = np.array([5.0, 4.0, 3.0, 2.0])
        # With < 5 points, it should warn and return raw d^2
        smoothed = smooth_d2_signal(t, d, window_length=11)
        np.testing.assert_array_almost_equal(smoothed, d ** 2)

    def test_smooth_d2_attenuates_noise(self):
        np.random.seed(42)
        t = np.linspace(0, 10, 60)
        d_clean = np.sqrt(np.maximum(64.0 - 5.0 * t, 0))
        noise = np.random.normal(0, 0.5, len(t))
        d_noisy = np.maximum(d_clean + noise, 0)

        smoothed = smooth_d2_signal(t, d_noisy, window_length=9, polyorder=2)
        clean_d2 = d_clean ** 2

        # Error of smoothed should be lower than raw noisy error
        noisy_err = np.mean(np.abs((d_noisy ** 2) - clean_d2))
        smooth_err = np.mean(np.abs(smoothed - clean_d2))
        self.assertLess(smooth_err, noisy_err)


class TestD2BurnRateRegression(unittest.TestCase):
    """Test linear regression fitting for droplet burning rate K."""

    def setUp(self):
        np.random.seed(123)
        self.t = np.linspace(0, 20, 100)
        self.k_true = 0.85  # mm²/s
        self.d0_sq_true = 64.0  # mm² (D0 = 8 mm)
        d2_ideal = self.d0_sq_true - self.k_true * self.t
        d2_noisy = d2_ideal + np.random.normal(0, 0.4, len(self.t))
        self.diameters = np.sqrt(np.maximum(d2_noisy, 0.01))

    def test_compute_d2_burn_rate_accuracy(self):
        result = compute_d2_burn_rate(self.t, self.diameters)
        self.assertIsInstance(result, D2LawResult)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.burning_rate_k, self.k_true, delta=0.08)
        self.assertAlmostEqual(result.d0_squared_mm2, self.d0_sq_true, delta=3.0)
        self.assertGreater(result.r_squared, 0.90)
        self.assertEqual(result.quality_flag, "GOOD")

    def test_compute_d2_burn_rate_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compute_d2_burn_rate([0.0, 1.0], [5.0])

    def test_compute_d2_burn_rate_insufficient_points(self):
        with self.assertRaises(ValueError):
            compute_d2_burn_rate([0.0, 1.0], [5.0, 4.0], min_points=10)


class TestExtinctionDetection(unittest.TestCase):
    """Test flame extinction timestamp detection and mechanism classification."""

    def test_extinction_detected_when_diameter_drops(self):
        t = np.linspace(0, 10, 50)
        # 5mm flame for 30 frames, then drops to 0.1mm for remaining 20 frames
        d = np.concatenate([np.full(30, 5.0), np.full(20, 0.1)])
        event = detect_extinction_event(d, t, threshold_mm=0.5, min_consecutive=3)

        self.assertIsNotNone(event)
        self.assertIsInstance(event, ExtinctionEvent)
        self.assertAlmostEqual(event.extinction_time_s, t[30], delta=0.25)
        self.assertGreaterEqual(event.confidence, 0.7)

    def test_extinction_none_if_sustained(self):
        t = np.linspace(0, 10, 50)
        d = np.full(50, 4.0)
        event = detect_extinction_event(d, t, threshold_mm=0.5)
        self.assertIsNone(event)

    def test_extinction_cool_flame_transition_classification(self):
        t = np.linspace(0, 15, 60)
        # Flame drops from 6mm to 1.0mm (between threshold_mm=0.5 and cool_flame_threshold=1.5)
        d = np.concatenate([np.linspace(6.0, 0.8, 30), np.full(30, 0.8)])
        # Use threshold_mm = 0.9 to trigger below-threshold detection
        event = detect_extinction_event(d, t, threshold_mm=0.9, cool_flame_threshold_mm=1.5)
        self.assertIsNotNone(event)
        self.assertEqual(event.mechanism, "cool_flame_transition")


class TestGrashofAnalysis(unittest.TestCase):
    """Test Grashof number calculations and convective regime classifications."""

    def test_microgravity_is_diffusion_dominated(self):
        analysis = compute_grashof_number(
            gravity_m_s2=1e-6,
            flame_temp_c=900.0,
            ambient_temp_c=25.0,
            characteristic_length_m=0.01,
        )
        self.assertIsInstance(analysis, GrashofAnalysis)
        self.assertLess(analysis.grashof_number, 0.1)
        self.assertEqual(analysis.regime, "diffusion_dominated")

    def test_terrestrial_is_buoyancy_dominated(self):
        analysis = compute_grashof_number(
            gravity_m_s2=9.81,
            flame_temp_c=1600.0,
            ambient_temp_c=25.0,
            characteristic_length_m=0.02,
        )
        self.assertGreater(analysis.grashof_number, 100.0)
        self.assertEqual(analysis.regime, "buoyancy_dominated")

    def test_transitional_regime(self):
        analysis = compute_grashof_number(
            gravity_m_s2=0.01,
            flame_temp_c=600.0,
            ambient_temp_c=25.0,
            characteristic_length_m=0.005,
        )
        self.assertTrue(0.1 <= analysis.grashof_number <= 100.0)
        self.assertEqual(analysis.regime, "transitional")


class TestCoolFlameTransition(unittest.TestCase):
    """Test two-stage combustion transition detection."""

    def test_cool_flame_transition_detected(self):
        t = np.linspace(0, 30, 90)
        # Stage 1: fast burning (K = 1.0)
        d2_stage1 = 64.0 - 1.0 * t[:45]
        # Stage 2: slow cool flame burning (K = 0.1)
        d2_stage2 = d2_stage1[-1] - 0.1 * (t[45:] - t[45])
        d = np.sqrt(np.maximum(np.concatenate([d2_stage1, d2_stage2]), 0.01))

        detected, trans_t, sec_k = detect_cool_flame_transition(t, d)
        self.assertTrue(detected)
        self.assertIsNotNone(trans_t)
        self.assertIsNotNone(sec_k)
        self.assertLess(sec_k, 0.3)

    def test_cool_flame_not_detected_on_single_slope(self):
        t = np.linspace(0, 20, 60)
        d2 = np.maximum(50.0 - 1.5 * t, 0.1)
        d = np.sqrt(d2)
        detected, _, _ = detect_cool_flame_transition(t, d)
        self.assertFalse(detected)


class TestExtinctionPredictionAndMOI(unittest.TestCase):
    """Test analytical extinction time prediction and Minimum Oxygen Index."""

    def test_predict_extinction_time(self):
        t = np.linspace(0, 10, 40)
        d = np.sqrt(np.maximum(40.0 - 2.0 * t, 0.1))
        d2_res = compute_d2_burn_rate(t, d)

        # At t=5s, d^2 ≈ 30, remaining time = 30 / 2 = 15s -> predicted extinction = 20s
        current_time = 5.0
        current_dia = np.sqrt(30.0)
        t_ext = predict_extinction_time(d2_res, current_time, current_dia)
        self.assertIsNotNone(t_ext)
        self.assertAlmostEqual(t_ext, 20.0, delta=2.5)

    def test_compute_minimum_oxygen_index(self):
        moi_heptane_micro = compute_minimum_oxygen_index("heptane", pressure_kpa=101.3, gravity="micro")
        moi_heptane_earth = compute_minimum_oxygen_index("heptane", pressure_kpa=101.3, gravity="earth")
        self.assertGreater(moi_heptane_micro, 10.0)
        self.assertLess(moi_heptane_micro, 20.0)
        # Buoyancy in Earth gravity reduces MOI compared to microgravity
        self.assertLess(moi_heptane_earth, moi_heptane_micro)


class TestAnalyzeFlameMetricsIntegration(unittest.TestCase):
    """Test full analysis orchestration on FlameMetrics lists."""

    def test_analyze_flame_metrics_pipeline(self):
        metrics = []
        for i in range(50):
            t = i * 0.2
            dia = max(8.0 - 0.12 * t, 0.2)
            metrics.append(FlameMetrics(
                frame_index=i,
                timestamp_s=t,
                flame_detected=True,
                centroid=(320.0, 240.0),
                equivalent_diameter_mm=dia,
                diameter_squared_mm2=dia**2,
                mean_blue_intensity=120.0,
            ))

        result = analyze_flame_metrics(metrics)
        self.assertIn("d2_law", result)
        self.assertIn("extinction", result)
        self.assertIn("grashof", result)
        self.assertIn("cool_flame", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["detected_frames"], 50)


if __name__ == "__main__":
    unittest.main()
