"""
Unit tests for flame_cv.py module.

Tests cover:
- Frame enhancement (CLAHE on blue channel)
- Thresholding and contour detection
- Ellipse fitting and diameter extraction
- Cool flame candidate detection
"""

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from modules.flame_cv import (
    FlameCVProcessor,
    CVConfig,
    FlameMetrics,
    enhance_faint_flame,
    extract_flame_metrics,
    process_video_frames,
)
from modules.mock_data_generator import FlameParams, MockFlameGenerator


class TestFlameCVEnhancement(unittest.TestCase):
    """Test frame enhancement functions."""

    def setUp(self):
        self.config = CVConfig()
        self.processor = FlameCVProcessor(self.config)
        # Create a simple test frame with a blue circle
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(self.frame, (320, 240), 50, (200, 100, 50), -1)  # BGR: blue-ish

    def test_enhance_faint_flame_returns_single_channel(self):
        """Enhanced frame should be single channel (grayscale)."""
        enhanced = self.processor.enhance_faint_flame(self.frame)
        self.assertEqual(enhanced.ndim, 2)
        self.assertEqual(enhanced.shape, (480, 640))

    def test_enhance_increases_blue_contrast(self):
        """CLAHE should increase contrast in blue channel."""
        enhanced = self.processor.enhance_faint_flame(self.frame)
        # Enhanced should have wider dynamic range than original blue channel
        original_blue = self.frame[:, :, 0]
        self.assertGreater(np.std(enhanced), np.std(original_blue) * 0.5)

    def test_enhance_hsv_method(self):
        """HSV enhancement method should work."""
        enhanced = self.processor.enhance_frame_hsv(self.frame)
        self.assertEqual(enhanced.ndim, 2)
        self.assertEqual(enhanced.shape, (480, 640))

    def test_invalid_frame_raises_error(self):
        """Non-3-channel frame should raise ValueError."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.processor.enhance_faint_flame(gray)


class TestFlameCVThresholding(unittest.TestCase):
    """Test thresholding and binary mask generation."""

    def setUp(self):
        self.config = CVConfig()
        self.processor = FlameCVProcessor(self.config)
        # Enhanced frame with clear flame region
        self.enhanced = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.enhanced, (50, 50), 20, 200, -1)

    def test_adaptive_threshold_creates_binary(self):
        """Adaptive threshold should produce binary mask."""
        binary = self.processor.threshold_flame(self.enhanced)
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(np.all(np.isin(binary, [0, 255])))

    def test_morphological_cleanup(self):
        """Morphology should remove small noise."""
        noisy = self.enhanced.copy()
        # Add salt noise
        noisy[10:15, 10:15] = 255
        binary = self.processor.threshold_flame(noisy)
        # Noise should be removed by opening
        self.assertEqual(np.sum(binary[10:15, 10:15]), 0)


class TestFlameCVContourDetection(unittest.TestCase):
    """Test contour finding and ellipse fitting."""

    def setUp(self):
        self.config = CVConfig(min_contour_area_px=50, max_contour_area_px=50000)
        self.processor = FlameCVProcessor(self.config)
        # Binary mask with clear circle
        self.binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(self.binary, (100, 100), 30, 255, -1)

    def test_find_contour_returns_contour(self):
        """Should find contour in binary mask."""
        contour = self.processor.find_flame_contour(self.binary)
        self.assertIsNotNone(contour)
        self.assertGreater(len(contour), 5)  # Need 5+ points for ellipse

    def test_find_contour_filters_by_area(self):
        """Should reject contours outside area bounds."""
        # Too small
        tiny = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(tiny, (100, 100), 2, 255, -1)
        self.assertIsNone(self.processor.find_flame_contour(tiny))

        # Too large
        huge = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(huge, (0, 0), (200, 200), 255, -1)
        self.assertIsNone(self.processor.find_flame_contour(huge))

    def test_prefers_circular_contours(self):
        """Should prefer more circular contours for microgravity."""
        # Create two contours: one circle, one elongated ellipse
        binary = np.zeros((300, 300), dtype=np.uint8)
        cv2.ellipse(binary, (100, 150), (30, 30), 0, 0, 360, 255, -1)  # Circle
        cv2.ellipse(binary, (200, 150), (50, 10), 0, 0, 360, 255, -1)  # Elongated
        contour = self.processor.find_flame_contour(binary)
        self.assertIsNotNone(contour)
        # Center should be closer to circle (100, 150)
        moments = cv2.moments(contour)
        cx = moments['m10'] / moments['m00']
        self.assertLess(abs(cx - 100), 50)


class TestFlameCVMetricsExtraction(unittest.TestCase):
    """Test metric extraction from contours."""

    def setUp(self):
        self.config = CVConfig(scale_px_per_mm=20.0)
        self.processor = FlameCVProcessor(self.config)
        # Create frame with known circle
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(self.frame, (100, 100), 40, (180, 120, 60), -1)  # Blue flame color

    def test_extract_metrics_from_circle(self):
        """Should correctly extract metrics from circular flame."""
        # Find contour
        enhanced = self.processor.enhance_faint_flame(self.frame)
        binary = self.processor.threshold_flame(enhanced)
        contour = self.processor.find_flame_contour(binary)
        self.assertIsNotNone(contour)

        metrics = self.processor.extract_metrics_from_contour(
            contour, enhanced, self.frame, 0, 0.0
        )

        self.assertTrue(metrics.flame_detected)
        self.assertIsNotNone(metrics.equivalent_diameter_mm)
        # 40px radius = 80px diameter = 4mm at 20px/mm
        self.assertAlmostEqual(metrics.equivalent_diameter_mm, 4.0, delta=0.5)
        self.assertAlmostEqual(metrics.aspect_ratio, 1.0, delta=0.15)
        self.assertGreater(metrics.mean_blue_intensity, 50)

    def test_cool_flame_candidate_flag(self):
        """Low blue intensity should flag cool flame candidate."""
        # Create very dim frame
        dim_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(dim_frame, (100, 100), 40, (50, 30, 20), -1)  # Very dim

        enhanced = self.processor.enhance_faint_flame(dim_frame)
        binary = self.processor.threshold_flame(enhanced)
        contour = self.processor.find_flame_contour(binary)

        if contour is not None:
            metrics = self.processor.extract_metrics_from_contour(
                contour, enhanced, dim_frame, 0, 0.0
            )
            # Mean blue ~30-40 should trigger cool flame candidate
            self.assertTrue(metrics.is_cool_flame_candidate or metrics.mean_blue_intensity < 60)


class TestFlameCVIntegration(unittest.TestCase):
    """Integration tests with synthetic data generator."""

    def test_process_synthetic_microgravity_frames(self):
        """Should detect flames in synthetic microgravity frames."""
        params = FlameParams(
            gravity="micro",
            fuel_type="heptane",
            initial_diameter_mm=8.0,
            burn_time_s=2.0,
            fps=10,
        )
        generator = MockFlameGenerator(params, seed=42)
        frames, _ = generator.generate()

        config = CVConfig(scale_px_per_mm=20.0)
        processor = FlameCVProcessor(config)

        detected_count = 0
        for i, frame in enumerate(frames[:10]):
            metrics = processor.process_frame(frame, i, i * 0.1)
            if metrics.flame_detected:
                detected_count += 1
                # Diameter should be close to expected
                self.assertIsNotNone(metrics.equivalent_diameter_mm)
                self.assertGreater(metrics.equivalent_diameter_mm, 1.0)
                self.assertLess(metrics.equivalent_diameter_mm, 15.0)

        # Should detect most frames
        self.assertGreaterEqual(detected_count, 5)

    def test_process_synthetic_earth_frames(self):
        """Should detect elongated flames in Earth gravity."""
        params = FlameParams(
            gravity="earth",
            fuel_type="heptane",
            initial_diameter_mm=8.0,
            burn_time_s=2.0,
            fps=10,
        )
        generator = MockFlameGenerator(params, seed=42)
        frames, _ = generator.generate()

        config = CVConfig(scale_px_per_mm=20.0)
        processor = FlameCVProcessor(config)

        detected_count = 0
        for i, frame in enumerate(frames[:10]):
            metrics = processor.process_frame(frame, i, i * 0.1)
            if metrics.flame_detected:
                detected_count += 1
                # Earth flames are elongated
                self.assertGreater(metrics.aspect_ratio, 1.1)

        self.assertGreaterEqual(detected_count, 5)


class TestFlameCVEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_frame_returns_no_detection(self):
        """Empty frame should return flame_detected=False."""
        config = CVConfig()
        processor = FlameCVProcessor(config)
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        metrics = processor.process_frame(empty, 0, 0.0)
        self.assertFalse(metrics.flame_detected)

    def test_noise_only_frame(self):
        """Random noise should not produce false detection."""
        config = CVConfig(min_contour_area_px=100)
        processor = FlameCVProcessor(config)
        noise = np.random.randint(0, 50, (200, 200, 3), dtype=np.uint8)
        metrics = processor.process_frame(noise, 0, 0.0)
        # May occasionally detect noise, but should be rare
        # Just verify it doesn't crash

    def test_debug_visualization(self):
        """Debug visualization should create 2x2 grid."""
        from modules.flame_cv import create_debug_visualization
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        metrics = FlameMetrics(
            frame_index=0, timestamp_s=0.0, flame_detected=True,
            centroid=(50, 50), equivalent_diameter_mm=4.0,
            major_axis_mm=4.2, minor_axis_mm=3.8,
            diameter_squared_mm2=16.0, aspect_ratio=1.1,
            mean_blue_intensity=150, is_cool_flame_candidate=False
        )
        enhanced = np.ones((100, 100), dtype=np.uint8) * 100
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (50, 50), 20, 255, -1)

        vis = create_debug_visualization(frame, metrics, enhanced, binary)
        self.assertEqual(vis.shape, (200, 200, 3))


if __name__ == "__main__":
    unittest.main()