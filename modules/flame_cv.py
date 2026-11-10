"""
Computer Vision Module for Microgravity Flame Detection

Implements frame enhancement (CLAHE on blue channel), adaptive thresholding,
contour detection, and ellipse fitting for flame diameter extraction.

Key Physics Considerations:
- Microgravity flames are faint blue (CH* 430nm, C2* 516nm) - standard luminance fails
- Must enhance BLUE channel specifically, not grayscale
- Spherical symmetry -> ellipse fitting with aspect ratio ~1.0
- Cool flames: near-infrared, extremely faint, require maximum sensitivity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class FlameMetrics:
    """Metrics extracted from a single flame frame."""
    frame_index: int
    timestamp_s: float
    flame_detected: bool
    centroid: tuple[float, float] | None = None
    major_axis_mm: float | None = None
    minor_axis_mm: float | None = None
    equivalent_diameter_mm: float | None = None
    diameter_squared_mm2: float | None = None
    aspect_ratio: float | None = None
    mean_blue_intensity: float | None = None
    is_cool_flame_candidate: bool = False
    contour_area_px: float | None = None
    ellipse_angle_deg: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "flame_detected": self.flame_detected,
            "centroid": list(self.centroid) if self.centroid else None,
            "major_axis_mm": self.major_axis_mm,
            "minor_axis_mm": self.minor_axis_mm,
            "equivalent_diameter_mm": self.equivalent_diameter_mm,
            "diameter_squared_mm2": self.diameter_squared_mm2,
            "aspect_ratio": self.aspect_ratio,
            "mean_blue_intensity": self.mean_blue_intensity,
            "is_cool_flame_candidate": self.is_cool_flame_candidate,
            "contour_area_px": self.contour_area_px,
            "ellipse_angle_deg": self.ellipse_angle_deg,
        }


@dataclass
class CVConfig:
    """Configuration for flame CV pipeline."""
    scale_px_per_mm: float = 20.0
    min_contour_area_px: int = 50
    max_contour_area_px: int = 50000
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    blue_channel_weight: float = 1.0  # Weight for blue channel in enhancement
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0
    otsu_fallback: bool = True
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    min_aspect_ratio: float = 0.5  # Reject extremely elongated (non-flame)
    max_aspect_ratio: float = 2.0
    cool_flame_blue_threshold: int = 60  # Mean blue intensity for cool flame flag


class FlameCVProcessor:
    """
    Processes video frames to detect and measure microgravity flames.

    Usage:
        processor = FlameCVProcessor(CVConfig(scale_px_per_mm=20.0))
        metrics = processor.process_frame(frame, frame_idx=0, timestamp=0.0)
        print(metrics.equivalent_diameter_mm)
    """

    def __init__(self, config: CVConfig | None = None):
        self.config = config or CVConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )

    def enhance_faint_flame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Enhance faint microgravity flame using CLAHE on blue channel.

        Microgravity flames emit primarily in blue (CH* 430nm, C2* 516nm).
        Standard grayscale (0.299R + 0.587G + 0.114B) suppresses blue channel.
        This method isolates and enhances the blue channel.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            Enhanced single-channel image (blue-enhanced)
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be 3-channel BGR image")

        # Split channels
        b, g, r = cv2.split(frame)

        # Apply CLAHE to blue channel (primary for microgravity flames)
        b_enhanced = self._clahe.apply(b)

        # Also enhance luminance in LAB space for additional contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, lab_b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, lab_b])
        lab_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        _, _, r_lab = cv2.split(lab_bgr)

        # Combine: weighted blue channel + enhanced luminance info
        # Blue channel carries the flame signal; LAB L-channel adds structural contrast
        combined = cv2.addWeighted(
            b_enhanced,
            self.config.blue_channel_weight,
            r_lab,
            1.0 - self.config.blue_channel_weight,
            0,
        )

        return combined

    def enhance_frame_hsv(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Alternative enhancement using HSV color space.

        Useful for cool flames where blue channel alone may be insufficient.
        Enhances Value channel after isolating blue hue range.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Blue hue range in OpenCV HSV (H: 0-179): ~100-130 for blue
        blue_mask = cv2.inRange(h, 90, 130)

        # Apply CLAHE to Value channel only where blue hue present
        v_enhanced = self._clahe.apply(v)
        v_combined = np.where(blue_mask > 0, v_enhanced, v)

        hsv_enhanced = cv2.merge([h, s, v_combined])
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        b_enhanced, _, _ = cv2.split(bgr_enhanced)

        return b_enhanced

    def threshold_flame(self, enhanced: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Threshold enhanced image to isolate flame region.

        Uses adaptive Gaussian thresholding for varying illumination,
        falls back to Otsu if adaptive fails.
        """
        if self.config.use_adaptive_threshold:
            # Adaptive threshold handles vignetting/chamber glare gradients
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_block_size,
                self.config.adaptive_c,
            )
        else:
            binary = np.zeros_like(enhanced)

        # Fallback to Otsu if adaptive produces too few/few many pixels
        if self.config.otsu_fallback:
            _, otsu_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Check if adaptive result is reasonable (not all black/white)
            white_ratio = np.mean(binary > 0)
            if white_ratio < 0.001 or white_ratio > 0.5:
                binary = otsu_binary

        # Morphological cleanup
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            self._morph_kernel,
            iterations=self.config.morph_iterations,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=self.config.morph_iterations,
        )

        # Remove small connected components (salt noise & sensor artifacts)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_contour_area_px:
                binary[labels == i] = 0

        return binary

    def find_flame_contour(self, binary: NDArray[np.uint8]) -> NDArray[np.int32] | None:
        """
        Find the best flame contour from binary mask.

        Returns the largest contour within area bounds that satisfies
        aspect ratio constraints for spherical/near-spherical flames.
        """
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        # Filter by area
        valid_contours = [
            c for c in contours
            if self.config.min_contour_area_px <= cv2.contourArea(c) <= self.config.max_contour_area_px
        ]

        if not valid_contours:
            return None

        # For microgravity: select most circular contour (aspect ratio ~ 1)
        # For terrestrial: may be elongated, so relax constraint
        best_contour = None
        best_score = -1

        for contour in valid_contours:
            if len(contour) < 5:
                continue  # Need at least 5 points for fitEllipse

            area = cv2.contourArea(contour)
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (ma, ma), angle = ellipse
                major = max(ma, ma)
                minor = min(ma, ma)
                aspect = major / minor if minor > 0 else 999

                # Score: prefer larger area, aspect ratio near 1
                aspect_score = 1.0 / (1.0 + abs(aspect - 1.0) * 2)
                area_score = min(1.0, area / 5000.0)  # Normalize
                score = 0.7 * aspect_score + 0.3 * area_score

                if score > best_score:
                    best_score = score
                    best_contour = contour
            except cv2.error:
                continue

        return best_contour

    def extract_metrics_from_contour(
        self,
        contour: NDArray[np.int32],
        enhanced_frame: NDArray[np.uint8],
        original_frame: NDArray[np.uint8],
        frame_index: int,
        timestamp_s: float,
    ) -> FlameMetrics:
        """Extract all flame metrics from a fitted ellipse."""
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (minor_axis_px, major_axis_px), angle = ellipse

        # Ensure major >= minor
        if minor_axis_px > major_axis_px:
            major_axis_px, minor_axis_px = minor_axis_px, major_axis_px

        area_px = cv2.contourArea(contour)
        equiv_diameter_px = 2 * math.sqrt(area_px / math.pi)

        # Convert to mm
        scale = self.config.scale_px_per_mm
        major_mm = major_axis_px / scale
        minor_mm = minor_axis_px / scale
        equiv_mm = equiv_diameter_px / scale
        d2_mm2 = equiv_mm ** 2
        aspect = major_mm / minor_mm if minor_mm > 0 else 1.0

        # Mean blue intensity in original frame (flame region)
        mask = np.zeros(original_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_blue = cv2.mean(original_frame[:, :, 0], mask=mask)[0]

        # Cool flame candidate: low blue intensity but detected
        is_cool = mean_blue < self.config.cool_flame_blue_threshold and equiv_mm > 1.0

        return FlameMetrics(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            flame_detected=True,
            centroid=(float(cx), float(cy)),
            major_axis_mm=round(major_mm, 3),
            minor_axis_mm=round(minor_mm, 3),
            equivalent_diameter_mm=round(equiv_mm, 3),
            diameter_squared_mm2=round(d2_mm2, 3),
            aspect_ratio=round(aspect, 3),
            mean_blue_intensity=round(mean_blue, 1),
            is_cool_flame_candidate=is_cool,
            contour_area_px=round(area_px, 1),
            ellipse_angle_deg=round(angle, 1),
        )

    def process_frame(
        self,
        frame: NDArray[np.uint8],
        frame_index: int = 0,
        timestamp_s: float = 0.0,
        enhancement_method: Literal["blue_clahe", "hsv"] = "blue_clahe",
    ) -> FlameMetrics:
        """
        Process a single frame and extract flame metrics.

        Args:
            frame: BGR image
            frame_index: Frame number in sequence
            timestamp_s: Time in seconds
            enhancement_method: "blue_clahe" (default) or "hsv"

        Returns:
            FlameMetrics object (flame_detected=False if no flame found)
        """
        if enhancement_method == "hsv":
            enhanced = self.enhance_frame_hsv(frame)
        else:
            enhanced = self.enhance_faint_flame(frame)

        binary = self.threshold_flame(enhanced)
        contour = self.find_flame_contour(binary)

        if contour is None:
            return FlameMetrics(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                flame_detected=False,
            )

        return self.extract_metrics_from_contour(
            contour, enhanced, frame, frame_index, timestamp_s
        )

    def process_video(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        sample_every: int = 1,
        enhancement_method: Literal["blue_clahe", "hsv"] = "blue_clahe",
    ) -> list[FlameMetrics]:
        """
        Process entire video file and return metrics for each frame.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            sample_every: Process every N-th frame (for performance)
            enhancement_method: Enhancement algorithm to use

        Returns:
            List of FlameMetrics for each processed frame
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default fallback

        metrics_list = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                timestamp = frame_idx / fps
                metrics = self.process_frame(frame, frame_idx, timestamp, enhancement_method)
                metrics_list.append(metrics)
                processed += 1

                if max_frames and processed >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        return metrics_list


def enhance_faint_flame(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convenience function: enhance faint flame using default FlameCVProcessor.

    Applies CLAHE on the Blue channel to isolate CH* and C2* chemiluminescence.
    """
    processor = FlameCVProcessor()
    return processor.enhance_faint_flame(frame)


def extract_flame_metrics(
    frame: NDArray[np.uint8],
    scale_px_per_mm: float = 20.0,
    min_area: int = 50,
) -> dict:
    """
    Convenience function: extract flame metrics dictionary from a single frame.

    Args:
        frame: BGR image frame
        scale_px_per_mm: Calibration scale (pixels per mm)
        min_area: Minimum contour area threshold in pixels

    Returns:
        Flame metrics dictionary conforming to project schema.
    """
    config = CVConfig(scale_px_per_mm=scale_px_per_mm, min_contour_area_px=min_area)
    processor = FlameCVProcessor(config)
    metrics = processor.process_frame(frame, frame_idx=0, timestamp_s=0.0)
    return metrics.to_dict()


def process_video_frames(
    video_path: str | Path,
    config: CVConfig | None = None,
    **kwargs,
) -> list[FlameMetrics]:
    """
    Convenience function to process video with default or custom config.

    Args:
        video_path: Path to video file
        config: Optional CVConfig
        **kwargs: Passed to FlameCVProcessor.process_video

    Returns:
        List of FlameMetrics
    """
    processor = FlameCVProcessor(config)
    return processor.process_video(video_path, **kwargs)


def create_debug_visualization(
    frame: NDArray[np.uint8],
    metrics: FlameMetrics,
    enhanced: NDArray[np.uint8] | None = None,
    binary: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """
    Create a debug visualization showing original, enhanced, binary, and overlay.

    Returns:
        2x2 grid visualization image
    """
    h, w = frame.shape[:2]
    vis = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # Top-left: Original
    vis[:h, :w] = frame

    # Top-right: Enhanced (if provided) or grayscale
    if enhanced is not None:
        vis[:h, w:] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis[:h, w:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Bottom-left: Binary mask
    if binary is not None:
        vis[h:, :w] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        vis[h:, :w] = np.zeros((h, w, 3), dtype=np.uint8)

    # Bottom-right: Overlay with ellipse and metrics
    overlay = frame.copy()
    if metrics.flame_detected and metrics.centroid:
        cx, cy = int(metrics.centroid[0]), int(metrics.centroid[1])
        # Draw centroid
        cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
        # Draw ellipse approximation
        if metrics.major_axis_mm and metrics.minor_axis_mm:
            scale = 20.0  # px/mm
            major_px = int(metrics.major_axis_mm * scale)
            minor_px = int(metrics.minor_axis_mm * scale)
            cv2.ellipse(overlay, (cx, cy), (major_px // 2, minor_px // 2),
                        0, 0, 360, (0, 255, 0), 2)

        # Add text
        text_lines = [
            f"D_eq: {metrics.equivalent_diameter_mm:.2f} mm" if metrics.equivalent_diameter_mm else "No flame",
            f"Aspect: {metrics.aspect_ratio:.2f}" if metrics.aspect_ratio else "",
            f"Blue: {metrics.mean_blue_intensity:.0f}" if metrics.mean_blue_intensity else "",
            "COOL FLAME" if metrics.is_cool_flame_candidate else "",
        ]
        for i, line in enumerate(text_lines):
            if line:
                cv2.putText(overlay, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    vis[h:, w:] = overlay

    return vis


if __name__ == "__main__":
    # Quick self-test with synthetic data
    from mock_data_generator import generate_synthetic_video, FlameParams, MockFlameGenerator

    print("Testing flame_cv.py with synthetic microgravity flame...")

    # Generate test frames
    params = FlameParams(
        gravity="micro",
        fuel_type="heptane",
        initial_diameter_mm=8.0,
        burn_time_s=5.0,
        fps=10,
    )
    generator = MockFlameGenerator(params, seed=42)
    frames, telemetry = generator.generate()

    # Process frames
    processor = FlameCVProcessor()
    results = []

    for i, (frame, tel) in enumerate(zip(frames[:20], telemetry[:20])):
        metrics = processor.process_frame(frame, i, tel["timestamp_s"])
        results.append(metrics)
        status = "DETECTED" if metrics.flame_detected else "MISSING"
        print(f"  Frame {i:2d} (t={tel['timestamp_s']:.2f}s): {status} "
              f"D_eq={metrics.equivalent_diameter_mm}mm "
              f"Blue={metrics.mean_blue_intensity}")

    # Summary
    detected = [r for r in results if r.flame_detected]
    print(f"\nDetected {len(detected)}/{len(results)} frames")
    if detected:
        avg_dia = np.mean([r.equivalent_diameter_mm for r in detected])
        print(f"Average diameter: {avg_dia:.2f} mm")
        print(f"Expected: ~8.0 mm (initial)")

    print("Test complete.")