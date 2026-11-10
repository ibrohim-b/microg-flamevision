"""
Synthetic Microgravity Combustion Data Generator

Generates realistic synthetic CIR video frames and telemetry CSV files
for testing without requiring access to NASA PSI archives.

Physics Basis:
- Microgravity flames: spherical, diffusion-dominated, faint blue (CH*/C2* chemiluminescence)
- Terrestrial flames: elongated teardrop, buoyant convection, bright yellow/orange (soot incandescence)
- d²-law: D²(t) = D₀² - K·t for droplet combustion
- Cool flame transition: secondary regression regime at lower temperatures
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# Type aliases
GravityLevel = Literal["micro", "lunar", "mars", "earth"]
FuelType = Literal["heptane", "methanol", "hexane", "pmma", "ethylene"]


@dataclass
class FlameParams:
    """Parameters defining a synthetic flame simulation."""
    gravity: GravityLevel = "micro"
    fuel_type: FuelType = "heptane"
    initial_diameter_mm: float = 8.0
    pressure_kpa: float = 101.3
    o2_percent: float = 21.0
    flow_velocity_mps: float = 0.05
    burn_time_s: float = 30.0
    fps: int = 10
    frame_size: tuple[int, int] = (640, 480)
    noise_level: float = 0.15
    cool_flame_transition: bool = False
    cool_flame_start_frac: float = 0.7  # Fraction of burn time when cool flame begins


# Physics constants for different fuels (approximate K values in mm²/s at 101.3 kPa, 21% O₂)
FUEL_BURN_RATES: dict[FuelType, float] = {
    "heptane": 0.85,
    "methanol": 0.62,
    "hexane": 0.78,
    "pmma": 0.45,  # Solid fuel, effective regression rate
    "ethylene": 1.10,
}

# Gravity multipliers for buoyancy effect on burning rate
GRAVITY_MULTIPLIERS: dict[GravityLevel, float] = {
    "micro": 1.0,      # Pure diffusion
    "lunar": 1.15,     # 0.16g - weak buoyancy
    "mars": 1.30,      # 0.38g - moderate buoyancy
    "earth": 1.80,     # 1g - strong buoyant convection
}

# Flame color profiles (BGR for OpenCV)
FLAME_COLORS: dict[GravityLevel, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # (core_color, outer_color) - BGR format
    "micro": ((200, 100, 50), (180, 60, 20)),      # Blue-dominant (CH* 430nm, C2* 516nm)
    "lunar": ((180, 120, 60), (150, 80, 30)),      # Blue-green transition
    "mars": ((100, 150, 100), (60, 120, 50)),      # Green-yellow transition
    "earth": ((30, 180, 220), (20, 120, 200)),     # Yellow-orange (soot incandescence)
}

COOL_FLAME_COLOR = ((80, 180, 200), (60, 150, 180))  # Faint blue-white (near-IR/blue)


class MockFlameGenerator:
    """
    Generates synthetic microgravity combustion video frames and telemetry.

    Usage:
        generator = MockFlameGenerator(FlameParams(gravity="micro", fuel_type="heptane"))
        frames, telemetry = generator.generate()
        generator.save_video(frames, "test_microgravity.mp4")
        generator.save_telemetry(telemetry, "test_telemetry.csv")
    """

    def __init__(self, params: FlameParams | None = None, seed: int | None = 42):
        self.params = params or FlameParams()
        self.rng = np.random.default_rng(seed)
        self._validate_params()

    def _validate_params(self) -> None:
        if self.params.gravity not in GRAVITY_MULTIPLIERS:
            raise ValueError(f"Unknown gravity level: {self.params.gravity}")
        if self.params.fuel_type not in FUEL_BURN_RATES:
            raise ValueError(f"Unknown fuel type: {self.params.fuel_type}")

    def _compute_effective_burn_rate(self) -> float:
        """Compute effective K accounting for pressure, O₂, gravity, and flow."""
        base_k = FUEL_BURN_RATES[self.params.fuel_type]
        gravity_mult = GRAVITY_MULTIPLIERS[self.params.gravity]

        # Pressure effect: K ∝ P^0.5 approximately
        pressure_factor = math.sqrt(self.params.pressure_kpa / 101.3)

        # Oxygen effect: roughly linear above 15%
        o2_factor = max(0.3, self.params.o2_percent / 21.0)

        # Forced flow effect: enhances burning rate
        flow_factor = 1.0 + 2.0 * self.params.flow_velocity_mps  # Linear approx

        return base_k * gravity_mult * pressure_factor * o2_factor * flow_factor

    def _diameter_at_time(self, t: float, k_eff: float) -> float:
        """Compute flame diameter at time t using d²-law with optional cool flame transition."""
        d0 = self.params.initial_diameter_mm

        if not self.params.cool_flame_transition:
            # Standard d²-law
            d2 = max(0.0, d0**2 - k_eff * t)
            return math.sqrt(d2)

        # Two-stage: primary burn then cool flame
        t_transition = self.params.burn_time_s * self.params.cool_flame_start_frac
        k_cool = k_eff * 0.15  # Cool flame burns ~15% as fast

        if t <= t_transition:
            d2 = max(0.0, d0**2 - k_eff * t)
        else:
            d_transition = math.sqrt(max(0.0, d0**2 - k_eff * t_transition))
            d2 = max(0.0, d_transition**2 - k_cool * (t - t_transition))

        return math.sqrt(d2)

    def _draw_flame_frame(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        diameter_px: float,
        is_cool_flame: bool = False,
    ) -> None:
        """Draw a synthetic flame on the frame using gradient ellipses."""
        h, w = frame.shape[:2]
        cx, cy = center

        # Select color profile
        if is_cool_flame:
            core_color, outer_color = COOL_FLAME_COLOR
        else:
            core_color, outer_color = FLAME_COLORS[self.params.gravity]

        # For microgravity: symmetric sphere
        # For earth: elongated teardrop (buoyant stretch)
        gravity_stretch = {
            "micro": 1.0,
            "lunar": 1.1,
            "mars": 1.25,
            "earth": 1.6,
        }[self.params.gravity]

        # Flow adds horizontal stretch
        flow_stretch = 1.0 + 3.0 * self.params.flow_velocity_mps

        major_axis = diameter_px / 2 * gravity_stretch
        minor_axis = diameter_px / 2

        # Draw multiple concentric ellipses for gradient effect
        num_layers = 12
        for i in range(num_layers):
            frac = i / (num_layers - 1)
            # Interpolate color
            color = tuple(
                int(outer_color[c] + (core_color[c] - outer_color[c]) * (1 - frac**2))
                for c in range(3)
            )
            # Size decreases toward center
            scale = 1.0 - 0.7 * frac
            axes = (int(major_axis * scale), int(minor_axis * scale))

            # Add slight centroid drift for terrestrial flames (buoyant flicker)
            drift_x = 0
            drift_y = 0
            if self.params.gravity == "earth" and not is_cool_flame:
                drift_y = int(-5 * frac * (1 + self.rng.normal(0, 0.3)))
                drift_x = int(3 * self.rng.normal(0, 0.2))

            cv2.ellipse(
                frame,
                (cx + drift_x, cy + drift_y),
                axes,
                0, 0, 360,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

        # Add core hot spot for earth gravity
        if self.params.gravity == "earth" and not is_cool_flame:
            cv2.circle(frame, (cx, cy - int(minor_axis * 0.3)), int(minor_axis * 0.15),
                       (50, 200, 255), -1, lineType=cv2.LINE_AA)

    def _add_background_noise(self, frame: np.ndarray) -> None:
        """Add chamber background glare and sensor noise."""
        h, w = frame.shape[:2]

        # Vignetting / chamber wall glare
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = math.sqrt(center_x**2 + center_y**2)
        vignette = 1.0 - 0.3 * (dist / max_dist)**2

        # Apply vignette to blue channel (chamber windows often blue-tinted)
        frame[:, :, 0] = (frame[:, :, 0] * vignette).astype(np.uint8)

        # Add Gaussian noise
        noise = self.rng.normal(0, 15 * self.params.noise_level, frame.shape).astype(np.int16)
        frame = cv2.add(frame, noise.astype(np.uint8))

        # Add occasional "hot pixels" (cosmic ray hits)
        if self.rng.random() < 0.02:
            num_hot = self.rng.integers(1, 5)
            for _ in range(num_hot):
                hy, hx = self.rng.integers(0, h), self.rng.integers(0, w)
                frame[hy, hx] = [255, 255, 255]

    def generate_frame(self, t: float, k_eff: float, frame_idx: int) -> np.ndarray:
        """Generate a single synthetic flame frame at time t."""
        h, w = self.params.frame_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Diameter in pixels (scale: 20 px/mm as per spec)
        scale_px_per_mm = 20.0
        diameter_mm = self._diameter_at_time(t, k_eff)
        diameter_px = diameter_mm * scale_px_per_mm

        # Check for cool flame transition
        is_cool_flame = (
            self.params.cool_flame_transition
            and t > self.params.burn_time_s * self.params.cool_flame_start_frac
            and diameter_mm > 0.5
        )

        # Centroid (center of frame with slight drift)
        cx = w // 2 + int(self.rng.normal(0, 2))
        cy = h // 2 + int(self.rng.normal(0, 2))

        if diameter_mm > 0.3:  # Only draw if flame exists
            self._draw_flame_frame(frame, (cx, cy), diameter_px, is_cool_flame)

        self._add_background_noise(frame)
        return frame

    def generate(self) -> tuple[list[np.ndarray], list[dict]]:
        """
        Generate complete video frames and synchronized telemetry.

        Returns:
            frames: List of BGR frames (numpy arrays)
            telemetry: List of dicts matching the NASA PSI CSV schema
        """
        k_eff = self._compute_effective_burn_rate()
        num_frames = int(self.params.burn_time_s * self.params.fps)
        dt = 1.0 / self.params.fps

        frames = []
        telemetry = []

        for i in range(num_frames):
            t = i * dt
            frame = self.generate_frame(t, k_eff, i)
            frames.append(frame)

            # Telemetry record
            diameter_mm = self._diameter_at_time(t, k_eff)
            is_cool = (
                self.params.cool_flame_transition
                and t > self.params.burn_time_s * self.params.cool_flame_start_frac
            )

            # Radiometer signal: proportional to flame surface area * emissivity
            # Microgravity: low soot -> low radiometer signal
            # Earth: high soot -> high signal
            base_emissivity = {"micro": 0.15, "lunar": 0.25, "mars": 0.4, "earth": 0.85}[self.params.gravity]
            if is_cool:
                base_emissivity *= 0.1  # Cool flames barely radiate

            flame_area = math.pi * (diameter_mm / 2)**2
            radiometer = base_emissivity * flame_area * 0.001  # W/cm² scale

            # Thermocouple temperature
            if is_cool:
                tc_temp = 600 + self.rng.normal(0, 50)  # Cool flame ~600-800°C
            else:
                base_temp = {"micro": 900, "lunar": 1000, "mars": 1150, "earth": 1600}[self.params.gravity]
                tc_temp = base_temp + self.rng.normal(0, 100)

            # Chamber pressure slight drop during combustion
            pressure = self.params.pressure_kpa - 0.5 * (t / self.params.burn_time_s) + self.rng.normal(0, 0.05)

            # O₂ consumption
            o2_consumed = min(2.0, 0.05 * t)  # Up to 2% drop
            o2_current = max(10.0, self.params.o2_percent - o2_consumed + self.rng.normal(0, 0.02))

            telemetry.append({
                "timestamp_s": round(t, 3),
                "chamber_pressure_kpa": round(pressure, 2),
                "o2_percent": round(o2_current, 2),
                "cabin_flow_mps": round(self.params.flow_velocity_mps + self.rng.normal(0, 0.005), 3),
                "fuel_type": self.params.fuel_type,
                "radiometer_w_cm2": round(radiometer, 5),
                "thermocouple_temp_c": round(tc_temp, 1),
                "flame_diameter_mm": round(diameter_mm, 3),
                "is_cool_flame": is_cool,
            })

            if diameter_mm < 0.3:
                # Flame extinguished - add a few more frames of empty chamber
                for j in range(3):
                    t_ext = t + j * dt
                    empty_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    self._add_background_noise(empty_frame)
                    frames.append(empty_frame)
                    telemetry.append({
                        "timestamp_s": round(t_ext, 3),
                        "chamber_pressure_kpa": round(pressure, 2),
                        "o2_percent": round(o2_current, 2),
                        "cabin_flow_mps": round(self.params.flow_velocity_mps, 3),
                        "fuel_type": self.params.fuel_type,
                        "radiometer_w_cm2": 0.0,
                        "thermocouple_temp_c": round(25 + self.rng.normal(0, 2), 1),
                        "flame_diameter_mm": 0.0,
                        "is_cool_flame": False,
                    })
                break

        return frames, telemetry

    def save_video(self, frames: list[np.ndarray], output_path: str | Path, fps: int | None = None) -> None:
        """Save frames as MP4 video."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not frames:
            raise ValueError("No frames to save")

        h, w = frames[0].shape[:2]
        fps = fps or self.params.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame)
        writer.release()

    def save_telemetry(self, telemetry: list[dict], output_path: str | Path) -> None:
        """Save telemetry as CSV matching NASA PSI schema."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp_s",
            "chamber_pressure_kpa",
            "o2_percent",
            "cabin_flow_mps",
            "fuel_type",
            "radiometer_w_cm2",
            "thermocouple_temp_c",
            "flame_diameter_mm",
            "is_cool_flame",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in telemetry:
                writer.writerow(row)


def generate_synthetic_video(
    output_path: str | Path,
    gravity: GravityLevel = "micro",
    fuel_type: FuelType = "heptane",
    initial_diameter_mm: float = 8.0,
    pressure_kpa: float = 101.3,
    o2_percent: float = 21.0,
    flow_velocity_mps: float = 0.05,
    burn_time_s: float = 30.0,
    fps: int = 10,
    cool_flame_transition: bool = False,
    seed: int | None = 42,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Convenience function to generate and save synthetic video + telemetry.

    Args:
        output_path: Base path (without extension) for .mp4 and .csv files
        gravity: Gravity level ("micro", "lunar", "mars", "earth")
        fuel_type: Fuel type ("heptane", "methanol", "hexane", "pmma", "ethylene")
        initial_diameter_mm: Initial flame/droplet diameter in mm
        pressure_kpa: Chamber pressure in kPa
        o2_percent: Oxygen percentage
        flow_velocity_mps: Forced ventilation velocity in m/s
        burn_time_s: Total burn time to simulate
        fps: Frames per second
        cool_flame_transition: Whether to simulate cool flame regime
        seed: Random seed for reproducibility

    Returns:
        (frames, telemetry) tuple
    """
    params = FlameParams(
        gravity=gravity,
        fuel_type=fuel_type,
        initial_diameter_mm=initial_diameter_mm,
        pressure_kpa=pressure_kpa,
        o2_percent=o2_percent,
        flow_velocity_mps=flow_velocity_mps,
        burn_time_s=burn_time_s,
        fps=fps,
        cool_flame_transition=cool_flame_transition,
    )
    generator = MockFlameGenerator(params, seed=seed)
    frames, telemetry = generator.generate()

    base = Path(output_path)
    generator.save_video(frames, base.with_suffix(".mp4"))
    generator.save_telemetry(telemetry, base.with_suffix(".csv"))

    return frames, telemetry


def generate_telemetry_csv(
    output_path: str | Path,
    num_samples: int = 100,
    seed: int | None = 42,
) -> None:
    """
    Generate a CSV file with varied telemetry samples for ML training.

    Creates diverse combinations of gravity, fuel, pressure, O₂, flow
    with computed burn rates and extinction times.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    gravities: list[GravityLevel] = ["micro", "lunar", "mars", "earth"]
    fuels: list[FuelType] = ["heptane", "methanol", "hexane", "pmma", "ethylene"]

    fieldnames = [
        "gravity",
        "fuel_type",
        "pressure_kpa",
        "o2_percent",
        "flow_velocity_mps",
        "initial_diameter_mm",
        "burning_rate_k_mm2_s",
        "burn_time_s",
        "cool_flame_risk",
        "extinction_mechanism",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_samples):
            gravity = rng.choice(gravities)
            fuel = rng.choice(fuels)
            pressure = rng.uniform(50, 150)
            o2 = rng.uniform(14, 34)
            flow = rng.uniform(0.0, 0.5)
            d0 = rng.uniform(3, 12)

            # Compute effective K
            base_k = FUEL_BURN_RATES[fuel]
            gravity_mult = GRAVITY_MULTIPLIERS[gravity]
            pressure_factor = math.sqrt(pressure / 101.3)
            o2_factor = max(0.3, o2 / 21.0)
            flow_factor = 1.0 + 2.0 * flow

            k_eff = base_k * gravity_mult * pressure_factor * o2_factor * flow_factor
            burn_time = (d0**2) / k_eff if k_eff > 0 else 0

            # Cool flame risk: higher in microgravity, low O₂, low flow
            cool_risk = 0.0
            if gravity == "micro":
                cool_risk += 0.4
            if o2 < 18:
                cool_risk += 0.3
            if flow < 0.05:
                cool_risk += 0.2
            if fuel in ["heptane", "hexane"]:  # Larger hydrocarbons more prone
                cool_risk += 0.1
            cool_risk = min(1.0, cool_risk + rng.normal(0, 0.05))

            extinction = "radiative" if gravity == "micro" else "buoyant_blowoff"
            if cool_risk > 0.5:
                extinction = "cool_flame_transition"

            writer.writerow({
                "gravity": gravity,
                "fuel_type": fuel,
                "pressure_kpa": round(pressure, 1),
                "o2_percent": round(o2, 1),
                "flow_velocity_mps": round(flow, 3),
                "initial_diameter_mm": round(d0, 2),
                "burning_rate_k_mm2_s": round(k_eff, 3),
                "burn_time_s": round(burn_time, 1),
                "cool_flame_risk": round(cool_risk, 3),
                "extinction_mechanism": extinction,
            })


if __name__ == "__main__":
    # Demo: Generate sample data for all gravity levels
    output_dir = Path(__file__).parent.parent / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic microgravity combustion data...")

    for grav in ["micro", "lunar", "mars", "earth"]:
        print(f"  Generating {grav}g heptane flame...")
        generate_synthetic_video(
            output_dir / f"heptane_{grav}g",
            gravity=grav,  # type: ignore
            fuel_type="heptane",
            initial_diameter_mm=8.0,
            burn_time_s=25.0,
            fps=10,
            cool_flame_transition=(grav == "micro"),
        )

    # Generate cool flame transition case
    print("  Generating microgravity cool flame transition...")
    generate_synthetic_video(
        output_dir / "heptane_microg_cool_flame",
        gravity="micro",
        fuel_type="heptane",
        initial_diameter_mm=8.0,
        burn_time_s=40.0,
        fps=10,
        cool_flame_transition=True,
    )

    # Generate training telemetry CSV
    print("  Generating ML training telemetry...")
    generate_telemetry_csv(output_dir / "training_telemetry.csv", num_samples=500)

    print(f"Done! Output saved to {output_dir}")