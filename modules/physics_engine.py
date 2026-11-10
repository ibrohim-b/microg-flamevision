"""
Physics Engine for Microgravity Combustion Analysis

Implements:
- d²-law regression (Savitzky-Golay smoothing + linear fit over steady regime)
- Extinction event detection
- Grashof number analysis (buoyancy vs diffusion)
- Cool flame transition detection
- Burning rate constant (K) calculation with uncertainty
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats

from .flame_cv import FlameMetrics


@dataclass
class D2LawResult:
    """Results from d²-law regression analysis."""
    burning_rate_k: float  # mm²/s (positive value)
    burning_rate_k_stderr: float  # Standard error of K
    r_squared: float
    d0_squared_mm2: float  # Initial D² intercept
    d0_squared_stderr: float
    steady_start_idx: int
    steady_end_idx: int
    timestamps_used: NDArray[np.float64]
    d2_smoothed: NDArray[np.float64]
    d2_raw: NDArray[np.float64]
    residuals: NDArray[np.float64]
    is_valid: bool
    quality_flag: str  # "GOOD", "FAIR", "POOR"


@dataclass
class ExtinctionEvent:
    """Detected flame extinction event."""
    extinction_time_s: float
    extinction_frame_idx: int
    mechanism: Literal["radiative", "diffusion_limited", "blowoff", "cool_flame_transition", "unknown"]
    confidence: float  # 0-1
    pre_extinction_diameter_mm: float


@dataclass
class GrashofAnalysis:
    """Grashof number analysis for buoyancy vs diffusion regime."""
    grashof_number: float
    regime: Literal["diffusion_dominated", "transitional", "buoyancy_dominated"]
    characteristic_length_m: float
    temperature_diff_k: float
    kinematic_viscosity_m2_s: float
    thermal_expansion_coeff: float


def smooth_d2_signal(
    timestamps: NDArray[np.float64],
    diameters: NDArray[np.float64],
    window_length: int = 11,
    polyorder: int = 3,
) -> NDArray[np.float64]:
    """
    Apply Savitzky-Golay filter to D²(t) signal.

    Args:
        timestamps: Time array (seconds)
        diameters: Diameter array (mm)
        window_length: Filter window (must be odd, < len(data))
        polyorder: Polynomial order (< window_length)

    Returns:
        Smoothed D² values
    """
    d2 = diameters ** 2
    n = len(d2)

    # Adjust window length if too large
    if window_length >= n:
        window_length = n - 1 if n % 2 == 0 else n - 2
    if window_length < 5:
        # Too few points for smoothing, return raw
        warnings.warn("Too few data points for Savitzky-Golay smoothing, returning raw D²")
        return d2

    # Ensure odd window length
    if window_length % 2 == 0:
        window_length -= 1

    # Ensure polyorder < window_length
    polyorder = min(polyorder, window_length - 1)

    try:
        d2_smooth = signal.savgol_filter(d2, window_length, polyorder)
        # Ensure physically valid (no negative D²)
        d2_smooth = np.maximum(d2_smooth, 0)
        return d2_smooth
    except Exception:
        warnings.warn("Savitzky-Golay failed, returning raw D²")
        return d2


def compute_d2_burn_rate(
    timestamps: list[float] | NDArray[np.float64],
    diameters: list[float] | NDArray[np.float64],
    steady_fraction: tuple[float, float] = (0.15, 0.90),
    min_points: int = 10,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
) -> D2LawResult:
    """
    Compute burning rate constant K from d²-law regression.

    Classical d²-law: D²(t) = D₀² - K·t
    Fits linear regression over steady burning regime (excluding ignition transient
    and extinction phase).

    Args:
        timestamps: Time values in seconds
        diameters: Equivalent flame diameters in mm
        steady_fraction: (start_frac, end_frac) of time range for steady regime
        min_points: Minimum data points required for valid fit
        smooth_window: Savitzky-Golay window length
        smooth_polyorder: Savitzky-Golay polynomial order

    Returns:
        D2LawResult with K, R², uncertainties, and quality assessment
    """
    t = np.asarray(timestamps, dtype=np.float64)
    d = np.asarray(diameters, dtype=np.float64)

    if len(t) != len(d):
        raise ValueError("Timestamps and diameters must have same length")
    if len(t) < min_points:
        raise ValueError(f"Need at least {min_points} data points, got {len(t)}")

    # Sort by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    d = d[sort_idx]

    # Compute raw D²
    d2_raw = d ** 2

    # Smooth D² signal
    d2_smooth = smooth_d2_signal(t, d, smooth_window, smooth_polyorder)

    # Determine steady burning regime indices
    t_min, t_max = t[0], t[-1]
    t_range = t_max - t_min
    steady_start_t = t_min + steady_fraction[0] * t_range
    steady_end_t = t_min + steady_fraction[1] * t_range

    steady_mask = (t >= steady_start_t) & (t <= steady_end_t)
    steady_indices = np.where(steady_mask)[0]

    if len(steady_indices) < min_points:
        # Fallback: use middle 75% of data
        n = len(t)
        steady_indices = np.arange(n // 8, 7 * n // 8)

    steady_start_idx = steady_indices[0]
    steady_end_idx = steady_indices[-1]

    t_steady = t[steady_indices]
    d2_steady = d2_smooth[steady_indices]

    # Linear regression: D² = D₀² - K·t
    # Using scipy.stats.linregress for uncertainty estimates
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_steady, d2_steady)

    # Burning rate K = -slope (positive value)
    k = -slope
    k_stderr = std_err

    # Quality assessment
    r2 = r_value ** 2
    n_steady = len(t_steady)

    if r2 > 0.95 and n_steady >= 20 and k > 0:
        quality = "GOOD"
    elif r2 > 0.85 and n_steady >= 10 and k > 0:
        quality = "FAIR"
    else:
        quality = "POOR"

    is_valid = k > 0 and r2 > 0.5 and n_steady >= 5

    # Residuals
    d2_pred = intercept + slope * t
    residuals = d2_smooth - d2_pred

    return D2LawResult(
        burning_rate_k=round(float(k), 4),
        burning_rate_k_stderr=round(float(k_stderr), 4),
        r_squared=round(float(r2), 4),
        d0_squared_mm2=round(float(intercept), 4),
        d0_squared_stderr=round(float(std_err * np.sqrt(np.mean(t_steady**2))), 4),
        steady_start_idx=int(steady_start_idx),
        steady_end_idx=int(steady_end_idx),
        timestamps_used=t_steady,
        d2_smoothed=d2_smooth,
        d2_raw=d2_raw,
        residuals=residuals,
        is_valid=is_valid,
        quality_flag=quality,
    )


def detect_extinction_event(
    diameters: list[float] | NDArray[np.float64],
    timestamps: list[float] | NDArray[np.float64],
    threshold_mm: float = 0.5,
    min_consecutive: int = 3,
    cool_flame_threshold_mm: float = 1.5,
) -> ExtinctionEvent | None:
    """
    Detect the exact frame/time of flame extinction.

    Identifies when flame diameter permanently falls below threshold.
    Handles cool flame transition (diameter drops then stabilizes at low value).

    Args:
        diameters: Diameter time series (mm)
        timestamps: Corresponding timestamps (s)
        threshold_mm: Diameter below which flame is considered extinguished
        min_consecutive: Consecutive frames below threshold to confirm extinction
        cool_flame_threshold_mm: If diameter stabilizes above this after drop, it's cool flame

    Returns:
        ExtinctionEvent or None if no clear extinction detected
    """
    d = np.asarray(diameters)
    t = np.asarray(timestamps)

    if len(d) != len(t) or len(d) == 0:
        return None

    # Find first index where diameter drops below threshold
    below_threshold = d < threshold_mm

    if not np.any(below_threshold):
        return None  # Never went below threshold

    # Find runs of consecutive below-threshold frames
    extinction_idx = None
    consecutive = 0

    for i, below in enumerate(below_threshold):
        if below:
            consecutive += 1
            if consecutive >= min_consecutive:
                extinction_idx = i - min_consecutive + 1
                break
        else:
            consecutive = 0

    if extinction_idx is None:
        return None

    # Check for cool flame transition: diameter drops then stabilizes at low non-zero value
    post_extinction = d[extinction_idx:]
    if len(post_extinction) > 5:
        stable_diameter = np.median(post_extinction[:10])
        if cool_flame_threshold_mm > stable_diameter > 0.1:
            mechanism = "cool_flame_transition"
            confidence = 0.8
        else:
            mechanism = "radiative"
            confidence = 0.9
    else:
        mechanism = "radiative"
        confidence = 0.7

    # Check for blowoff (sudden drop with high flow - would need flow data)
    # For now, use diameter derivative
    if extinction_idx > 2:
        pre_slope = (d[extinction_idx] - d[extinction_idx - 2]) / (t[extinction_idx] - t[extinction_idx - 2])
        if pre_slope < -5.0:  # Very rapid diameter decrease
            mechanism = "blowoff"
            confidence = 0.75

    pre_dia = d[max(0, extinction_idx - 1)]

    return ExtinctionEvent(
        extinction_time_s=float(t[extinction_idx]),
        extinction_frame_idx=int(extinction_idx),
        mechanism=mechanism,  # type: ignore
        confidence=confidence,
        pre_extinction_diameter_mm=float(pre_dia),
    )


def compute_grashof_number(
    gravity_m_s2: float,
    flame_temp_c: float,
    ambient_temp_c: float,
    characteristic_length_m: float = 0.01,
    pressure_kpa: float = 101.3,
) -> GrashofAnalysis:
    """
    Compute Grashof number to quantify buoyancy vs diffusion dominance.

    Gr = g * β * (T_flame - T_ambient) * L³ / ν²

    Where:
    - g: gravitational acceleration (m/s²)
    - β: thermal expansion coefficient ≈ 1/T_film (K⁻¹)
    - T_flame, T_ambient: temperatures in Kelvin
    - L: characteristic length (m) - flame diameter
    - ν: kinematic viscosity (m²/s)

    Regimes:
    - Gr << 1: Diffusion dominated (microgravity)
    - Gr ~ 1-100: Transitional
    - Gr >> 100: Buoyancy dominated (terrestrial)
    """
    # Film temperature for property evaluation
    t_film_k = (flame_temp_c + ambient_temp_c) / 2 + 273.15
    t_flame_k = flame_temp_c + 273.15
    t_amb_k = ambient_temp_c + 273.15

    # Thermal expansion coefficient for ideal gas: β = 1/T_film
    beta = 1.0 / t_film_k

    # Temperature difference
    delta_t = t_flame_k - t_amb_k

    # Kinematic viscosity of air at film temperature (approximate)
    # ν ≈ 1.5e-5 * (T/300)^1.75 m²/s
    nu = 1.5e-5 * (t_film_k / 300.0) ** 1.75

    # Pressure correction for viscosity (ν ∝ 1/P)
    nu *= 101.3 / pressure_kpa

    grashof = gravity_m_s2 * beta * delta_t * characteristic_length_m**3 / nu**2

    # Classify regime
    if grashof < 0.1:
        regime = "diffusion_dominated"
    elif grashof < 100:
        regime = "transitional"
    else:
        regime = "buoyancy_dominated"

    return GrashofAnalysis(
        grashof_number=grashof,
        regime=regime,
        characteristic_length_m=characteristic_length_m,
        temperature_diff_k=delta_t,
        kinematic_viscosity_m2_s=nu,
        thermal_expansion_coeff=beta,
    )


def detect_cool_flame_transition(
    timestamps: NDArray[np.float64],
    diameters: NDArray[np.float64],
    blue_intensities: NDArray[np.float64] | None = None,
    min_secondary_duration_s: float = 2.0,
) -> tuple[bool, float | None, float | None]:
    """
    Detect transition to cool flame regime (second-stage combustion).

    Cool flame signatures:
    1. Diameter decay rate suddenly decreases (K drops by ~80-90%)
    2. Blue intensity drops significantly (lower temperature)
    3. Long duration at small diameter (> 2s)

    Returns:
        (detected, transition_time_s, secondary_k)
    """
    if len(diameters) < 10:
        return False, None, None

    d2 = diameters ** 2
    t = timestamps

    # Compute local burning rate using sliding window
    window = min(10, len(t) // 3)
    local_k = np.zeros(len(t) - window)

    for i in range(len(local_k)):
        t_win = t[i:i+window]
        d2_win = d2[i:i+window]
        if len(np.unique(t_win)) > 2:
            slope, _, _, _, _ = stats.linregress(t_win, d2_win)
            local_k[i] = -slope
        else:
            local_k[i] = 0

    # Find significant drop in K (transition)
    if len(local_k) < 5:
        return False, None, None

    k_median = np.median(local_k[:len(local_k)//2])
    k_drop_threshold = k_median * 0.2  # 80% drop

    transition_idx = None
    for i in range(window, len(local_k)):
        if local_k[i] < k_drop_threshold and local_k[i-1] > k_drop_threshold:
            transition_idx = i
            break

    if transition_idx is None:
        return False, None, None

    transition_time = t[transition_idx + window]

    # Verify sustained low K for minimum duration
    post_k = local_k[transition_idx:]
    post_t = t[transition_idx + window:]
    mask = post_t - transition_time < min_secondary_duration_s
    if np.any(mask) and np.median(post_k[mask]) < k_drop_threshold:
        secondary_k = np.median(post_k[mask])
        return True, float(transition_time), float(secondary_k)

    return False, None, None


def predict_extinction_time(
    d2_result: D2LawResult,
    current_time: float,
    current_diameter_mm: float,
) -> float | None:
    """
    Predict time to extinction based on current d²-law fit.

    Extinction when D²(t) = 0 => t_ext = D₀² / K
    """
    if not d2_result.is_valid or d2_result.burning_rate_k <= 0:
        return None

    # Current D²
    d2_current = current_diameter_mm ** 2
    # Remaining time = D²_current / K
    remaining = d2_current / d2_result.burning_rate_k
    return current_time + remaining


def compute_minimum_oxygen_index(
    fuel_type: str,
    pressure_kpa: float = 101.3,
    gravity: str = "micro",
) -> float:
    """
    Estimate Minimum Oxygen Index (MOI) for flame extinction.

    Based on NASA experimental correlations for various fuels in microgravity.
    Returns O₂ percentage below which flame cannot sustain.
    """
    # Base MOI values at 101.3 kPa, microgravity
    base_moi = {
        "heptane": 13.5,
        "methanol": 11.0,
        "hexane": 13.0,
        "pmma": 14.5,
        "ethylene": 12.0,
    }

    moi = base_moi.get(fuel_type.lower(), 14.0)

    # Pressure correction: MOI increases slightly with pressure
    moi *= (pressure_kpa / 101.3) ** 0.1

    # Gravity correction: higher gravity -> lower MOI (buoyancy helps)
    gravity_factors = {"micro": 1.0, "lunar": 0.95, "mars": 0.9, "earth": 0.85}
    moi *= gravity_factors.get(gravity, 1.0)

    return round(moi, 1)


def analyze_flame_metrics(metrics_list: list[FlameMetrics]) -> dict:
    """
    Full physics analysis pipeline on a list of FlameMetrics.

    Returns comprehensive results dictionary.
    """
    # Filter detected frames
    detected = [m for m in metrics_list if m.flame_detected]
    if len(detected) < 5:
        return {"error": "Insufficient detected frames for analysis"}

    timestamps = np.array([m.timestamp_s for m in detected])
    diameters = np.array([m.equivalent_diameter_mm for m in detected])
    blue_intensities = np.array([m.mean_blue_intensity for m in detected])

    # d²-law analysis
    d2_result = compute_d2_burn_rate(timestamps, diameters)

    # Extinction detection
    extinction = detect_extinction_event(diameters, timestamps)

    # Cool flame detection
    cool_detected, cool_time, cool_k = detect_cool_flame_transition(
        timestamps, diameters, blue_intensities
    )

    # Grashof analysis (using average flame temp from thermocouple if available)
    # Default to typical microgravity flame temp
    grashof = compute_grashof_number(
        gravity_m_s2=1e-6,  # Microgravity
        flame_temp_c=900,
        ambient_temp_c=25,
        characteristic_length_m=np.mean(diameters) / 1000,  # mm to m
    )

    # Extinction prediction
    pred_extinction = None
    if d2_result.is_valid and len(diameters) > 0:
        pred_extinction = predict_extinction_time(
            d2_result, timestamps[-1], diameters[-1]
        )

    return {
        "d2_law": {
            "burning_rate_k_mm2_s": d2_result.burning_rate_k,
            "burning_rate_stderr": d2_result.burning_rate_k_stderr,
            "r_squared": d2_result.r_squared,
            "d0_squared_mm2": d2_result.d0_squared_mm2,
            "quality": d2_result.quality_flag,
            "steady_regime_frames": (d2_result.steady_start_idx, d2_result.steady_end_idx),
        },
        "extinction": {
            "detected": extinction is not None,
            "time_s": extinction.extinction_time_s if extinction else None,
            "mechanism": extinction.mechanism if extinction else None,
            "confidence": extinction.confidence if extinction else None,
            "predicted_time_s": pred_extinction,
        },
        "cool_flame": {
            "detected": cool_detected,
            "transition_time_s": cool_time,
            "secondary_burn_rate_k": cool_k,
        },
        "grashof": {
            "number": grashof.grashof_number,
            "regime": grashof.regime,
        },
        "summary": {
            "total_frames": len(metrics_list),
            "detected_frames": len(detected),
            "initial_diameter_mm": diameters[0] if len(diameters) > 0 else 0,
            "final_diameter_mm": diameters[-1] if len(diameters) > 0 else 0,
            "burn_duration_s": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
        },
    }


if __name__ == "__main__":
    # Quick test with synthetic d²-law data
    print("Testing physics_engine.py...")

    # Simulate d²-law: D² = 64 - 0.85*t, with noise
    np.random.seed(42)
    t = np.linspace(0, 30, 100)
    k_true = 0.85
    d0_sq = 64.0
    d2_true = d0_sq - k_true * t
    d2_noisy = d2_true + np.random.normal(0, 1.5, len(t))
    d_noisy = np.sqrt(np.maximum(d2_noisy, 0))

    result = compute_d2_burn_rate(t, d_noisy)
    print(f"True K: {k_true:.3f}, Fitted K: {result.burning_rate_k:.3f} ± {result.burning_rate_k_stderr:.3f}")
    print(f"R²: {result.r_squared:.4f}, Quality: {result.quality_flag}")
    print(f"True D₀²: {d0_sq:.1f}, Fitted D₀²: {result.d0_squared_mm2:.1f}")

    # Test extinction detection
    ext = detect_extinction_event(d_noisy, t)
    if ext:
        print(f"\nExtinction detected at t={ext.extinction_time_s:.2f}s, mechanism={ext.mechanism}")

    # Test Grashof
    gr = compute_grashof_number(9.81, 1600, 25, 0.01)
    print(f"\nEarth (1g) Grashof: {gr.grashof_number:.2e}, Regime: {gr.regime}")

    gr_micro = compute_grashof_number(1e-6, 900, 25, 0.01)
    print(f"Microgravity Grashof: {gr_micro.grashof_number:.2e}, Regime: {gr_micro.regime}")

    print("\nTest complete.")