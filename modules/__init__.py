"""
MicroG-FlameVision Core Modules

Aerospace combustion analysis platform for microgravity flame detection,
physics metric derivation, and spacecraft fire hazard prediction.
"""

from .flame_cv import enhance_faint_flame, extract_flame_metrics, process_video_frames
from .physics_engine import compute_d2_burn_rate, detect_extinction_event, compute_grashof_number
from .ml_model import FlameHazardPredictor, train_synthetic_model
from .mock_data_generator import generate_synthetic_video, generate_telemetry_csv, MockFlameGenerator

__all__ = [
    "enhance_faint_flame",
    "extract_flame_metrics",
    "process_video_frames",
    "compute_d2_burn_rate",
    "detect_extinction_event",
    "compute_grashof_number",
    "FlameHazardPredictor",
    "train_synthetic_model",
    "generate_synthetic_video",
    "generate_telemetry_csv",
    "MockFlameGenerator",
]

__version__ = "0.1.0"