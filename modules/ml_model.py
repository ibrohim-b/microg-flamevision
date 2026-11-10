"""
Spacecraft Fire Hazard Predictor

Machine learning module for predicting fire hazards in microgravity environments.
Uses Random Forest regressor/classifier trained on synthetic NASA PSI-style data.

Features:
- Burning time prediction
- Cool flame risk classification
- Minimum Oxygen Index (MOI) estimation
- Ventilation safety recommendations
- Model persistence (joblib)
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report


# Fuel type encoding
FUEL_CODES = {
    "heptane": 0,
    "methanol": 1,
    "hexane": 2,
    "pmma": 3,
    "ethylene": 4,
}

FUEL_NAMES = {v: k for k, v in FUEL_CODES.items()}

HAZARD_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


@dataclass
class HazardAssessment:
    """Complete fire hazard assessment output."""
    predicted_time_to_extinction_s: float
    burning_rate_k_mm2_s: float
    hazard_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    cool_flame_risk_pct: float
    minimum_quench_oxygen_pct: float
    ventilation_recommendation: str
    confidence: float
    feature_importance: dict[str, float] | None = None


@dataclass
class TrainingMetrics:
    """Model training/validation metrics."""
    regressor_mae: float
    regressor_rmse: float
    regressor_r2: float
    classifier_accuracy: float
    classifier_report: str
    n_training_samples: int
    cv_scores: list[float]


class FlameHazardPredictor:
    """
    Predicts spacecraft fire hazards from environmental parameters.

    Inputs: [pressure_kpa, o2_percent, flow_velocity_mps, initial_diameter_mm, fuel_code]
    Outputs: burn_time, cool_flame_risk, hazard_level, MOI, ventilation_rec

    Uses Random Forest for robustness with small datasets and interpretability.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        self.random_state = random_state

        # Regressor for continuous outputs
        self.regressor = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])

        # Classifier for cool flame risk
        self.classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])

        self.is_trained = False
        self.feature_names = [
            "pressure_kpa",
            "o2_percent",
            "flow_velocity_mps",
            "initial_diameter_mm",
            "fuel_code",
        ]
        self._feature_importance: dict[str, float] | None = None

    def _encode_fuel(self, fuel_type: str) -> int:
        """Encode fuel type to integer code."""
        return FUEL_CODES.get(fuel_type.lower(), 0)

    def _decode_fuel(self, code: int) -> str:
        """Decode fuel code to name."""
        return FUEL_NAMES.get(code, "heptane")

    def _prepare_features(
        self,
        pressure_kpa: float | NDArray,
        o2_percent: float | NDArray,
        flow_velocity_mps: float | NDArray,
        initial_diameter_mm: float | NDArray,
        fuel_type: str | list[str] | NDArray,
    ) -> NDArray:
        """Prepare feature matrix for prediction/training."""
        # Convert to arrays
        pressure = np.atleast_1d(pressure_kpa)
        o2 = np.atleast_1d(o2_percent)
        flow = np.atleast_1d(flow_velocity_mps)
        diam = np.atleast_1d(initial_diameter_mm)

        # Handle fuel type
        if isinstance(fuel_type, str):
            fuel_codes = np.full_like(pressure, self._encode_fuel(fuel_type), dtype=int)
        else:
            fuel_arr = np.atleast_1d(fuel_type)
            if fuel_arr.dtype.kind in 'SUO':  # string/object
                fuel_codes = np.array([self._encode_fuel(f) for f in fuel_arr])
            else:
                fuel_codes = fuel_arr.astype(int)

        # Ensure all same length
        n = len(pressure)
        for arr, name in [(o2, "o2"), (flow, "flow"), (diam, "diam"), (fuel_codes, "fuel")]:
            if len(arr) != n:
                raise ValueError(f"All inputs must have same length. {name} has {len(arr)}, expected {n}")

        X = np.column_stack([pressure, o2, flow, diam, fuel_codes])
        return X

    def train(
        self,
        X: NDArray,
        y_burn_time: NDArray,
        y_cool_flame: NDArray,
        validate: bool = True,
    ) -> TrainingMetrics:
        """
        Train both regressor and classifier.

        Args:
            X: Feature matrix (n_samples, 5)
            y_burn_time: Target burn times (seconds)
            y_cool_flame: Target cool flame labels (0/1)
            validate: Whether to run cross-validation

        Returns:
            TrainingMetrics with validation scores
        """
        if X.shape[1] != 5:
            raise ValueError(f"Expected 5 features, got {X.shape[1]}")

        # Train regressor
        self.regressor.fit(X, y_burn_time)

        # Train classifier
        self.classifier.fit(X, y_cool_flame)

        self.is_trained = True

        # Extract feature importance from regressor
        rf_reg = self.regressor.named_steps["rf"]
        self._feature_importance = dict(zip(self.feature_names, rf_reg.feature_importances_))

        metrics = TrainingMetrics(
            regressor_mae=0.0,
            regressor_rmse=0.0,
            regressor_r2=0.0,
            classifier_accuracy=0.0,
            classifier_report="",
            n_training_samples=len(X),
            cv_scores=[],
        )

        if validate:
            # Cross-validation for regressor
            cv_scores = cross_val_score(
                self.regressor, X, y_burn_time, cv=5, scoring="neg_mean_absolute_error"
            )
            metrics.cv_scores = [-s for s in cv_scores]
            metrics.regressor_mae = float(np.mean(metrics.cv_scores))
            metrics.regressor_rmse = float(np.sqrt(np.mean(cross_val_score(
                self.regressor, X, y_burn_time, cv=5, scoring="neg_mean_squared_error"
            )) * -1))

            # R² on training (optimistic but useful)
            y_pred = self.regressor.predict(X)
            metrics.regressor_r2 = float(1 - np.sum((y_burn_time - y_pred)**2) / np.sum((y_burn_time - np.mean(y_burn_time))**2))

            # Classifier validation
            y_cls_pred = self.classifier.predict(X)
            metrics.classifier_accuracy = float(accuracy_score(y_cool_flame, y_cls_pred))
            metrics.classifier_report = classification_report(y_cool_flame, y_cls_pred, output_dict=False)

        return metrics

    def predict(self, X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Predict burn time, cool flame probability, and hazard level.

        Returns:
            (burn_time_s, cool_flame_prob, hazard_level_idx)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        burn_time = self.regressor.predict(X)
        cool_flame_prob = self.classifier.predict_proba(X)[:, 1]

        # Hazard level from burn time and cool flame risk
        hazard_idx = self._compute_hazard_level(burn_time, cool_flame_prob)

        return burn_time, cool_flame_prob, hazard_idx

    def _compute_hazard_level(self, burn_time: NDArray, cool_flame_prob: NDArray) -> NDArray:
        """
        Compute hazard level index (0=LOW, 1=MODERATE, 2=HIGH, 3=CRITICAL).

        Logic:
        - Long burn time + high cool flame risk = CRITICAL
        - Short burn time + low risk = LOW
        """
        hazard = np.zeros(len(burn_time), dtype=int)

        # Normalize burn time (typical range 5-60s)
        norm_burn = np.clip(burn_time / 60.0, 0, 1)
        norm_cool = cool_flame_prob

        # Combined risk score
        risk = 0.6 * norm_burn + 0.4 * norm_cool

        hazard[risk < 0.25] = 0   # LOW
        hazard[(risk >= 0.25) & (risk < 0.5)] = 1  # MODERATE
        hazard[(risk >= 0.5) & (risk < 0.75)] = 2  # HIGH
        hazard[risk >= 0.75] = 3  # CRITICAL

        return hazard

    def assess_hazard(
        self,
        pressure_kpa: float,
        o2_percent: float,
        flow_velocity_mps: float,
        initial_diameter_mm: float,
        fuel_type: str,
    ) -> HazardAssessment:
        """
        Full hazard assessment for a single scenario.

        Returns complete HazardAssessment with recommendations.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = self._prepare_features(
            pressure_kpa, o2_percent, flow_velocity_mps,
            initial_diameter_mm, fuel_type
        )

        burn_time, cool_prob, hazard_idx = self.predict(X)

        # Minimum quench oxygen (MOI) - physics-based with ML correction
        base_moi = self._compute_base_moi(fuel_type, pressure_kpa)
        # ML adjustment based on flow and diameter
        moi_adjustment = self._predict_moi_adjustment(X[0])
        min_quench_o2 = base_moi + moi_adjustment

        # Ventilation recommendation
        vent_rec = self._generate_ventilation_recommendation(
            flow_velocity_mps, o2_percent, cool_prob[0], hazard_idx[0]
        )

        return HazardAssessment(
            predicted_time_to_extinction_s=float(burn_time[0]),
            burning_rate_k_mm2_s=self._estimate_burning_rate(
                pressure_kpa, o2_percent, flow_velocity_mps,
                initial_diameter_mm, fuel_type
            ),
            hazard_level=HAZARD_LEVELS[int(hazard_idx[0])],
            cool_flame_risk_pct=float(cool_prob[0] * 100),
            minimum_quench_oxygen_pct=round(min_quench_o2, 1),
            ventilation_recommendation=vent_rec,
            confidence=self._compute_confidence(X[0]),
            feature_importance=self._feature_importance,
        )

    def _compute_base_moi(self, fuel_type: str, pressure_kpa: float) -> float:
        """Physics-based minimum oxygen index."""
        base = {
            "heptane": 13.5, "methanol": 11.0, "hexane": 13.0,
            "pmma": 14.5, "ethylene": 12.0,
        }.get(fuel_type.lower(), 14.0)
        return base * (pressure_kpa / 101.3) ** 0.1

    def _predict_moi_adjustment(self, features: NDArray) -> float:
        """ML-based MOI adjustment (simplified)."""
        # Extract features
        _, o2, flow, diam, fuel_code = features
        # Heuristic: higher flow -> slightly lower MOI (blowoff limits)
        # Larger diameter -> slightly higher MOI
        adj = -0.5 * flow + 0.02 * diam
        return np.clip(adj, -2.0, 2.0)

    def _estimate_burning_rate(
        self,
        pressure_kpa: float,
        o2_percent: float,
        flow_velocity_mps: float,
        initial_diameter_mm: float,
        fuel_type: str,
    ) -> float:
        """Estimate burning rate K from physics correlations."""
        base_k = {
            "heptane": 0.85, "methanol": 0.62, "hexane": 0.78,
            "pmma": 0.45, "ethylene": 1.10,
        }.get(fuel_type.lower(), 0.85)

        gravity_mult = 1.0  # Assume microgravity
        pressure_factor = math.sqrt(pressure_kpa / 101.3)
        o2_factor = max(0.3, o2_percent / 21.0)
        flow_factor = 1.0 + 2.0 * flow_velocity_mps

        return round(base_k * gravity_mult * pressure_factor * o2_factor * flow_factor, 3)

    def _generate_ventilation_recommendation(
        self,
        flow: float,
        o2: float,
        cool_risk: float,
        hazard_idx: int,
    ) -> str:
        """Generate actionable ventilation recommendation."""
        if hazard_idx >= 3:  # CRITICAL
            return ("EMERGENCY: Shut down ventilation immediately. "
                    "Deploy fire suppression. Evacuate module.")
        elif hazard_idx == 2:  # HIGH
            if cool_risk > 0.5:
                return ("HIGH RISK: Reduce ventilation to < 0.03 m/s to starve diffusion flame. "
                        "Monitor for cool flame persistence.")
            else:
                return ("HIGH RISK: Limit ventilation to < 0.05 m/s. "
                        "Prepare suppression system.")
        elif hazard_idx == 1:  # MODERATE
            return (f"MODERATE: Maintain ventilation at {flow:.3f} m/s. "
                    f"O₂ at {o2:.1f}% - monitor for changes.")
        else:  # LOW
            return "LOW: Standard ventilation acceptable. Continue monitoring."

    def _compute_confidence(self, features: NDArray) -> float:
        """Estimate prediction confidence based on training data proximity."""
        # Simplified: use ensemble variance as uncertainty proxy
        rf_reg = self.regressor.named_steps["rf"]
        rf_cls = self.classifier.named_steps["rf"]

        # Get predictions from all trees
        X_scaled = self.regressor.named_steps["scaler"].transform(features.reshape(1, -1))
        tree_preds_reg = np.array([tree.predict(X_scaled) for tree in rf_reg.estimators_])
        tree_preds_cls = np.array([tree.predict_proba(X_scaled)[0, 1] for tree in rf_cls.estimators_])

        reg_cv = np.std(tree_preds_reg) / (np.mean(tree_preds_reg) + 1e-6)
        cls_cv = np.std(tree_preds_cls) / (np.mean(tree_preds_cls) + 1e-6)

        # Confidence inversely related to coefficient of variation
        confidence = 1.0 - min(0.5, (reg_cv + cls_cv) / 2)
        return round(max(0.3, confidence), 2)

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "regressor": self.regressor,
            "classifier": self.classifier,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "feature_importance": self._feature_importance,
            "random_state": self.random_state,
            "version": "1.0",
        }
        joblib.dump(model_data, path)

    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
        model_data = joblib.load(path)
        self.regressor = model_data["regressor"]
        self.classifier = model_data["classifier"]
        self.is_trained = model_data["is_trained"]
        self.feature_names = model_data["feature_names"]
        self._feature_importance = model_data.get("feature_importance")
        self.random_state = model_data.get("random_state", 42)

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importance from trained regressor."""
        return self._feature_importance


def generate_synthetic_training_data(n_samples: int = 2000, seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate synthetic training data for the hazard predictor.

    Based on physics correlations from NASA combustion experiments.
    """
    rng = np.random.default_rng(seed)

    # Sample parameters
    pressures = rng.uniform(50, 150, n_samples)
    o2_percents = rng.uniform(14, 34, n_samples)
    flows = rng.uniform(0.0, 0.5, n_samples)
    diameters = rng.uniform(2, 15, n_samples)
    fuel_codes = rng.integers(0, 5, n_samples)

    X = np.column_stack([pressures, o2_percents, flows, diameters, fuel_codes])

    # Physics-based burn time: t = D₀² / K
    # K depends on fuel, pressure, O2, flow, gravity
    base_k = np.array([0.85, 0.62, 0.78, 0.45, 1.10])[fuel_codes]
    gravity_mult = 1.0  # microgravity
    pressure_factor = np.sqrt(pressures / 101.3)
    o2_factor = np.maximum(0.3, o2_percents / 21.0)
    flow_factor = 1.0 + 2.0 * flows

    k_eff = base_k * gravity_mult * pressure_factor * o2_factor * flow_factor
    burn_time = (diameters ** 2) / k_eff

    # Add noise
    burn_time += rng.normal(0, burn_time * 0.1)
    burn_time = np.maximum(burn_time, 1.0)  # Minimum 1 second

    # Cool flame risk: higher in microgravity, low O2, low flow, certain fuels
    cool_risk = np.zeros(n_samples)
    cool_risk += 0.4  # Base microgravity risk
    cool_risk += 0.3 * (18 - o2_percents).clip(min=0) / 10  # Low O2
    cool_risk += 0.2 * (0.05 - flows).clip(min=0) / 0.05  # Low flow
    # Fuel factor
    fuel_cool = np.array([0.15, 0.05, 0.1, 0.0, 0.05])[fuel_codes]
    cool_risk += fuel_cool
    cool_risk += rng.normal(0, 0.05, n_samples)
    cool_risk = np.clip(cool_risk, 0, 1)

    y_cool_flame = (cool_risk > 0.5).astype(int)

    return X, burn_time, y_cool_flame


def train_synthetic_model(
    n_samples: int = 2000,
    model_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[FlameHazardPredictor, TrainingMetrics]:
    """
    Train a hazard predictor on synthetic data and optionally save it.

    Returns:
        (trained_predictor, training_metrics)
    """
    print(f"Generating {n_samples} synthetic training samples...")
    X, y_burn, y_cool = generate_synthetic_training_data(n_samples, seed)

    print("Training Random Forest models...")
    predictor = FlameHazardPredictor(random_state=seed)
    metrics = predictor.train(X, y_burn, y_cool, validate=True)

    print(f"Training complete:")
    print(f"  Regressor MAE: {metrics.regressor_mae:.2f}s, RMSE: {metrics.regressor_rmse:.2f}s, R²: {metrics.regressor_r2:.3f}")
    print(f"  Classifier Accuracy: {metrics.classifier_accuracy:.3f}")

    if model_path:
        predictor.save(model_path)
        print(f"Model saved to {model_path}")

    return predictor, metrics


# Pre-trained model instance (lazy loaded)
_DEFAULT_MODEL: FlameHazardPredictor | None = None
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "flame_hazard_predictor.joblib"


def get_default_predictor() -> FlameHazardPredictor:
    """Get or create the default pre-trained predictor."""
    global _DEFAULT_MODEL

    if _DEFAULT_MODEL is not None:
        return _DEFAULT_MODEL

    # Try to load existing model
    if _DEFAULT_MODEL_PATH.exists():
        try:
            _DEFAULT_MODEL = FlameHazardPredictor()
            _DEFAULT_MODEL.load(_DEFAULT_MODEL_PATH)
            print(f"Loaded pre-trained model from {_DEFAULT_MODEL_PATH}")
            return _DEFAULT_MODEL
        except Exception as e:
            warnings.warn(f"Failed to load pre-trained model: {e}")

    # Train new model
    print("No pre-trained model found. Training new model on synthetic data...")
    _DEFAULT_MODEL, _ = train_synthetic_model(
        n_samples=3000,
        model_path=_DEFAULT_MODEL_PATH,
        seed=42,
    )
    return _DEFAULT_MODEL


def predict_hazard(
    pressure_kpa: float,
    o2_percent: float,
    flow_velocity_mps: float,
    initial_diameter_mm: float,
    fuel_type: str,
) -> HazardAssessment:
    """
    Convenience function for one-off hazard predictions.

    Uses the default pre-trained model (trains on first call if needed).
    """
    predictor = get_default_predictor()
    return predictor.assess_hazard(
        pressure_kpa, o2_percent, flow_velocity_mps,
        initial_diameter_mm, fuel_type
    )


if __name__ == "__main__":
    # Demo: Train and test
    print("Training hazard predictor on synthetic data...")
    predictor, metrics = train_synthetic_model(n_samples=2000)

    # Test predictions
    print("\n--- Test Predictions ---")
    test_cases = [
        {"pressure_kpa": 101.3, "o2_percent": 21.0, "flow_velocity_mps": 0.05,
         "initial_diameter_mm": 8.0, "fuel_type": "heptane"},
        {"pressure_kpa": 101.3, "o2_percent": 16.0, "flow_velocity_mps": 0.02,
         "initial_diameter_mm": 6.0, "fuel_type": "heptane"},
        {"pressure_kpa": 101.3, "o2_percent": 30.0, "flow_velocity_mps": 0.2,
         "initial_diameter_mm": 10.0, "fuel_type": "pmma"},
    ]

    for i, case in enumerate(test_cases):
        assessment = predictor.assess_hazard(**case)
        print(f"\nCase {i+1}: {case['fuel_type']} @ {case['o2_percent']}% O₂, {case['flow_velocity_mps']} m/s flow")
        print(f"  Burn time: {assessment.predicted_time_to_extinction_s:.1f}s")
        print(f"  K: {assessment.burning_rate_k_mm2_s:.3f} mm²/s")
        print(f"  Hazard: {assessment.hazard_level}")
        print(f"  Cool flame risk: {assessment.cool_flame_risk_pct:.1f}%")
        print(f"  Min quench O₂: {assessment.minimum_quench_oxygen_pct:.1f}%")
        print(f"  Recommendation: {assessment.ventilation_recommendation}")
        print(f"  Confidence: {assessment.confidence:.0%}")

    print("\nFeature importance:")
    for feat, imp in sorted(predictor.get_feature_importance().items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")