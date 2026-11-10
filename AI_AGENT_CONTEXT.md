# AI Agent Context & Development Guide: MicroG-FlameVision

> **Purpose:** This file provides complete technical context, domain knowledge, coding invariants, data schemas, and implementation checklists for AI coding agents (Antigravity, Claude Code, Cursor, Copilot, etc.) working on the **MicroG-FlameVision** repository.

---

## 1. System Persona & Agent Role

When generating or editing code in this repository, you act as:
* **Role:** Lead Aerospace Combustion Scientist & Senior Full-Stack AI/CV Engineer.
* **Domain Focus:** Spacecraft Fire Safety, Fluid Dynamics (Microgravity vs. Terrestrial Convection), Computer Vision (OpenCV), and Scientific Web Dashboards (Streamlit/HTML5 Canvas).
* **Communication Style:** Precise, modular, mathematically rigorous, and clean.

---

## 2. Fundamental Domain Invariants (Do Not Violate)

1. **No Buoyancy in Microgravity:**
   * When $g = 0$, the buoyancy term $\rho \beta (T - T_\infty) \vec{g}$ **must evaluate to 0**.
   * Do NOT apply any upward drift velocity to microgravity flame particles or models unless forced ventilation ($v_{air}$) is explicitly non-zero.
2. **Soot & Color Differences:**
   * Earth ($1g$): Combustion produces soot in the hot plume $\to$ bright yellow/orange incandescent radiation ($580\text{–}650\text{ nm}$).
   * Space ($0g$): Minimal soot, diffusion-dominated $\to$ faint blue chemiluminescence ($\text{CH}^*$ and $\text{C}_2^*$ radicals, $430\text{ nm}$ and $516\text{ nm}$) or near-infrared cool flames.
   * **CV Consequence:** Never rely on standard grayscale luminance ($0.299R + 0.587G + 0.114B$) to detect microgravity flames. You must isolate and enhance the **Blue channel** ($B$) or use adaptive CLAHE in the LAB/HSV color space.
3. **The $d^2$-Law Consistency:**
   * Droplet diameter squared follows $D^2(t) = D_0^2 - K \cdot t$.
   * Units: Diameter $D$ in millimeters ($\text{mm}$), time $t$ in seconds ($\text{s}$), burning rate constant $K$ in $\text{mm}^2/\text{s}$.
   * When fitting $K$, exclude the initial ignition transient (first $10\text{–}15\%$ of time) and final extinction phase (last $5\%$).
4. **Unit Standards:**
   * Pressure: kilopascals ($\text{kPa}$) — standard ISS cabin is $\approx 101.3\text{ kPa}$.
   * Oxygen fraction: percentage ($14\%\text{–}34\%$) — standard ISS atmosphere is $\approx 21\%$.
   * Airflow: meters per second ($\text{m/s}$) — ISS ventilation loops typically $0.05\text{–}0.2\text{ m/s}$.

---

## 3. Recommended Repository Structure

```
microg-flamevision/
├── PROJECT_SPECIFICATION.md     # Full architectural and scientific specification
├── AI_AGENT_CONTEXT.md          # This file: Agent context, schemas, and instructions
├── README.md                    # Public GitHub presentation and hackathon summary
├── requirements.txt             # Python dependencies
├── app.py                       # Streamlit multi-tab main application
├── modules/
│   ├── __init__.py
│   ├── flame_cv.py              # Video enhancement, contour detection, diameter extraction
│   ├── physics_engine.py        # d²-law regression, Grashof analysis, extinction detector
│   ├── ml_model.py              # Spacecraft fire hazard predictor & synthetic trainer
│   └── mock_data_generator.py   # Generates realistic synthetic CIR video & telemetry for testing
├── static/
│   ├── playground.html          # Standalone HTML5 Canvas interactive 0g/1g physics simulator
│   └── style.css                # Custom CSS for Streamlit and dashboard
├── data/
│   ├── samples/                 # Sample video clips and telemetry CSV files
│   └── synthetic/               # Test fixtures and generated telemetry
└── tests/
    ├── test_cv.py               # Unit tests for flame extraction
    ├── test_physics.py          # Unit tests for d²-law calculation
    └── test_model.py            # Unit tests for ML predictor
```

---

## 4. Data Contracts & Schemas

### 4.1 Input Telemetry CSV Schema
NASA PSI flight experiment records must be parsed or mocked according to this schema:

```csv
timestamp_s,chamber_pressure_kpa,o2_percent,cabin_flow_mps,fuel_type,radiometer_w_cm2,thermocouple_temp_c
0.0,101.3,21.0,0.05,Heptane,0.012,24.5
0.1,101.3,20.98,0.05,Heptane,0.045,180.2
0.2,101.2,20.95,0.05,Heptane,0.089,450.0
...
```

### 4.2 Frame Processing Result (Python Dictionary)
Returned for every processed video frame by `flame_cv.py`:

```json
{
  "frame_index": 142,
  "timestamp_s": 4.73,
  "flame_detected": true,
  "centroid": [320.5, 240.1],
  "major_axis_mm": 8.42,
  "minor_axis_mm": 8.35,
  "equivalent_diameter_mm": 8.38,
  "diameter_squared_mm2": 70.22,
  "aspect_ratio": 1.008,
  "mean_blue_intensity": 142.3,
  "is_cool_flame_candidate": false
}
```

### 4.3 Hazard Assessment Output (from `ml_model.py`)

```json
{
  "predicted_time_to_extinction_s": 14.8,
  "burning_rate_k_mm2_s": 0.85,
  "hazard_level": "MODERATE",
  "cool_flame_risk_pct": 28.5,
  "minimum_quench_oxygen_pct": 14.2,
  "ventilation_recommendation": "Maintain cabin ventilation below 0.04 m/s to starve diffusion envelope."
}
```

---

## 5. Module Implementation Specifications

### 5.1 `modules/flame_cv.py`
* **Dependencies:** `opencv-python-headless`, `numpy`, `scipy`.
* **Key Functions to Implement:**
  1. `enhance_faint_flame(frame: np.ndarray) -> np.ndarray`:
     * Convert BGR frame to LAB or split into BGR channels.
     * Apply CLAHE (`cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))`) to the Blue and Luminance channels.
  2. `extract_flame_metrics(frame: np.ndarray, scale_px_per_mm: float = 20.0) -> dict`:
     * Threshold enhanced frame using Otsu or adaptive Gaussian thresholding.
     * Find contours using `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
     * If valid contour found ($Area > \text{min\_area}$):
       * Fit ellipse using `cv2.fitEllipse(contour)`.
       * Compute $D_{eff} = 2 \cdot \sqrt{\text{Area} / \pi}$.
       * Return metrics dictionary.
     * Else, return `flame_detected: False`.

### 5.2 `modules/physics_engine.py`
* **Key Functions to Implement:**
  1. `compute_d2_burn_rate(timestamps: list[float], diameters: list[float]) -> tuple[float, float, list[float]]`:
     * Compute $D^2(t)$.
     * Smooth $D^2(t)$ using `scipy.signal.savgol_filter`.
     * Fit linear regression $D^2(t) = D_0^2 - K \cdot t$ over the steady burning regime ($0.15 \le t/t_{max} \le 0.90$).
     * Return $(K, R^2, smoothed\_d2)$.
  2. `detect_extinction_event(diameters: list[float], timestamps: list[float], threshold_mm: float = 0.5) -> float | None`:
     * Identify the timestamp where flame diameter permanently falls below threshold.

### 5.3 `modules/ml_model.py`
* **Model Type:** Scikit-learn `RandomForestRegressor` and `RandomForestClassifier`.
* **Inputs:** `[pressure_kpa, o2_percent, flow_velocity_mps, initial_diameter_mm, fuel_code]`.
* **Targets:**
  * Regressor: `burn_time_s`
  * Classifier: `cool_flame_risk` ($0$ or $1$)
* Provide a fallback trained model or synthetic training loop so the app works out-of-the-box without requiring gigabytes of downloaded datasets.

### 5.4 `static/playground.html`
* A self-contained, dependency-free HTML5/Canvas interactive simulator.
* **Physics Engine Rules:**
  * Grid or particle array simulating hot gas and fuel vapor.
  * Particle velocity: $\vec{v} = \vec{v}_{diffusion} + \vec{v}_{buoyancy} + \vec{v}_{forced}$.
  * Buoyancy velocity: $v_{buoyant\_y} \propto -g \cdot (T_{particle} - T_{ambient})$.
  * When $g = 0$, $v_{buoyant\_y} = 0$.
  * Oxygen depletion field: particles consume nearby virtual $O_2$. When local $O_2 < 12\%$, particles cease combustion and turn to smoke.
  * UI Controls: Gravity slider ($0g\text{--}1g$), Oxygen slider ($10\%\text{--}35\%$), Airflow slider ($0\text{--}1\text{ m/s}$), and preset buttons.

---

## 6. Step-by-Step Implementation Task Checklist

When prompted to build or expand the project, follow this order:

- [ ] **Step 1:** Create `requirements.txt` with:
  ```text
  streamlit>=1.30.0
  opencv-python-headless>=4.8.0
  numpy>=1.24.0
  scipy>=1.11.0
  pandas>=2.0.0
  plotly>=5.18.0
  scikit-learn>=1.3.0
  ```
- [ ] **Step 2:** Implement `modules/mock_data_generator.py` so the team can test video extraction and telemetry plots immediately without needing raw NASA archives locally.
- [ ] **Step 3:** Implement `modules/flame_cv.py` with frame enhancement and ellipse fitting.
- [ ] **Step 4:** Implement `modules/physics_engine.py` with $d^2$-law regression and extinction detection.
- [ ] **Step 5:** Implement `modules/ml_model.py` with the pre-trained/synthetic hazard predictor.
- [ ] **Step 6:** Implement `static/playground.html` with real-time 0g vs 1g canvas particle physics.
- [ ] **Step 7:** Implement `app.py` in Streamlit with modern dark UI styling, three tabs (Analyzer, Playground, Safety Handbook), and Plotly interactive charts.
- [ ] **Step 8:** Add unit tests in `tests/` verifying CV and physics math.

---

## 7. Quality & Code Conventions
* **Docstrings:** Use Google-style or NumPy-style docstrings with explicit parameter types.
* **Typing:** Use Python 3.10+ type hints (`float | None`, `list[dict]`, etc.).
* **Error Handling:** Gracefully handle missing video codecs or corrupt frames without crashing the Streamlit session.
* **Performance:** Never process high-res video at 60fps frame-by-frame in Streamlit loops; sample every $N$-th frame (e.g. 5–10 fps) for real-time responsiveness.
