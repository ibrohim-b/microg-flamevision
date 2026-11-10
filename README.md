# MicroG-FlameVision 🔥
### AI-Powered Microgravity Combustion Computer Vision & Spacecraft Fire Safety Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![NASA Space Apps](https://img.shields.io/badge/NASA%20Space%20Apps-2026-0B3D91.svg)](https://www.spaceappschallenge.org/)

**NASA International Space Apps Challenge 2026**  
**Challenge Track:** *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*  
**Theme:** *The Next Frontier*

---

## 1. Executive Summary & Mission Statement

Fire is among the most catastrophic risks in human spaceflight. Inside a hermetically sealed spacecraft (e.g., ISS, Orion, Lunar Gateway, or Mars Transit Habitat), toxic combustion products cannot be vented outside, evacuation routes are non-existent, and extinguishing agents are finite.

Decades of NASA experiments—from drop towers at Glenn Research Center to the **Combustion Integrated Rack (CIR)** on the ISS—have revealed that **fire behaves fundamentally differently in microgravity**:
1. **Vanishing Buoyant Convection:** When gravitational acceleration $g \approx 0$, hot gases do not rise. Flames form calm, symmetric spheres governed strictly by molecular diffusion rather than turbulent buoyant plumes.
2. **Invisible & "Cool Flame" Regimes:** Microgravity flames burn at lower temperatures ($800\text{–}1200^\circ\text{C}$ vs $1400\text{–}1800^\circ\text{C}$ on Earth), emitting almost zero soot incandescence. Instead of bright yellow flickers, they emit faint blue chemiluminescence ($\text{CH}^*$ at $430\text{ nm}$ and $\text{C}_2^*$ at $516\text{ nm}$) or enter low-temperature cool flame states that are completely invisible to standard optical smoke detectors.
3. **The Forced Ventilation Trap:** Without natural convection, quiescent space flames self-extinguish as they suffocate in their own stagnant $CO_2/H_2O$ blanket. However, forced cabin ventilation (air conditioning) acts as an artificial gravity, supplying fresh oxygen, elongating the flame front, and triggering sudden flare-ups.

**MicroG-FlameVision** is an end-to-end open-source platform combining an **AI Computer Vision & Physics Analytics Pipeline** with an **Interactive Combustion Physics Simulator** and **Astronaut Mission Control Dashboard** to detect invisible flames, extract fundamental droplet burning metrics ($d^2$-law), and predict spacecraft fire hazard thresholds.

---

## 2. Scientific & Physics Foundation

### Convection vs. Pure Diffusion

| Physical Characteristic | Terrestrial Gravity ($1g$) | Microgravity ($\approx 0g$) |
| :--- | :--- | :--- |
| **Dominant Transport** | Natural buoyant convection ($\rho \beta \Delta T \vec{g}$) | Pure molecular diffusion ($\nabla^2 C$) |
| **Flame Geometry** | Elongated teardrop, flickering buoyant tip | Symmetrical sphere, stationary |
| **Luminescence & Color** | Bright yellow/orange soot incandescence ($580\text{–}650\text{ nm}$) | Faint blue chemiluminescence ($\text{CH}^*, \text{C}_2^*$) / near-IR |
| **Soot Formation** | High (pyrolysis in hot buoyant plume) | Minimal to zero (clean diffusion envelope) |
| **Extinction Mechanism** | Fuel depletion or thermal blowoff | Suffocation by self-emitted $CO_2/H_2O$ blanket |
| **Temperature** | Hot core ($1400\text{–}1800^\circ\text{C}$) | Cooler, diffused envelope ($800\text{–}1200^\circ\text{C}$) |

### Mathematical Formulations

#### 1. The $d^2$-Law of Droplet Combustion
Classical spherical droplet combustion follows the $d^2$-law:
$$D^2(t) = D_0^2 - K \cdot t$$
Where:
- $D(t)$ = Equivalent flame/droplet diameter at time $t$ ($\text{mm}$)
- $D_0$ = Initial diameter ($\text{mm}$)
- $K = -\frac{d(D^2)}{dt}$ = Burning rate constant ($\text{mm}^2/\text{s}$)

In `modules/physics_engine.py`, linear regression is performed over the steady burning regime ($0.15 \le t/t_{max} \le 0.90$) using Savitzky-Golay smoothed signals to filter droplet vibration while isolating true burn kinetics.

#### 2. Buoyancy Convection vs. Diffusion (Grashof Number)
The Grashof number ($Gr$) quantifies the ratio of buoyant to viscous forces:
$$Gr = \frac{g \cdot \beta \cdot (T_{flame} - T_\infty) \cdot L^3}{\nu^2}$$
- On Earth ($g = 9.81\text{ m/s}^2$): $Gr \gg 100$ (buoyancy dominates transport).
- In Microgravity ($g \approx 10^{-6}\text{ m/s}^2$): $Gr \ll 0.1$ (pure molecular diffusion governs heat and mass transfer).

---

## 3. Architecture & Core Modules

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MicroG-FlameVision Core                         │
└────────────────────────────────────────────────────────────────────────┘
                                     │
      ┌──────────────────────────────┴──────────────────────────────┐
      ▼                                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│   ANALYTICAL ENGINE (AI/CV)  │              │    INTERACTIVE PLAYGROUND    │
├──────────────────────────────┤              ├──────────────────────────────┤
│ 1. Blue-Channel CLAHE Engine │              │ 1. 2D Particle Fluid Model   │
│ 2. Adaptive Contour Detector │              │ 2. Gravity Slider (0g to 1g) │
│ 3. Ellipse & Geometry Track  │              │ 3. Oxygen & Airflow Modulator│
│ 4. d²-Law & Physics Deriver  │              │ 4. Real-time Extinction Check│
│ 5. Random Forest ML Model    │              │ 5. Guided Mission Scenarios  │
└──────────────────────────────┘              └──────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│               ASTRONAUT MISSION CONTROL DASHBOARD (Streamlit)          │
│  • Synchronized Video Scrubbing & Contour Mask Overlays                │
│  • Real-Time Physics Telemetry Charts (Temp, Pressure, Burn Rate)      │
│  • Predictive Fire Hazard Assessment & Ventilation Guidance Card       │
│  • Flight Safety Handbook & Cool Flame Emergency Checklist             │
└────────────────────────────────────────────────────────────────────────┘
```

### Module Descriptions
- **`modules/flame_cv.py` (Computer Vision):**
  - Isolates and enhances the **Blue channel** using Contrast Limited Adaptive Histogram Equalization (CLAHE) to reveal ultra-faint $\text{CH}^*$ chemiluminescence.
  - Adaptive thresholding and morphological cleanup.
  - Ellipse fitting to compute equivalent spherical diameter $D_{eff} = 2\sqrt{A/\pi}$, aspect ratio, and centroid drift $(\Delta x, \Delta y)$.
- **`modules/physics_engine.py` (Combustion Physics):**
  - Savitzky-Golay noise filtering and $d^2$-law linear regression.
  - Automatic extinction detection (radiative quenching vs cool flame transition vs flow blowoff).
  - Grashof regime classifier and Minimum Oxygen Index (MOI) physics calculations.
- **`modules/ml_model.py` (Spacecraft Fire Hazard Predictor):**
  - Dual Random Forest Regressor & Classifier trained on NASA PSI flight test conditions.
  - Inputs: Pressure ($kPa$), $O_2\%$, forced ventilation ($m/s$), droplet diameter ($mm$), fuel type.
  - Outputs: Predicted burn time, cool flame risk probability, composite hazard rating (LOW / MODERATE / HIGH / CRITICAL), and actionable ventilation throttling commands.
- **`static/playground.html` (Interactive Physics Simulator):**
  - Standalone, zero-dependency HTML5 Canvas / p5.js particle simulation.
  - Models radial thermal diffusion, variable gravitational buoyancy ($0g$ to $1g$), forced airflow convection, and local oxygen consumption.
  - Includes one-click guided NASA mission scenarios (*Suffocation Experiment*, *Ventilation Trap*, *Apollo 1 Warning*).
- **`modules/mock_data_generator.py` (Synthetic NASA PSI Data Generator):**
  - Synthesizes realistic CIR video clips (.mp4) and synchronized chamber telemetry (.csv) across all gravity levels ($0g$, Lunar, Mars, Earth) and fuel profiles.

---

## 4. Repository Structure

```
microg-flamevision/
├── AI_AGENT_CONTEXT.md          # Technical specifications, domain invariants & data schemas
├── PROJECT_SPECIFICATION.md     # Hackathon mission scope & architectural roadmap
├── README.md                    # Project documentation & presentation summary
├── requirements.txt             # Python package dependencies
├── app.py                       # Streamlit multi-tab main application
├── modules/
│   ├── __init__.py              # Package exports
│   ├── flame_cv.py              # Blue CLAHE enhancement, contour segmentation, ellipse fitting
│   ├── physics_engine.py        # d²-law regression, Grashof analysis, extinction detector
│   ├── ml_model.py              # Random Forest fire hazard predictor & training pipeline
│   └── mock_data_generator.py   # Synthetic video and telemetry generator
├── static/
│   ├── playground.html          # Interactive 0g/1g Canvas physics simulator
│   └── style.css                # Custom CSS design system
├── data/
│   ├── samples/                 # Sample NASA CIR video clips and telemetry CSV files
│   └── synthetic/               # Generated test fixtures and synthetic datasets
└── tests/
    ├── test_cv.py               # Unit tests for computer vision pipeline
    ├── test_physics.py          # Unit tests for d²-law and physics math
    └── test_model.py            # Unit tests for ML hazard predictor
```

---

## 5. Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/microg-flamevision.git
cd microg-flamevision

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Demo Datasets
Pre-populate synthetic microgravity combustion video clips and telemetry CSVs:
```bash
python3 modules/mock_data_generator.py
```

### 3. Launch the Astronaut Mission Control Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to explore:
- **Tab 1 (🧪 Mission Telemetry & CV Analyzer):** Upload or select combustion videos, analyze frame contours, inspect $d^2$-law regression, and view ML hazard assessments.
- **Tab 2 (🎮 Gravity & Combustion Playground):** Interact with the real-time 0g vs 1g particle simulation, throttle oxygen, and test emergency scenarios.
- **Tab 3 (📋 Spacecraft Safety Manual):** Review operational protocols, cool flame technical summaries, and extinguishing procedures.

### 4. Run Unit Test Suite
Verify all computer vision, physics regression, and ML hazard components:
```bash
pytest tests/ -v
```

---

## 6. NASA Space Apps Judging Alignment

- **Impact (25%):** Directly addresses life-critical fire safety for long-duration deep-space exploration (Artemis Lunar Base, Gateway, Mars Transit) by detecting invisible cool flames and optimizing life-support ventilation controls.
- **Creativity (25%):** Pairs specialized blue-channel CLAHE computer vision with an intuitive 2D fluid-particle simulator that allows users to experience $0g$ combustion dynamics firsthand.
- **Validity (25%):** Strictly adheres to verified NASA Physical Sciences Informatics (PSI) flight telemetry, the classical droplet $d^2$-law, and peer-reviewed combustion fluid dynamics.
- **Relevance (15%):** Purpose-built for the *Flame in Freefall* challenge track of the 2026 NASA Space Apps Challenge.
- **Presentation (10%):** Clean aerospace dark-mode UI, modular clean code, comprehensive unit test suite, and standalone educational simulator.

---

## 7. License & Credits

Developed for the **NASA International Space Apps Challenge 2026**.  
Released under the **MIT License**.
