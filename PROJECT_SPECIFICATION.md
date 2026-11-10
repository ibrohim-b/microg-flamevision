# MicroG-FlameVision: Project Specification
**NASA International Space Apps Challenge 2026**  
**Challenge Track:** *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*  
**Theme:** *The Next Frontier*

---

## 1. Executive Summary & Mission Statement

Fire is one of the most perilous hazards in human spaceflight. Inside an enclosed spacecraft or orbital habitat, toxic smoke cannot be ventilated to an outside atmosphere, and crew evacuation is limited or impossible. 

Decades of NASA experiments have revealed that **fire behaves fundamentally differently in microgravity**:
* Without gravity, natural buoyant convection vanishes. Flames become spherical, burn cooler, emit minimal soot, and glow faintly blue or near-infrared.
* In some conditions, flames enter a self-sustaining **"cool flame" regime** undetectable by standard optical smoke sensors.
* Forced ventilation (cabin air conditioning) acts as an artificial gravity, suddenly elongating flames and driving unexpected re-ignition.

**MicroG-FlameVision** is an open-source, dual-purpose platform combining:
1. **An AI Computer Vision & Physics Analytics Engine:** Processes raw microgravity combustion video archives from the International Space Station (ISS) and drop towers, isolating faint flames, measuring burn rates ($d^2$-law), and predicting fire-suppression limits.
2. **An Interactive Educational & Analytical Playground:** An interactive simulation tool that demonstrates the physics of flame propagation under variable gravity ($0g$ to $1g$), oxygen levels, and ventilation velocities for students, educators, and flight controllers.

---

## 2. Scientific & Physics Foundation

### 2.1 Convection vs. Diffusion

| Physical Characteristic | Terrestrial Gravity ($1g$) | Microgravity ($\approx 0g$) |
| :--- | :--- | :--- |
| **Dominant Transport Mechanism** | Natural buoyant convection | Pure molecular diffusion |
| **Flame Geometry** | Elongated teardrop, flickering tip | Symmetrical sphere, stationary |
| **Color & Luminescence** | Bright yellow/orange (incandescent soot) | Faint blue / near-UV / infrared |
| **Soot Formation** | High (pyrolysis in hot buoyant plume) | Low to none (clean, slow diffusion burn) |
| **Extinction Mechanism** | Thermal losses or fuel depletion | Suffocation by self-emitted $CO_2/H_2O$ blanket |
| **Temperature Profile** | Hot core ($1400\text{–}1800^\circ\text{C}$) | Cooler, diffused envelope ($800\text{–}1200^\circ\text{C}$) |

### 2.2 Mathematical Formulations

#### 1. The $d^2$-Law of Droplet Combustion
Classical droplet burning in quiescent microgravity follows the classical $d^2$-law:
$$D^2(t) = D_0^2 - K \cdot t$$
Where:
* $D(t)$ = Equivalent flame or droplet diameter at time $t$
* $D_0$ = Initial diameter
* $K$ = Evaporation/Burning rate constant ($-\frac{d(D^2)}{dt}$), expressed in $\text{mm}^2/\text{s}$

Deviations from linearity indicate droplet oscillation, radiative extinction, or the transition into second-stage cool flame combustion.

#### 2. Buoyancy Convection vs. Diffusion (Grashof Number)
The Grashof number $Gr$ quantifies the ratio of buoyant forces to viscous forces:
$$Gr = \frac{g \cdot \beta \cdot (T_{flame} - T_\infty) \cdot L^3}{\nu^2}$$
* On Earth ($g = 9.81\text{ m/s}^2$): $Gr \gg 1$ (buoyancy dominates transport).
* In Microgravity ($g \approx 10^{-4}\text{ to } 10^{-6}\text{ m/s}^2$): $Gr \approx 0$ (buoyancy drops to zero; pure diffusion governs heat and mass transfer).

---

## 3. NASA Datasets & Mission Archives

MicroG-FlameVision utilizes open datasets and mission records from:

1. **NASA Physical Sciences Informatics (PSI) System** ([`psi.nasa.gov`](https://psi.nasa.gov))
   * **FLEX / FLEX-2 (Flame Extinguishment Experiment):** Droplet combustion tests on the ISS Combustion Integrated Rack (CIR).
   * **ACME (Advanced Combustion via Microgravity Experiments):** Gaseous burner and soot suppression trials.
   * **BASS & BASS-II (Burning and Suppression of Solids):** Polymer combustion tests under variable airflows.
   * **Saffire Missions (I through VI):** Large-scale spacecraft fire safety tests conducted inside Cygnus cargo resupply vehicles.
2. **NASA Glenn Research Center (GRC) 2.2-Second & 5.18-Second Zero-G Drop Towers:** High-speed video archives of ignition and transient flame development.
3. **Sensor Telemetry Data:** Chamber pressure ($kPa$), oxygen mole fraction ($X_{O_2}$), radiometer fluxes ($\text{W/cm}^2$), thermocouple temperatures ($^\circ\text{C}$), and chamber ventilation speed ($m/s$).

---

## 4. System Architecture & Modules

The platform is structured into five core modules:

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
│ 1. Frame Enhancer (CLAHE/UV) │              │ 1. 2D Fluid / Particle Sim   │
│ 2. Flame Contour Segmentation│              │ 2. Gravity Slider (0g to 1g) │
│ 3. Diameter & Centroid Track │              │ 3. Oxygen & Airflow Modulator│
│ 4. d²-Law & Physics Deriver  │              │ 4. Real-time Extinction Check│
│ 5. Hazard Predictor (ML)     │              │ 5. Guided Mission Scenarios  │
└──────────────────────────────┘              └──────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│            ASTRONAUT MISSION DASHBOARD (Streamlit / Next.js)           │
│  • Video telemetry playback with live bounding masks                   │
│  • Synchronized sensor charts (Temp, Pressure, Burn Rate)              │
│  • Spacecraft Fire Hazard Assessment Card                              │
└────────────────────────────────────────────────────────────────────────┘
```

### Module 1: Computer Vision & Faint-Flame Enhancement (`flame_cv`)
* **Problem:** Space station combustion videos feature low contrast, background chamber glare, and very faint blue emissions.
* **Techniques:**
  * Contrast Limited Adaptive Histogram Equalization (CLAHE) on blue/cyan channels.
  * Adaptive Otsu thresholding or lightweight YOLOv8-segmentation trained on labeled microgravity flame frames.
  * Ellipse fitting to extract minor axis, major axis, equivalent spherical radius $R_{eff} = \sqrt{a \cdot b}$, and centroid drift $(\Delta x, \Delta y)$.

### Module 2: Physics Metric Derivation (`physics_engine`)
* Computes equivalent diameter squared $D^2(t)$ over time.
* Fits linear segments using Savitzky-Golay filtering to obtain burning rate $K$.
* Automatically detects the exact frame and timestamp of visible extinction ($D \to 0$) and radiative quenching.

### Module 3: Spacecraft Fire Hazard Predictor (`ml_model`)
* Supervised model (Random Forest / XGBoost / 1D-CNN) trained on NASA PSI test runs.
* **Inputs:** Ambient pressure ($P$), $O_2\%$, fuel type, cabin ventilation velocity ($v_{air}$), ambient temperature.
* **Outputs:** 
  * Predicted time to self-extinction ($t_{extinct}$).
  * Cool-flame continuation risk score ($0\text{–}100\%$).
  * Minimum oxygen index (MOI) required to extinguish the fire.

### Module 4: Interactive Educational & Analytical Playground (`playground`)
* A web-based, real-time 2D simulation running client-side (HTML5 Canvas / WebGL).
* Enables side-by-side comparison:
  * **Chamber A (Earth - 1g):** Buoyant upward draft, flickering yellow soot flame.
  * **Chamber B (Space - 0g):** Pure radial diffusion, calm blue sphere.
  * **Chamber C (Lunar 0.16g / Mars 0.38g):** Transitional buoyancy regimes.
* Interactive controls: Gravity slider, oxygen slider, airflow fan slider, fuel selector.

### Module 5: Astronaut Mission Control Dashboard (`dashboard`)
* Unified web UI with three tabs:
  1. **🧪 Mission Telemetry & CV Analyzer:** Upload or select test videos and view real-time tracking overlays and metric charts.
  2. **🎮 Gravity & Combustion Playground:** The interactive educational simulator with guided scenarios.
  3. **📋 Spacecraft Safety Manual:** Emergency protocols, cool flame technical summaries, and extinguishing guidelines.

---

## 5. UI / UX Design Specifications

### Color Palette
* Background: Deep space dark (`#0A0E17`, `#121826`)
* Primary Accent: Cyan / Electric Blue (`#38BDF8`) — representing microgravity diffusion
* Warning Accent: Radiant Amber / Solar Flare (`#F59E0B`, `#EF4444`) — representing terrestrial fire & hazards
* Text: Off-white (`#F8FAFC`, `#94A3B8`)

### Key User Interactions
1. **Video Scrubbing & Sync:** Scrubbing the video player dynamically syncs the time-series plot of flame diameter and chamber pressure.
2. **One-Click Guided Scenarios in Playground:**
   * *"The Suffocation Experiment"* (0g, 0 m/s airflow $\to$ flame chokes on $CO_2$).
   * *"The Ventilation Trap"* (0g + AC airflow $\to$ artificial buoyant stretching).
   * *"Apollo 1 Warning"* (34% $O_2$ enriched atmosphere $\to$ explosive flame front).

---

## 6. Technical Stack Recommendations

| Layer | Recommended Technology | Alternatives |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | TypeScript / Node.js |
| **Computer Vision** | OpenCV (`opencv-python-headless`), NumPy, SciPy | PyTorch / YOLOv8-seg |
| **Machine Learning** | Scikit-learn, XGBoost | PyTorch (1D-CNN) |
| **Data Visualization** | Plotly / Altair | Matplotlib, Chart.js |
| **Web Dashboard** | Streamlit or Gradio | Next.js + FastAPI |
| **Physics Playground** | HTML5 Canvas + JavaScript / WebGL | p5.js / Three.js |
| **Deployment** | Hugging Face Spaces / Streamlit Cloud / Vercel | Docker on Render / AWS |

---

## 7. 48-Hour Hackathon Execution Roadmap

```
[Hour 0-8: Data Ingestion & Preprocessing]
  ├── Download sample CIR/FLEX video clips from NASA PSI
  ├── Parse synchronized telemetry CSV files
  └── Build basic OpenCV frame extractor and contrast enhancement script

[Hour 8-20: Computer Vision & Physics Pipeline]
  ├── Implement contour segmentation & ellipse fitting (flame_cv.py)
  ├── Extract D(t) and calculate d²-law burning rate slope K (physics_engine.py)
  └── Implement extinction timestamp detection

[Hour 20-30: Predictive ML & Interactive Playground]
  ├── Train Random Forest hazard model on environmental parameters (ml_model.py)
  ├── Develop 2D Canvas/JS fluid & diffusion simulation (playground.html)
  └── Test 0g vs 1g particle advection under variable ventilation

[Hour 30-40: Dashboard Integration & UI Polish]
  ├── Build Streamlit application integrating all 3 tabs
  ├── Embed Interactive Playground via components.html
  └── Add video overlay player with synchronized Plotly telemetry

[Hour 40-48: Deployment, Documentation & Video Pitch]
  ├── Deploy working demo to Hugging Face Spaces / Web
  ├── Record 2-minute video pitch highlighting NASA data & live demo
  └── Finalize README and NASA Space Apps submission page
```

---

## 8. NASA Judging Criteria Alignment

* **Impact (25%):** Directly addresses crew survival on long-duration missions (Artemis, Mars Transit) by discovering undetectable cool-flame regimes and optimizing fire-suppression protocols.
* **Creativity (25%):** Pairs cutting-edge computer vision with an intuitive, hands-on physics playground for educational and outreach impact.
* **Validity (25%):** Relies strictly on verified NASA Physical Sciences Informatics (PSI) flight telemetry and established combustion physics ($d^2$-law, Grashof analysis).
* **Relevance (15%):** Directly answers all prompts of the official 2026 challenge track.
* **Presentation (10%):** Clean, modern, accessible UI with live web deployment and a crisp 2-minute pitch video.
