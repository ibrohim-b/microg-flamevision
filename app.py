"""
MicroG-FlameVision: Astronaut Mission Control Dashboard

Streamlit application with three tabs:
1. 🧪 Mission Telemetry & CV Analyzer - Video upload, frame processing, metric charts
2. 🎮 Gravity & Combustion Playground - Embedded interactive physics simulator
3. 📋 Spacecraft Safety Manual - Emergency protocols, cool flame summaries, guidelines
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Local modules
from modules.flame_cv import FlameCVProcessor, CVConfig, process_video_frames, create_debug_visualization
from modules.physics_engine import (
    analyze_flame_metrics,
    compute_d2_burn_rate,
    detect_extinction_event,
    compute_grashof_number,
    D2LawResult,
)
from modules.ml_model import predict_hazard, HazardAssessment
from modules.mock_data_generator import (
    generate_synthetic_video,
    generate_telemetry_csv,
    FlameParams,
    MockFlameGenerator,
)

# ============================================
# PAGE CONFIG & STYLING
# ============================================
st.set_page_config(
    page_title="MicroG-FlameVision | NASA Space Apps 2026",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown("""
<style>
/* Main theme colors */
:root {
    --bg-primary: #0A0E17;
    --bg-secondary: #121826;
    --bg-card: #1E293B;
    --accent-cyan: #38BDF8;
    --accent-amber: #F59E0B;
    --accent-red: #EF4444;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --border: #334155;
}

/* Override Streamlit defaults */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

header[data-testid="stHeader"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.metric-card h4 {
    margin: 0 0 0.5rem 0;
    color: var(--text-secondary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-cyan);
}

.metric-card .value.warning { color: var(--accent-amber); }
.metric-card .value.danger { color: var(--accent-red); }
.metric-card .value.good { color: #22C55E; }

/* Hazard level badges */
.hazard-badge {
    display: inline-block;
    padding: 0.375rem 0.875rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.hazard-LOW { background: rgba(34, 197, 94, 0.2); color: #22C55E; border: 1px solid #22C55E; }
.hazard-MODERATE { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
.hazard-HIGH { background: rgba(249, 115, 22, 0.2); color: #F97316; border: 1px solid #F97316; }
.hazard-CRITICAL { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-header::before {
    content: '';
    width: 10px;
    height: 10px;
    border-radius: 2px;
    background: var(--accent-cyan);
}

/* Buttons */
.stButton > button {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    border-radius: 8px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent-cyan);
    color: var(--bg-primary);
}
.stButton > button[kind="secondary"] {
    background: transparent;
    border-color: var(--border);
    color: var(--text-secondary);
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-amber);
    color: var(--accent-amber);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-cyan) !important;
    color: var(--bg-primary) !important;
}

/* Dataframes */
.stDataFrame {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* File uploader */
.stFileUploader {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 2rem;
}
.stFileUploader:hover {
    border-color: var(--accent-cyan);
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
}
.streamlit-expanderContent {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* Progress bar */
.stProgress > div > div > div {
    background: var(--accent-cyan);
}
</style>
""", unsafe_allow_html=True)


# ============================================
# HELPER FUNCTIONS
# ============================================
@st.cache_data(show_spinner=False)
def load_sample_data(gravity: str = "micro", fuel: str = "heptane") -> tuple[list, pd.DataFrame]:
    """Generate or load sample video frames and telemetry."""
    data_dir = Path(__file__).parent / "data" / "synthetic"
    video_path = data_dir / f"{fuel}_{gravity}g.mp4"
    csv_path = data_dir / f"{fuel}_{gravity}g.csv"

    if video_path.exists() and csv_path.exists():
        # Load existing
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        telemetry = pd.read_csv(csv_path)
        return frames, telemetry

    # Generate new
    with st.spinner(f"Generating synthetic {gravity}g {fuel} data..."):
        frames, telemetry_data = generate_synthetic_video(
            video_path.with_suffix(""),
            gravity=gravity,  # type: ignore
            fuel_type=fuel,  # type: ignore
            burn_time_s=20.0,
            fps=10,
            cool_flame_transition=(gravity == "micro"),
        )
        telemetry = pd.DataFrame(telemetry_data)
    return frames, telemetry


def render_metric_card(title: str, value: str, delta: str | None = None, status: str = "normal") -> None:
    """Render a styled metric card."""
    status_class = {"normal": "", "warning": "warning", "danger": "danger", "good": "good"}.get(status, "")
    delta_html = f'<div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.25rem;">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <div class="value {status_class}">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_hazard_badge(level: str) -> None:
    """Render hazard level badge."""
    st.markdown(f'<span class="hazard-badge hazard-{level}">{level}</span>', unsafe_allow_html=True)


def plot_d2_law(timestamps: np.ndarray, diameters: np.ndarray, result: D2LawResult) -> go.Figure:
    """Create Plotly figure for d²-law analysis."""
    d2_raw = diameters ** 2
    d2_smooth = result.d2_smooth if hasattr(result, 'd2_smooth') else result.d2_smoothed

    fig = go.Figure()

    # Raw D²
    fig.add_trace(go.Scatter(
        x=timestamps, y=d2_raw,
        mode='markers', name='Raw D²',
        marker=dict(size=4, color='rgba(148, 163, 184, 0.6)'),
        hovertemplate='t=%{x:.2f}s<br>D²=%{y:.2f} mm²<extra></extra>'
    ))

    # Smoothed D²
    fig.add_trace(go.Scatter(
        x=timestamps, y=d2_smooth,
        mode='lines', name='Smoothed D²',
        line=dict(color='#38BDF8', width=2),
        hovertemplate='t=%{x:.2f}s<br>D²=%{y:.2f} mm²<extra></extra>'
    ))

    # Linear fit
    if result.is_valid:
        t_fit = np.array([timestamps[result.steady_start_idx], timestamps[result.steady_end_idx]])
        d2_fit = result.d0_squared_mm2 - result.burning_rate_k * t_fit
        fig.add_trace(go.Scatter(
            x=t_fit, y=d2_fit,
            mode='lines', name=f'Fit: K={result.burning_rate_k:.3f} mm²/s',
            line=dict(color='#F59E0B', width=3, dash='dash'),
            hovertemplate='K=%{y:.3f} mm²/s<extra></extra>'
        ))

    # Steady regime shading
    fig.add_vrect(
        x0=timestamps[result.steady_start_idx],
        x1=timestamps[result.steady_end_idx],
        fillcolor="rgba(56, 189, 248, 0.1)",
        layer="below", line_width=0,
        annotation_text="Steady Regime",
        annotation_position="top left"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#121826',
        plot_bgcolor='#0A0E17',
        font=dict(color='#F8FAFC'),
        xaxis_title="Time (s)",
        yaxis_title="D² (mm²)",
        title="d²-Law Regression Analysis",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig


def plot_diameter_timeseries(timestamps: np.ndarray, diameters: np.ndarray,
                              extinction_event=None, cool_flame_time=None) -> go.Figure:
    """Create diameter vs time plot with annotations."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps, y=diameters,
        mode='lines+markers', name='Diameter',
        line=dict(color='#38BDF8', width=2),
        marker=dict(size=4),
        hovertemplate='t=%{x:.2f}s<br>D=%{y:.2f} mm<extra></extra>'
    ))

    if extinction_event:
        fig.add_vline(
            x=extinction_event.extinction_time_s,
            line=dict(color='#EF4444', width=2, dash='dot'),
            annotation_text=f"Extinction: {extinction_event.mechanism}",
            annotation_position="top right"
        )

    if cool_flame_time:
        fig.add_vline(
            x=cool_flame_time,
            line=dict(color='#A855F7', width=2, dash='dot'),
            annotation_text="Cool Flame Transition",
            annotation_position="top left"
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#121826',
        plot_bgcolor='#0A0E17',
        font=dict(color='#F8FAFC'),
        xaxis_title="Time (s)",
        yaxis_title="Diameter (mm)",
        title="Flame Diameter Evolution",
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig


def plot_telemetry(telemetry: pd.DataFrame) -> go.Figure:
    """Create multi-panel telemetry plot."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Chamber Pressure (kPa)", "O₂ Concentration (%)", "Radiometer (W/cm²) / Thermocouple (°C)"),
        row_heights=[0.3, 0.3, 0.4]
    )

    t = telemetry['timestamp_s']

    fig.add_trace(go.Scatter(x=t, y=telemetry['chamber_pressure_kpa'],
                             mode='lines', name='Pressure', line=dict(color='#38BDF8')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=telemetry['o2_percent'],
                             mode='lines', name='O₂ %', line=dict(color='#F59E0B')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=telemetry['radiometer_w_cm2'],
                             mode='lines', name='Radiometer', line=dict(color='#EF4444'), yaxis='y'), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=telemetry['thermocouple_temp_c'],
                             mode='lines', name='Thermocouple', line=dict(color='#22C55E'), yaxis='y2'), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#121826',
        plot_bgcolor='#0A0E17',
        font=dict(color='#F8FAFC'),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="kPa", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="W/cm²", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


# ============================================
# TAB 1: MISSION TELEMETRY & CV ANALYZER
# ============================================
def tab_analyzer():
    st.markdown('<div class="section-header"><h2>🧪 Mission Telemetry & CV Analyzer</h2></div>', unsafe_allow_html=True)

    # Sidebar controls for this tab
    with st.sidebar:
        st.markdown("### 🎛️ Analyzer Controls")

        data_source = st.radio(
            "Data Source",
            ["Synthetic (ISS Microgravity)", "Synthetic (Earth 1g)", "Synthetic (Cool Flame)", "Upload Video"],
            index=0,
        )

        if data_source == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload combustion video",
                type=["mp4", "avi", "mov", "mkv"],
                help="MP4/H.264 recommended. Max 100MB."
            )
            uploaded_csv = st.file_uploader(
                "Upload telemetry CSV (optional)",
                type=["csv"],
                help="Columns: timestamp_s, chamber_pressure_kpa, o2_percent, cabin_flow_mps, fuel_type, radiometer_w_cm2, thermocouple_temp_c",
            )
        else:
            uploaded_file = None
            uploaded_csv = None

        st.divider()

        # CV Parameters
        st.markdown("#### CV Parameters")
        scale_px_per_mm = st.slider("Scale (px/mm)", 5.0, 50.0, 20.0, 0.5)
        min_area = st.slider("Min Contour Area (px)", 10, 500, 50)
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 3.0, 0.5)
        sample_every = st.slider("Sample Every N Frames", 1, 10, 1)
        enhancement = st.selectbox("Enhancement Method", ["blue_clahe", "hsv"])

        st.divider()
        if st.button("🔄 Regenerate Synthetic Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load data
    if data_source == "Upload Video" and uploaded_file:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if uploaded_csv:
            telemetry = pd.read_csv(uploaded_csv)
        else:
            telemetry = pd.DataFrame()  # Empty
    else:
        # Map selection to gravity/fuel
        gravity_map = {
            "Synthetic (ISS Microgravity)": ("micro", "heptane"),
            "Synthetic (Earth 1g)": ("earth", "heptane"),
            "Synthetic (Cool Flame)": ("micro", "heptane"),
        }
        gravity, fuel = gravity_map[data_source]
        frames, telemetry = load_sample_data(gravity, fuel)

    if not frames:
        st.error("No frames loaded. Please check your video file.")
        return

    # Process frames
    config = CVConfig(
        scale_px_per_mm=scale_px_per_mm,
        min_contour_area_px=min_area,
        clahe_clip_limit=clahe_clip,
    )
    processor = FlameCVProcessor(config)

    progress_bar = st.progress(0, text="Processing frames...")
    metrics_list = []

    for i in range(0, len(frames), sample_every):
        frame = frames[i]
        timestamp = i / 10.0  # Assume 10 fps for synthetic
        metrics = processor.process_frame(frame, i, timestamp, enhancement)
        metrics_list.append(metrics)
        progress_bar.progress((i + 1) / len(frames), text=f"Processing frame {i+1}/{len(frames)}")

    progress_bar.empty()

    # Filter detected
    detected = [m for m in metrics_list if m.flame_detected]
    timestamps = np.array([m.timestamp_s for m in detected])
    diameters = np.array([m.equivalent_diameter_mm for m in detected])
    blue_intensities = np.array([m.mean_blue_intensity for m in detected])

    # ============================================
    # RESULTS DISPLAY
    # ============================================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Frames", str(len(frames)))
    with col2:
        render_metric_card("Frames Analyzed", str(len(metrics_list)))
    with col3:
        render_metric_card("Flames Detected", str(len(detected)),
                          f"{len(detected)/len(metrics_list)*100:.1f}% detection rate")
    with col4:
        if len(detected) > 0:
            render_metric_card("Avg Diameter", f"{np.mean(diameters):.2f} mm",
                              f"Range: {np.min(diameters):.2f}-{np.max(diameters):.2f}")

    st.divider()

    # Main charts row
    if len(detected) >= 5:
        # Physics analysis
        analysis = analyze_flame_metrics(metrics_list)

        d2_result = analysis.get('d2_law', {})
        extinction = analysis.get('extinction', {})
        cool_flame = analysis.get('cool_flame', {})
        grashof = analysis.get('grashof', {})
        summary = analysis.get('summary', {})

        # Row 1: d²-law and Diameter evolution
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if d2_result:
                d2_obj = D2LawResult(
                    burning_rate_k=d2_result.get('burning_rate_k_mm2_s', 0),
                    burning_rate_k_stderr=d2_result.get('burning_rate_stderr', 0),
                    r_squared=d2_result.get('r_squared', 0),
                    d0_squared_mm2=d2_result.get('d0_squared_mm2', 0),
                    d0_squared_stderr=0,
                    steady_start_idx=d2_result.get('steady_regime_frames', (0,0))[0],
                    steady_end_idx=d2_result.get('steady_regime_frames', (0,0))[1],
                    timestamps_used=timestamps,
                    d2_smoothed=np.array([m.diameter_squared_mm2 for m in detected]),
                    d2_raw=diameters**2,
                    residuals=np.array([]),
                    is_valid=d2_result.get('quality') != 'POOR',
                    quality_flag=d2_result.get('quality', 'POOR'),
                )
                fig_d2 = plot_d2_law(timestamps, diameters, d2_obj)
                st.plotly_chart(fig_d2, use_container_width=True)

        with chart_col2:
            ext_event = None
            if extinction.get('detected'):
                from modules.physics_engine import ExtinctionEvent
                ext_event = ExtinctionEvent(
                    extinction_time_s=extinction['time_s'],
                    extinction_frame_idx=0,
                    mechanism=extinction['mechanism'],
                    confidence=extinction['confidence'],
                    pre_extinction_diameter_mm=0,
                )
            cool_time = cool_flame.get('transition_time_s')
            fig_dia = plot_diameter_timeseries(timestamps, diameters, ext_event, cool_time)
            st.plotly_chart(fig_dia, use_container_width=True)

        # Row 2: Telemetry
        if not telemetry.empty:
            fig_tele = plot_telemetry(telemetry)
            st.plotly_chart(fig_tele, use_container_width=True)

        # Row 3: Metrics & Hazard Assessment
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("Burning Rate K", f"{d2_result.get('burning_rate_k_mm2_s', 0):.3f} mm²/s",
                              f"R² = {d2_result.get('r_squared', 0):.3f} ({d2_result.get('quality', 'N/A')})",
                              "good" if d2_result.get('quality') == 'GOOD' else "warning" if d2_result.get('quality') == 'FAIR' else "danger")
        with metric_cols[1]:
            render_metric_card("Initial D₀²", f"{d2_result.get('d0_squared_mm2', 0):.1f} mm²")
        with metric_cols[2]:
            ext_time = extinction.get('time_s', 'N/A')
            render_metric_card("Extinction Time", f"{ext_time:.2f} s" if isinstance(ext_time, float) else "N/A",
                              f"Mechanism: {extinction.get('mechanism', 'N/A')}")
        with metric_cols[3]:
            regime = grashof.get('regime', 'unknown')
            gr_num = grashof.get('number', 0)
            render_metric_card("Grashof Regime", regime.replace('_', ' ').title(),
                              f"Gr = {gr_num:.2e}")

        st.divider()

        # Hazard Assessment
        st.markdown('<div class="section-header"><h3>🚨 Spacecraft Fire Hazard Assessment</h3></div>', unsafe_allow_html=True)

        # Get fuel type from telemetry or default
        fuel_type = telemetry['fuel_type'].iloc[0] if not telemetry.empty and 'fuel_type' in telemetry.columns else "heptane"
        pressure = telemetry['chamber_pressure_kpa'].mean() if not telemetry.empty else 101.3
        o2 = telemetry['o2_percent'].mean() if not telemetry.empty else 21.0
        flow = telemetry['cabin_flow_mps'].mean() if not telemetry.empty else 0.05
        init_dia = diameters[0] if len(diameters) > 0 else 8.0

        hazard: HazardAssessment = predict_hazard(pressure, o2, flow, init_dia, fuel_type)

        haz_cols = st.columns([1, 1, 1, 1])
        with haz_cols[0]:
            render_metric_card("Predicted Extinction", f"{hazard.predicted_time_to_extinction_s:.1f} s",
                              f"K = {hazard.burning_rate_k_mm2_s:.3f} mm²/s")
        with haz_cols[1]:
            render_metric_card("Cool Flame Risk", f"{hazard.cool_flame_risk_pct:.1f}%",
                              status="danger" if hazard.cool_flame_risk_pct > 50 else "warning" if hazard.cool_flame_risk_pct > 25 else "good")
        with haz_cols[2]:
            render_metric_card("Min Quench O₂", f"{hazard.minimum_quench_oxygen_pct:.1f}%",
                              f"Current: {o2:.1f}%",
                              "danger" if o2 < hazard.minimum_quench_oxygen_pct else "good")
        with haz_cols[3]:
            st.markdown("**Hazard Level**")
            render_hazard_badge(hazard.hazard_level)
            st.markdown(f"<div style='font-size:0.875rem;color:var(--text-secondary);margin-top:0.5rem;'>Confidence: {hazard.confidence:.0%}</div>", unsafe_allow_html=True)

        st.markdown("#### 🛡️ Ventilation Recommendation")
        st.info(hazard.ventilation_recommendation)

        # Feature importance
        if hazard.feature_importance:
            st.markdown("#### 📊 Feature Importance (Model Explainability)")
            fi_df = pd.DataFrame(list(hazard.feature_importance.items()), columns=['Feature', 'Importance'])
            fi_df = fi_df.sort_values('Importance', ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'], y=fi_df['Feature'],
                orientation='h', marker_color='#38BDF8'
            ))
            fig_fi.update_layout(
                template="plotly_dark", paper_bgcolor='#121826', plot_bgcolor='#0A0E17',
                height=300, margin=dict(l=100, r=20, t=20, b=40),
                xaxis_title="Importance", yaxis_title=""
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    else:
        st.warning("Insufficient flame detections for physics analysis. Need at least 5 detected frames.")

    # Frame gallery
    with st.expander("🖼️ Frame Gallery (First 20 frames)", expanded=False):
        gallery_cols = st.columns(5)
        for idx, m in enumerate(metrics_list[:20]):
            with gallery_cols[idx % 5]:
                if m.flame_detected:
                    # Create visualization
                    frame = frames[m.frame_index] if m.frame_index < len(frames) else frames[0]
                    enhanced = processor.enhance_faint_flame(frame)
                    binary = processor.threshold_flame(enhanced)
                    vis = create_debug_visualization(frame, m, enhanced, binary)
                    st.image(vis, caption=f"Frame {m.frame_index} (t={m.timestamp_s:.1f}s)", use_container_width=True)
                else:
                    st.image(frames[m.frame_index] if m.frame_index < len(frames) else frames[0],
                            caption=f"Frame {m.frame_index}: No flame", use_container_width=True)

    # Download results
    if len(detected) > 0:
        st.divider()
        results_df = pd.DataFrame([m.to_dict() for m in detected])
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download CV Results (CSV)",
            csv,
            file_name="flame_cv_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================
# TAB 2: GRAVITY & COMBUSTION PLAYGROUND
# ============================================
def tab_playground():
    st.markdown('<div class="section-header"><h2>🎮 Gravity & Combustion Playground</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Interactive real-time simulation of flame physics across gravity regimes.
    Adjust parameters in the sidebar to see how gravity, oxygen, and ventilation affect flame behavior.
    """)

    # Embed the HTML playground
    playground_path = Path(__file__).parent / "static" / "playground.html"
    if playground_path.exists():
        with open(playground_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.error("Playground HTML not found. Please check static/playground.html exists.")

    st.divider()

    # Quick reference
    with st.expander("📚 Physics Reference", expanded=False):
        st.markdown("""
        ### Key Physics Principles

        **Microgravity (≈0g)**
        - No buoyant convection → spherical, stationary flames
        - Diffusion-dominated heat/mass transport
        - Faint blue chemiluminescence (CH* 430nm, C₂* 516nm)
        - Low soot, low radiative emission
        - Self-extinction by CO₂/H₂O blanket

        **Terrestrial (1g)**
        - Strong buoyant convection → elongated teardrop flames
        - Convection-dominated transport
        - Bright yellow/orange soot incandescence
        - High radiative emission
        - Extinction by thermal losses or blowoff

        **Grashof Number** = g·β·ΔT·L³/ν²
        - Gr << 1: Diffusion dominated (microgravity)
        - Gr ~ 1-100: Transitional
        - Gr >> 100: Buoyancy dominated (Earth)

        **d²-Law**: D²(t) = D₀² - K·t
        - K = burning rate constant (mm²/s)
        - Linear in steady regime
        - Deviations indicate instabilities or cool flames

        **Cool Flames**
        - Secondary low-temperature oxidation
        - Near-infrared/blue emission
        - Can persist after visible extinction
        - Hazard: undetectable by standard smoke sensors
        """)


# ============================================
# TAB 3: SPACECRAFT SAFETY MANUAL
# ============================================
def tab_safety_manual():
    st.markdown('<div class="section-header"><h2>📋 Spacecraft Safety Manual</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    **MicroG-FlameVision Safety Reference** — Based on NASA FLEX, ACME, BASS, and Saffire flight data.
    """)

    # Emergency Protocols
    with st.expander("🚨 Emergency Fire Response Protocols", expanded=True):
        st.markdown("""
        ### Immediate Actions (First 30 Seconds)

        1. **ALERT**: Activate fire alarm, notify crew and ground
        2. **VENTILATION**:
           - **Microgravity**: REDUCE ventilation to < 0.03 m/s (starve diffusion flame)
           - **DO NOT INCREASE** flow — creates artificial buoyancy, spreads flame
        3. **SUPPRESSION**: Deploy water mist or CO₂ at base of flame
        4. **ISOLATE**: Close hatches, power down affected rack
        5. **MONITOR**: Watch for cool flame persistence (IR camera required)

        ### Decision Tree
        ```
        Fire Detected
            │
            ├─→ Visible flame? → YES → Reduce ventilation → Suppress → Monitor
            │                      NO  → Check IR/thermal → Cool flame?
            │                                              │
            │                                              ├─ YES → Maintain low ventilation, deploy suppression
            │                                              └─ NO  → Verify false alarm, monitor 30 min
        ```
        """)

    # Cool Flame Technical Summary
    with st.expander("❄️ Cool Flame Technical Summary", expanded=False):
        st.markdown("""
        ### What Are Cool Flames?

        Cool flames are low-temperature (~600-800°C) self-sustaining oxidation reactions that occur
        after hot-flame extinction in microgravity. They are **invisible to standard smoke detectors**
        and visible-light cameras.

        **Characteristics:**
        - Temperature: 600-800°C (vs 1400-1800°C for hot flames)
        - Emission: Near-infrared (CH₂O* formaldehyde) and faint blue
        - Duration: Can persist 10-100x longer than hot flame
        - Fuel consumption: Continues consuming fuel at reduced rate
        - Re-ignition risk: Can transition back to hot flame if ventilation increases

        **Detection Requirements:**
        - IR cameras (3-5 μm band) or UV cameras
        - Gas analyzers (CO, formaldehyde detection)
        - Thermocouple arrays (surface temperature monitoring)

        **NASA Findings (FLEX/ACME):**
        - Observed in n-alkanes (heptane, hexane, dodecane) at 101 kPa, 21% O₂
        - More likely at lower O₂ (< 18%) and quiescent conditions
        - Transition occurs at ~70% of hot-flame burn time
        - Second-stage K ≈ 0.1-0.15 × primary K
        """)

    # Extinguishing Guidelines
    with st.expander("🧯 Extinguishing Guidelines by Gravity Regime", expanded=False):
        st.markdown("""
        ### Microgravity (ISS, Transit Vehicles)
        | Agent | Effectiveness | Notes |
        |-------|--------------|-------|
        | Water Mist | ★★★★★ | Best overall; cools + displaces O₂ |
        | CO₂ | ★★★★☆ | Good but creates toxic CO₂ buildup |
        | N₂ | ★★★☆☆ | Requires high concentration |
        | Halon replacements | ★★★★☆ | Effective but limited supply |

        **Key Principle**: Starve the diffusion envelope. Reduce ventilation FIRST.

        ### Lunar Gravity (0.16g)
        - Weak buoyancy elongates flame slightly
        - Water mist still preferred
        - Ventilation reduction less critical but still recommended

        ### Mars Gravity (0.38g)
        - Moderate buoyancy; flame shape transitional
        - Standard CO₂ extinguishers effective
        - Directed nozzle at flame base critical

        ### Terrestrial (1g) - Ground Testing
        - Standard fire protocols apply
        - Aim at base of flame
        - Ventilation control less effective (buoyancy dominates)
        """)

    # Minimum Oxygen Index Table
    with st.expander("📊 Minimum Oxygen Index (MOI) Reference", expanded=False):
        moi_data = {
            "Fuel": ["Heptane", "Methanol", "Hexane", "Ethylene", "PMMA", "Polyethylene", "Cotton"],
            "Microgravity MOI (%)": [13.5, 11.0, 13.0, 12.0, 14.5, 14.0, 16.0],
            "Lunar MOI (%)": [12.8, 10.5, 12.4, 11.4, 13.8, 13.3, 15.2],
            "Mars MOI (%)": [12.2, 9.9, 11.7, 10.8, 13.1, 12.6, 14.4],
            "Earth MOI (%)": [11.5, 9.4, 11.1, 10.2, 12.3, 11.9, 13.6],
        }
        moi_df = pd.DataFrame(moi_data)
        st.dataframe(moi_df, use_container_width=True, hide_index=True)

        st.caption("MOI = Minimum O₂ concentration to sustain combustion. Below this, flame self-extinguishes.")

    # Ventilation Safety Limits
    with st.expander("💨 Ventilation Safety Limits", expanded=False):
        st.markdown("""
        ### ISS Cabin Ventilation Parameters

        **Standard Operation:**
        - Velocity: 0.05 - 0.20 m/s
        - Direction: Ceiling-to-floor (creates artificial gravity)

        **Fire Emergency Limits:**
        | Scenario | Max Velocity | Rationale |
        |----------|-------------|-----------|
        | Microgravity diffusion flame | **0.03 m/s** | Prevent artificial buoyant stretching |
        | Cool flame persistence | **0.02 m/s** | Minimize O₂ replenishment |
        | Solid fuel (PMMA) | **0.05 m/s** | Balance smoke removal vs flame spread |
        | Post-extinction monitoring | **0.05 m/s** | Clear combustion products |

        **The Ventilation Trap:**
        Increasing ventilation in microgravity to "clear smoke" creates forced convection
        that elongates the flame, increases burning rate 2-3x, and can cause sudden
        re-ignition of cool flames. **Always reduce ventilation first.**
        """)

    # Quick Reference Cards
    st.divider()
    st.markdown("### ⚡ Quick Reference Cards")

    qr_cols = st.columns(3)
    with qr_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <h4>🔴 Hot Flame</h4>
            <div class="value">1400-1800°C</div>
            <div style="font-size:0.875rem;color:var(--text-secondary);margin-top:0.5rem;">
            Visible yellow/orange (soot)<br>
            High radiometer signal<br>
            Standard smoke detection works
            </div>
        </div>
        """, unsafe_allow_html=True)

    with qr_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <h4>🔵 Microgravity Flame</h4>
            <div class="value">800-1200°C</div>
            <div style="font-size:0.875rem;color:var(--text-secondary);margin-top:0.5rem;">
            Spherical, blue (CH*/C₂*)<br>
            Low radiometer signal<br>
            Need blue-channel CV enhancement
            </div>
        </div>
        """, unsafe_allow_html=True)

    with qr_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <h4>❄️ Cool Flame</h4>
            <div class="value warning">600-800°C</div>
            <div style="font-size:0.875rem;color:var(--text-secondary);margin-top:0.5rem;">
            Near-IR / faint blue<br>
            Invisible to smoke detectors<br>
            Requires IR/UV/thermal monitoring
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:var(--text-secondary);font-size:0.875rem;">
        <p><strong>MicroG-FlameVision</strong> — NASA Space Apps Challenge 2026</p>
        <p>Track: <em>Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data</em></p>
        <p>Data Sources: NASA PSI (FLEX, ACME, BASS, Saffire) | Physics: d²-Law, Grashof Analysis, Combustion Theory</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #121826 0%, #0A0E17 100%);
        border-bottom: 1px solid #1E293B;
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div style="display:flex;align-items:center;gap:1rem;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <path d="M8.5 14.5A2.5 2.5 0 0 1 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a5.5 5.5 0 0 1-10 0c0-2.5 1-4.4 3-5.5"/>
                <path d="M15.5 14.5A2.5 2.5 0 0 0 13 12c0-1.38.5-2 1-3 1.072-2.143.224-4.054-2-6-.5 2.5-2 4.9-4 6.5-2 1.6-3 3.5-3 5.5a5.5 5.5 0 0 0 10 0c0-2.5-1-4.4-3-5.5"/>
            </svg>
            <div>
                <h1 style="margin:0;font-size:1.5rem;font-weight:700;">MicroG-FlameVision</h1>
                <p style="margin:0;color:#94A3B8;font-size:0.875rem;">NASA Space Apps 2026 — Flame in Freefall</p>
            </div>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center;">
            <span style="color:#94A3B8;font-size:0.875rem;">Mission Control Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧪 Mission Telemetry & CV Analyzer",
        "🎮 Gravity & Combustion Playground",
        "📋 Spacecraft Safety Manual"
    ])

    with tab1:
        tab_analyzer()

    with tab2:
        tab_playground()

    with tab3:
        tab_safety_manual()


if __name__ == "__main__":
    main()