import streamlit as st
import pandas as pd
import plotly.graph_objects as go
 
from data_utils import extract_curve_fingerprint
from solver import run_inverse_calibration, TRAINING_BOUNDS
from ai_engine import get_ai_interpretation
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="OPTIC", layout="centered", page_icon="🎯")
 
st.title("🎯 OPTIC")
st.subheader("Ogden Parameter Tuning via Inverse Calibration")
st.write("Upload a physical test CSV to find the optimal Ogden simulation parameters.")
 
# ── Sidebar: info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About OPTIC")
    st.write(
        "OPTIC uses a Gaussian Process surrogate model trained on 28 LPBF "
        "simulation datasets to perform inverse calibration of N=3 Ogden "
        "hyperelastic parameters."
    )
    st.markdown("**Surrogate model:** Gaussian Process (sklearn)")
    st.markdown("**Validation:** Leave-one-out cross validation")
    st.markdown("**AI engine:** Gemma 4 26B via LM Studio")
    st.divider()
    st.markdown("**Training data bounds:**")
    for p, (lo, hi) in TRAINING_BOUNDS.items():
        st.markdown(f"- {p}: [{lo}, {hi}]")
 
# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Force-Displacement CSV",
    type=["csv"],
    help="CSV must have 'Displacement' (mm) and 'Force' (N) columns"
)
 
if uploaded_file is not None:
    st.success("File successfully loaded!")
 
    # ── Step 1: Extract features ──────────────────────────────────────────────
    with st.spinner("Extracting curve fingerprint..."):
        try:
            fingerprint, disp, force = extract_curve_fingerprint(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
 
    st.write("### 📊 Extracted Physical Features")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak Force",  f"{fingerprint['peak_force']:.2f} N")
    col2.metric("Stiffness",   f"{fingerprint['initial_stiffness']:.2f} N/mm")
    col3.metric("Energy",      f"{fingerprint['energy_absorbed']:.4f} N.mm")
    col4.metric("Max Disp",    f"{fingerprint['max_displacement']:.2f} mm")
 
    col5, col6, col7 = st.columns(3)
    col5.metric("Nonlinearity",  f"{fingerprint.get('nonlinearity', 0):.3f}")
    col6.metric("Mid Force",     f"{fingerprint.get('mid_force', 0):.2f} N")
    col7.metric("Power Exponent",f"{fingerprint.get('power_exp', 0):.3f}")
 
    # ── Step 2: Run inverse calibration ──────────────────────────────────────
    with st.spinner("Running GP inverse calibration (this takes ~30 seconds on first run)..."):
        try:
            optimal_params, loo_confidence, bounds_warnings = run_inverse_calibration(fingerprint)
        except FileNotFoundError:
            st.error(
                "Training data not found. Make sure "
                "`rom_history_analysis_format.csv` is in the same folder as `solver.py`."
            )
            st.stop()
        except Exception as e:
            st.error(f"Solver error: {e}")
            st.stop()
 
    # ── Step 3: Display parameters ────────────────────────────────────────────
    st.write("### ⚙️ Recommended Ogden Parameters")
 
    # Honest confidence display
    conf_color = "green" if loo_confidence >= 80 else ("orange" if loo_confidence >= 60 else "red")
    st.markdown(
        f"**Model Confidence (LOO validated): "
        f"<span style='color:{conf_color}'>{loo_confidence}%</span>**",
        unsafe_allow_html=True
    )
    st.caption(
        "Confidence = mean leave-one-out R² on the 28 training datasets. "
        "This is the honest out-of-sample accuracy, not the training score."
    )
 
    # Out-of-bounds warnings
    if bounds_warnings:
        st.warning("⚠️ Some parameters are outside the training data range:")
        for w in bounds_warnings:
            st.markdown(f"- {w}")
 
    # Parameter display
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown("**Term 1**")
        st.markdown(f"Mu1: `{optimal_params['Mu1']}`")
        st.markdown(f"A1: `{optimal_params['A1']}`")
    with p_col2:
        st.markdown("**Term 2**")
        st.markdown(f"Mu2: `{optimal_params['Mu2']}`")
        st.markdown(f"A2: `{optimal_params['A2']}`")
    with p_col3:
        st.markdown("**Term 3**")
        st.markdown(f"Mu3: `{optimal_params['Mu3']}`")
        st.markdown(f"A3: `{optimal_params['A3']}`")
 
    # Copy-friendly parameter block
    with st.expander("📋 Copy parameters"):
        param_text = "\n".join([f"{k} = {v}" for k, v in optimal_params.items()])
        st.code(param_text, language="text")
 
    # ── Step 4: AI Interpretation ─────────────────────────────────────────────
    st.write("### 🤖 AI Engineering Insights")
    with st.spinner("Consulting Gemma 4 26B..."):
        analysis = get_ai_interpretation(
            optimal_params, fingerprint, loo_confidence, bounds_warnings
        )
    st.markdown(analysis)
 
    # ── Step 5: Force-Deflection Chart ────────────────────────────────────────
    st.write("### 📈 Physical Target Curve")
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp, y=force,
        mode="lines+markers",
        name="Physical test",
        line=dict(color="#7F77DD", width=2.5),
        marker=dict(size=5)
    ))
    fig.update_layout(
        xaxis_title="Displacement (mm)",
        yaxis_title="Force (N)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # ── Step 6: Feature breakdown table ──────────────────────────────────────
    with st.expander("📊 Full feature breakdown"):
        feat_df = pd.DataFrame([{
            "Feature":     k,
            "Value":       v,
            "Description": {
                "peak_force":        "Maximum force reached (N)",
                "initial_stiffness": "Slope at start of loading (N/mm)",
                "energy_absorbed":   "Area under F-D curve (N.mm)",
                "max_displacement":  "Final displacement (mm)",
                "q25_force":         "Force at 25% displacement (N)",
                "mid_force":         "Force at 50% displacement (N)",
                "q75_force":         "Force at 75% displacement (N)",
                "nonlinearity":      "Late slope / early slope ratio",
                "power_exp":         "Power law exponent b in F = a·d^b",
            }.get(k, "")
        } for k, v in fingerprint.items()])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
 