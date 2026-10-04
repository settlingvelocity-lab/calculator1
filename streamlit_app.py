# -*- coding: utf-8 -*-
"""
Settling-velocity predictor for graded sandy sediments (Streamlit).

Calculation order
  1) Single-particle settling velocity: Wu and Wang (2006) at D50
  2) Concentration correction: Richardson-Zaki with Re-dependent index n
     (Re computed from the single-particle velocity)
  3) ML residual correction: Gradient Boosting (GB) on the log residual
     w = w_conc * exp(r_hat)

Files: streamlit_app.py, physics_wuwang.py, gb_logresid_model.joblib, requirements.txt
"""
import numpy as np
import joblib
import streamlit as st
from physics_wuwang import wu_wang_velocity, concentration_factor

st.set_page_config(page_title="Settling-Velocity Predictor", page_icon="🌊", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("gb_logresid_model.joblib")


BUNDLE = load_model()
MODEL = BUNDLE["model"]
FEAT = BUNDLE["feat"]          # ['CSF', 'Cu', 'D10', 'D50', 'nu', 'c']
RANGES = BUNDLE["ranges"]

LABEL = {
    "Cu": ("C\u1d64", "-"),
    "D10": ("D\u2081\u2080", "mm"),
    "D50": ("D\u2085\u2080", "mm"),
    "CSF": ("CSF", "-"),
    "nu": ("\u03bd", "m\u00b2/s"),
    "c": ("c", "-"),
}

st.markdown("""
<style>
/* input labels: large, bold, dark */
[data-testid="stWidgetLabel"] p { font-size:1.05rem !important; font-weight:600 !important; color:#0b1f2a !important; }
[data-testid="stWidgetLabel"] .katex { font-size:1.12em !important; }
/* helper notes under inputs: small, light grey */
.def { color:#7a8791; font-size:0.80rem; line-height:1.35; margin:-4px 0 14px 2px; }
/* section titles inside the input panel */
.sec { color:#0b6ea8; font-weight:700; font-size:1.08rem; margin:14px 0 6px 0;
       padding-left:8px; border-left:4px solid #38b6ff; }
/* predict button */
.stButton>button { background:#38b6ff; color:#fff; border:none; border-radius:8px; font-weight:600; font-size:1.0rem; }
.stButton>button:hover { background:#1aa1f0; color:#fff; }
/* result cards */
[data-testid="stMetric"] { background:#f4f9fd; border:1px solid #d6e8f5; border-radius:10px; padding:12px 14px; }
[data-testid="stMetricLabel"] p { font-size:0.95rem !important; color:#3f4b54 !important; }
[data-testid="stMetricValue"] { color:#0b6ea8; }
/* training-range panel */
.panel { background:#f4f9fd; border:1px solid #d6e8f5; border-radius:10px; padding:10px 12px; }
.rng { border-collapse:collapse; font-size:0.95rem; width:100%; }
.rng th { text-align:left; color:#3f4b54; font-weight:600; padding:4px 8px; border-bottom:2px solid #d6e8f5; }
.rng td { padding:4px 8px; border-bottom:1px solid #e3edf5; }
.out { color:#c0392b; font-weight:700; }
</style>
""", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='def'>{text}</div>", unsafe_allow_html=True)


def fmt(v):
    return f"{v:.3e}" if (abs(v) < 1e-3 and v != 0) else f"{v:g}"


st.title("Settling-Velocity Predictor")
st.caption("Hybrid physics and machine learning model for graded sandy sediments: "
           "Wu and Wang (2006) single-particle velocity, Richardson-Zaki concentration "
           "correction, and Gradient Boosting residual correction")

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Inputs")

    st.markdown("<div class='sec'>Particle properties</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        D10 = st.number_input(r"$D_{10}$ (mm)", min_value=0.001, max_value=100.0, value=0.62, step=0.01, format="%.3f")
        note("Grain size at 10% passing by mass.")
        D50 = st.number_input(r"$D_{50}$ (mm)", min_value=0.001, max_value=100.0, value=0.72, step=0.01, format="%.3f")
        note("Median grain size (50% passing by mass).")
        Cu = st.number_input(r"$C_u$ (-)", min_value=1.0, max_value=30.0, value=1.19, step=0.01, format="%.2f")
        note("Uniformity coefficient, D\u2086\u2080/D\u2081\u2080. Larger values mean a wider PSD.")
    with c2:
        CSF = st.number_input("CSF (-)", min_value=0.1, max_value=1.0, value=0.67, step=0.01, format="%.2f")
        note("Corey shape factor, c/\u221a(a b). 1 = sphere; natural sand is about 0.7.")
        rho_s = st.number_input(r"$\rho_s$ (kg/m³)", min_value=1000.0, max_value=8000.0, value=2650.0, step=10.0)
        note("Particle density. Quartz sand is about 2650 kg/m\u00b3; glass beads about 2430 kg/m\u00b3.")

    st.markdown("<div class='sec'>Fluid properties</div>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        rho_f = st.number_input(r"$\rho_w$ (kg/m³)", min_value=500.0, max_value=2000.0, value=998.2, step=0.1)
        note("Fluid density. Water at 20 \u00b0C is 998.2 kg/m\u00b3.")
    with f2:
        nu = st.number_input(r"$\nu$ (m²/s)", min_value=1.0e-7, max_value=1.0e-2, value=1.004e-6,
                             step=1.0e-7, format="%.3e")
        note("Kinematic viscosity. Water at 20 \u00b0C is 1.004e-06 m\u00b2/s.")

    st.markdown("<div class='sec'>Concentration</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    mass_g = m1.number_input("Sediment mass (g)", min_value=0.0, max_value=1.0e6, value=5.0, step=0.1)
    vol_L = m2.number_input("Total volume (L)", min_value=0.001, max_value=1.0e4, value=1.0, step=0.1)
    Cm = mass_g / vol_L                      # g/L = kg/m3
    c = Cm / rho_s                           # volumetric concentration
    note(f"Mass concentration = mass / total volume = <b>{Cm:.2f} kg/m\u00b3</b>; "
         f"volumetric concentration c = mass concentration / \u03c1\u209b = <b>{c:.5f}</b>.")

    go = st.button("Predict", use_container_width=True)

current = {"CSF": CSF, "Cu": Cu, "D10": D10, "D50": D50, "nu": nu, "c": c}
out_vars = [k for k in FEAT if not (RANGES[k][0] <= current[k] <= RANGES[k][1])]

with right:
    st.subheader("Training range for ML residual correction")
    st.caption(f"Range of the input variables used to train the GB residual model "
               f"(n = {BUNDLE.get('n_train', 66)} specimens).")
    rows = ""
    for k in ["Cu", "D10", "D50", "CSF", "nu", "c"]:
        lo, hi = RANGES[k]
        sym, unit = LABEL[k]
        cls = "out" if k in out_vars else ""
        rows += (f"<tr><td><i>{sym}</i> ({unit})</td><td>{fmt(lo)} to {fmt(hi)}</td>"
                 f"<td class='{cls}'>{fmt(current[k])}</td></tr>")
    st.markdown(f"<div class='panel'><table class='rng'><tr><th>Variable</th><th>Training range</th><th>Input</th></tr>"
                f"{rows}</table></div>", unsafe_allow_html=True)
    st.caption("Only water (1.004e-06 m\u00b2/s) and glycerin (1.12e-03 m\u00b2/s) were tested, "
               "so \u03bd values between them have not been tested experimentally. "
               r"$\rho_s$ and $\rho_w$ are used only in the physics-based calculation.")

st.divider()

if go:
    try:
        # 1) single particle (Wu and Wang, 2006) at D50
        w_single = wu_wang_velocity(D50, CSF, rho_s, rho_f, nu)
        # 2) concentration correction (Re from the single-particle velocity)
        Re, n_rz, R_fac = concentration_factor(w_single, D50, nu, c)
        w_conc = w_single * R_fac
        # 3) GB residual correction
        x = np.array([[current[k] for k in FEAT]])
        r_hat = float(MODEL.predict(x)[0])
        w_final = w_conc * np.exp(r_hat)

        st.subheader("Results")
        a, b, cc = st.columns(3)
        a.metric("Single-particle velocity", f"{w_single:.3f} cm/s")
        b.metric("After concentration correction", f"{w_conc:.3f} cm/s")
        cc.metric("Group settling velocity (GB corrected)", f"{w_final:.3f} cm/s")

        st.markdown("**Calculation path**")
        st.markdown(
            f"1. Wu and Wang (2006) single-particle velocity at D\u2085\u2080 = {D50:.3f} mm: **{w_single:.3f} cm/s**\n"
            f"2. Concentration correction (Re = {Re:.3g}, n = {n_rz:.2f}): (1 - c)\u207f = {R_fac:.4f}, "
            f"giving **{w_conc:.3f} cm/s**\n"
            f"3. GB residual correction: ln-residual = {r_hat:+.3f}, w = w_conc exp(r) = **{w_final:.3f} cm/s**")

        if out_vars:
            names = ", ".join(LABEL[k][0] for k in out_vars)
            st.warning("The input is outside the training range of the ML residual correction model. "
                       "The prediction therefore involves extrapolation. "
                       f"(Out of range: {names})")
        else:
            st.success("All ML input variables are within the training range.")
    except Exception as e:
        st.error(f"The prediction could not be completed: {e}")
else:
    st.info("Enter the inputs and select Predict.")

st.caption("Model: Gradient Boosting residual correction trained on 66 specimens "
           "(repeated 5-fold cross-validation MAPE = 6.2%).")
