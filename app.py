import streamlit as st
import numpy as np
import joblib
import time

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0B0F19;
}

.big-title {
    font-size: 44px;
    font-weight: 800;
    color: #00E5FF;
}

.subtitle {
    font-size: 18px;
    color: #94A3B8;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #38BDF8;
    margin-top: 30px;
}

.input-card {
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(145deg, #111827, #1F2937);
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}

.result-card {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1E293B, #0F172A);
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    margin-top: 30px;
}

.stNumberInput input {
    font-size: 18px !important;
    height: 45px !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="big-title">🚦 Smart City Traffic Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Urban Traffic Pattern Segmentation using K-Means Clustering</div>', unsafe_allow_html=True)
st.divider()

# -------------------------
# INPUT SECTION
# -------------------------
st.markdown('<div class="section-title">📊 Traffic & Environmental Inputs</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    traffic_volume = st.number_input("🚗 Traffic Volume (Vehicles Count)", value=3000)
    temp = st.number_input("🌡 Temperature (°C)", value=15.0)
    rain_1h = st.number_input("🌧 Rainfall in Last 1 Hour (mm)", value=0.0)

with col2:
    snow_1h = st.number_input("❄ Snowfall in Last 1 Hour (mm)", value=0.0)
    clouds_all = st.number_input("☁ Cloud Coverage (%)", value=40)
    hour = st.number_input("⏰ Hour of Day (0–23)", min_value=0, max_value=23, value=8)

with col3:
    day = st.number_input("📅 Day of Month", min_value=1, max_value=31, value=15)
    month = st.number_input("🗓 Month", min_value=1, max_value=12, value=10)
    day_of_week = st.number_input("📆 Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button("🚀 Analyze Traffic Pattern", use_container_width=True):

    input_data = np.array([[traffic_volume, temp, rain_1h, snow_1h,
                            clouds_all, hour, day, month, day_of_week]])

    scaled = scaler.transform(input_data)
    pca_data = pca.transform(scaled)
    cluster = model.predict(pca_data)[0]

    # Traffic Intensity Logic
    if traffic_volume > 5000:
        traffic_label = "High Traffic / Congested"
        intensity = 90
        color = "#FF4B4B"
    elif traffic_volume < 1500:
        traffic_label = "Low Traffic / Free Flow"
        intensity = 25
        color = "#22C55E"
    else:
        traffic_label = "Moderate Traffic"
        intensity = 55
        color = "#FACC15"

    # -------------------------
    # RESULT CARD
    # -------------------------
    st.markdown(f"""
    <div class="result-card">
        <h2 style="color:{color}; font-size:32px;">Predicted Cluster: {cluster}</h2>
        <h3 style="color:{color};">{traffic_label}</h3>
        <p style="color:#CBD5E1;">
        This cluster represents a traffic behavior pattern identified using unsupervised learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚦 Traffic Intensity Indicator")

    progress = st.progress(0)

    for i in range(intensity):
        progress.progress(i + 1)
        time.sleep(0.01)

    st.write(f"Traffic Intensity Level: {intensity}%")