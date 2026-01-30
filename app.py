# ===============================
# HealthAI Predictor - Final App
# ===============================

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="HealthAI Predictor",
    page_icon="🫀",
    layout="wide"
)

# ----------------------------------
# CSS
# ----------------------------------
st.markdown("""
<style>

body {
    background-color:#020617;
}

.title {
    font-size:38px;
    font-weight:800;
    color:#38bdf8;
}

.subtitle {
    color:#94a3b8;
}

.card {
    background:rgba(255,255,255,0.05);
    padding:25px;
    border-radius:18px;
    box-shadow:0 0 25px rgba(56,189,248,0.15);
}

.good {
    background:#052e16;
    border-left:6px solid #22c55e;
    padding:20px;
    border-radius:12px;
}

.bad {
    background:#450a0a;
    border-left:6px solid #ef4444;
    padding:20px;
    border-radius:12px;
}

.stButton>button {
    background:linear-gradient(90deg,#38bdf8,#0ea5e9);
    color:black;
    font-size:18px;
    font-weight:700;
    border-radius:18px;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load model
# ----------------------------------
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------
# Header
# ----------------------------------
c1, c2 = st.columns([1, 6])

with c1:
    st.image("logo.jpg", width=80)

with c2:
    st.markdown("<div class='title'>HealthAI Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI Powered Medical Risk Analysis System</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------
# Inputs
# ----------------------------------
st.markdown("## 🧾 Patient Health Information")

a, b, c = st.columns(3)

with a:
    age = st.number_input("Age", 1, 100)
    bmi = st.number_input("BMI", 10.0, 60.0)
    bp = st.number_input("Blood Pressure", 70, 220)

with b:
    glucose = st.number_input("Glucose Level", 60, 350)
    heart = st.number_input("Heart Rate", 40, 160)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)

with c:
    stress = st.slider("Stress Level", 1, 10, 5)
    exercise = st.slider("Exercise Hours", 0.0, 5.0, 1.0)
    water = st.slider("Water Intake (Litres)", 0.0, 6.0, 2.0)

smoking = st.selectbox("Smoking", [0, 1])
alcohol = st.selectbox("Alcohol", [0, 1])
medical = st.selectbox("Medical History", [0, 1])

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("📄 Generate Medical Report"):

    X = np.array([[
        age, bmi, bp, 0, glucose, heart,
        sleep, exercise, water, stress,
        smoking, alcohol, 0, 5,
        5, medical, 0,
        0, 0, 0, 0, 0
    ]])

    X = scaler.transform(X)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.markdown("## 🧠 AI Diagnosis Result")

    if pred == 1:
        st.markdown(
            f"<div class='bad'>⚠️ High Health Risk<br><b>Probability:</b> {prob:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='good'>✅ Healthy Individual<br><b>Probability:</b> {prob:.2f}</div>",
            unsafe_allow_html=True
        )

    # ----------------------------------
    # Visual summary
    # ----------------------------------
    st.markdown("## 📊 Health Risk Summary")

    col1, col2 = st.columns(2)

    # ---- Donut chart ----
    with col1:
        st.markdown("### 🧬 Risk Distribution")

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.pie(
            [prob, 1 - prob],
            labels=["High Risk", "Healthy"],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops=dict(width=0.4)
        )
        ax.set_aspect("equal")
        st.pyplot(fig)

    # ---- Feature importance ----
    with col2:
        st.markdown("### 🔍 Top Risk Factors")

        fi = pd.DataFrame({
            "Feature": ["BMI", "Age", "Blood Pressure", "Glucose", "Stress"],
            "Importance": sorted(model.feature_importances_, reverse=True)[:5]
        })

        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.barh(fi["Feature"], fi["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

    # ----------------------------------
    # Final report
    # ----------------------------------
    st.markdown("## 🧾 Final Medical Assessment")

    risk_level = (
        "LOW RISK" if prob < 0.4 else
        "MODERATE RISK" if prob < 0.7 else
        "HIGH RISK"
    )

    st.success(f"Overall Risk Category: {risk_level}")

    st.markdown("""
    ✔ **Model:** Random Forest Classifier  
    ✔ **Learning Type:** Supervised Machine Learning  
    ✔ **Dataset:** Clinical + Lifestyle Health Records  
    ✔ **Output:** Individual Health Risk Score  
    """)
