import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import base64
import os
from datetime import datetime
import random

# Read the background image
image_path = os.path.join(os.path.dirname(__file__), "img/heart 2.jpg")
with open(image_path, "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Load model and scaler with verification
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/heart_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_models()
if model is None or scaler is None:
    st.stop()

# Set background and style with blurred effect
st.markdown(
    f"""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/jpg;base64,{img_data}");
        background-size: cover;
        background-position: center;
        filter: blur(8px);
        z-index: -1;
    }}
    .stApp > div {{
        background-color: rgba(0, 0, 0, 0.6);
    }}
    
    /* Heart icon styling */
    .header-icon {{
        display: inline-block;
        margin-right: 10px;
        font-size: 1.3em;
        vertical-align: middle;
        animation: heartbeat 1.5s infinite;
        filter: drop-shadow(0 0 2px rgba(255,77,109,0.5));
    }}
    
    @keyframes heartbeat {{
        0% {{ transform: scale(1); }}
        25% {{ transform: scale(1.1); }}
        50% {{ transform: scale(1); }}
        75% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); }}
    }}
    
    /* Rest of your existing styles */
    .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stRadio, .stSlider {{
        color: #FFFFFF !important;
        text-shadow: 1px 1px 2px #000000;
    }}
    .card {{
        background-color: rgba(20, 20, 20, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.7);
        margin-bottom: 20px;
        color: #FFFFFF;
    }}
    .pulse {{
        width: 30px;
        height: 30px;
        background: #ff4d6d;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin: auto;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.5); opacity: 0.6; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    </style>
    
    <div class="header">
        <h1><span class="header-icon"><i class="fa fa-heartbeat" style="color:#ff4d6d;"></i></span> CardioVision AI</h1>
        <h3>Advanced Heart Disease Risk Prediction</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Improved prediction function
def predict_risk():
    sex_val = 1 if sex == "Male" else 0
    smoker_val = 1 if smoker == "Yes" else 0
    features = np.array([[age, sex_val, 0, bp, chol, 0, 0, hr, smoker_val, 0.0, 0, 0, 1]])
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame(features, columns=columns)
    
    try:
        scaled = scaler.transform(input_df)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled)[0][1] * 100
        else:
            prob = model.predict(scaled)[0] * 100
        return prob
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0

# Sidebar Inputs
with st.sidebar:
    st.header("Patient Information")
    name = st.text_input("Full Name")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 45)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    st.subheader("Health Metrics")
    bp = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    hr = st.slider("Resting Heart Rate (bpm)", 40, 120, 72)

    st.subheader("Lifestyle")
    smoker = st.radio("Smoker", ["Yes", "No"], horizontal=True)
    weight = st.number_input("Weight (kg)", 40, 200, 70)
    height = st.number_input("Height (cm)", 140, 220, 170)
    bmi = weight / ((height / 100) ** 2)
    st.metric("BMI", f"{bmi:.1f}", "Healthy" if 18.5 <= bmi <= 24.9 else "Attention")

# Tabs Layout
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Dashboard", "Tips"])

with tab1:
    st.header("Cardiovascular Risk Estimation")
    st.markdown("<div class='pulse'></div>", unsafe_allow_html=True)

    if st.button("Assess My Risk", use_container_width=True):
        with st.spinner('Analyzing your profile...'):
            time.sleep(1.5)
            risk = predict_risk()
            st.session_state.risk = risk

            st.markdown(f"""
            <div class="card">
                <h3 style="text-align:center;">Your 10-Year Cardiovascular Risk</h3>
                <h1 style="text-align:center; color:#FF9800;">{risk:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)

            # Adjusted risk thresholds
            if risk > 55:
                st.error("High Risk - Immediate consultation recommended!")
                with st.expander("Recommendations"):
                    st.markdown("""
                    - Consult a cardiologist immediately
                    - Strictly control blood pressure and cholesterol
                    - Adopt a heart-healthy diet
                    - Engage in supervised physical activity
                    """)
            elif risk > 20:
                st.warning("Moderate Risk - Lifestyle changes needed.")
                with st.expander("Recommendations"):
                    st.markdown("""
                    - Improve diet (more fruits, vegetables)
                    - Exercise at least 150 minutes/week
                    - Quit smoking
                    - Manage stress
                    """)
            else:
                st.success("Low Risk - Keep maintaining good habits!")
                with st.expander("Recommendations"):
                    st.markdown("""
                    - Maintain healthy lifestyle
                    - Stay active and hydrated
                    - Annual health checkups
                    """)

with tab2:
    st.header("Health Dashboard")
    
    if 'risk' in st.session_state:
        # Create radar chart
        categories = ['Age', 'BP', 'Cholesterol', 'Heart Rate', 'BMI']
        actual_values = [age/100, bp/200, chol/600, hr/150, bmi/40]
        ideal_values = [0.5, 120/200, 180/600, 72/150, 22.5/40]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        actual_values += actual_values[:1]
        ideal_values += ideal_values[:1]
        
        ax.plot(angles, actual_values, 'o-', linewidth=2, color='#38bdf8', label='Your Values')
        ax.fill(angles, actual_values, color='#38bdf8', alpha=0.25)
        ax.plot(angles, ideal_values, 'o-', linewidth=2, color='#4ECDC4', label='Ideal Values')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='white', fontsize=12)
        ax.set_rlabel_position(30)
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color='white')
        ax.set_title("Health Parameters vs Ideal Values", color='white', pad=20)
        ax.grid(color='#334155')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        st.pyplot(fig)
        
        # Health metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{st.session_state.risk:.1f}%")
        with col2:
            st.metric("Blood Pressure", f"{bp} mmHg", "High" if bp > 130 else "Normal")
        with col3:
            st.metric("Cholesterol", f"{chol} mg/dl", "High" if chol > 200 else "Normal")
    else:
        st.info("Please complete the risk assessment first to view dashboard")

with tab3:
    st.header("Health Recommendations")
    with st.expander("Daily Habits"):
        st.success("• Drink 2-3 liters of water daily")
        st.success("• Get 7-9 hours of quality sleep")
    with st.expander("Nutrition"):
        st.success("• Eat plenty of fruits and vegetables")
        st.success("• Limit processed foods and sugars")
    with st.expander("Activity"):
        st.success("• 150+ minutes of exercise weekly")
        st.success("• Include strength training")

st.caption("Copyright 2026 CardioVision AI. All Rights Reserved.")