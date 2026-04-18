import streamlit as st
import pandas as pd
import joblib

# Load the model, scaler, and expected columns
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #4ECDC4;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.1rem; color: #666;">
        This AI-powered tool helps assess heart disease risk based on clinical parameters.<br>
        <strong>⚠️ This is for educational purposes only. Consult a healthcare professional for medical advice.</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown('<h2 class="sidebar-header">📋 Patient Information</h2>', unsafe_allow_html=True)

# Personal Information Section
st.sidebar.markdown("### 👤 Personal Details")
age = st.sidebar.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=50,
    help="Patient's age in years"
)
sex = st.sidebar.selectbox(
    "Sex",
    ["Female", "Male"],
    help="Biological sex of the patient"
)

# Vital Signs Section
st.sidebar.markdown("### 🩺 Vital Signs")
col1, col2 = st.sidebar.columns(2)
with col1:
    resting_bp = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=80,
        max_value=200,
        value=120,
        help="Blood pressure at rest (systolic)"
    )
with col2:
    max_hr = st.number_input(
        "Maximum Heart Rate (bpm)",
        min_value=60,
        max_value=220,
        value=150,
        help="Maximum heart rate achieved"
    )

cholesterol = st.sidebar.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200,
    help="Serum cholesterol level"
)

# Clinical Parameters Section
st.sidebar.markdown("### 🏥 Clinical Parameters")
col3, col4 = st.sidebar.columns(2)
with col3:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Fasting blood sugar level"
    )
with col4:
    oldpeak = st.number_input(
        "Oldpeak (ST depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise relative to rest"
    )

# Categorical Parameters
st.sidebar.markdown("### 📊 Categorical Parameters")
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"],
    format_func=lambda x: {
        "ATA": "Atypical Angina",
        "NAP": "Non-Anginal Pain",
        "TA": "Typical Angina",
        "ASY": "Asymptomatic"
    }[x],
    help="Type of chest pain experienced"
)

resting_ecg = st.sidebar.selectbox(
    "Resting ECG Results",
    ["Normal", "ST", "LVH"],
    format_func=lambda x: {
        "Normal": "Normal",
        "ST": "ST-T Wave Abnormality",
        "LVH": "Left Ventricular Hypertrophy"
    }[x],
    help="Resting electrocardiogram results"
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["N", "Y"],
    format_func=lambda x: "Yes" if x == "Y" else "No",
    help="Angina induced by exercise"
)

st_slope = st.sidebar.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"],
    format_func=lambda x: {
        "Up": "Upsloping",
        "Flat": "Flat",
        "Down": "Downsloping"
    }[x],
    help="Slope of the peak exercise ST segment"
)

# Main content area
st.markdown("---")

# Prediction section
col_pred, col_reset = st.columns([3, 1])
with col_pred:
    predict_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
with col_reset:
    reset_button = st.button("🔄 Reset", use_container_width=True)

if reset_button:
    st.rerun()

if predict_button:
    # Prepare input data
    input_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == "Male" else 0,
        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
        'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == "Y" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Up" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[expected_columns]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Try to get prediction probability if available
    try:
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = max(prediction_proba) * 100
    except:
        confidence = None

    # Display results
    st.markdown("### 📊 Prediction Results")

    if prediction[0] == 1:
        st.markdown(f"""
        <div class="prediction-result" style="background-color: #ffebee; border: 2px solid #f44336; color: #c62828;">
            ⚠️ <strong>HIGH RISK</strong> of heart disease detected!<br>
            Please consult a healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-result" style="background-color: #e8f5e8; border: 2px solid #4caf50; color: #2e7d32;">
            ✅ <strong>LOW RISK</strong> of heart disease detected.<br>
            Continue maintaining a healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)

    if confidence:
        st.markdown(f"**Model Confidence:** {confidence:.1f}%")

    # Show input summary
    with st.expander("📋 Input Summary"):
        summary_data = {
            "Parameter": ["Age", "Sex", "Resting BP", "Cholesterol", "Max HR", "Fasting BS >120", "Oldpeak", "Chest Pain", "Resting ECG", "Exercise Angina", "ST Slope"],
            "Value": [f"{age} years", sex, f"{resting_bp} mmHg", f"{cholesterol} mg/dL", f"{max_hr} bpm",
                     "Yes" if fasting_bs else "No", f"{oldpeak}", chest_pain, resting_ecg,
                     "Yes" if exercise_angina == "Y" else "No", st_slope]
        }
        st.table(pd.DataFrame(summary_data))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🩺 <strong>Disclaimer:</strong> This tool uses machine learning for risk assessment but is not a substitute for professional medical diagnosis.</p>
    <p>Built with ❤️ using Streamlit and KNN Classification</p>
</div>
""", unsafe_allow_html=True)