import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="AI Diabetes Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2ff, #fdf2f8);
}

.main-title {
    text-align: center;
    font-size: 58px;
    font-weight: bold;
    color: #4f46e5;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #64748b;
    margin-bottom: 40px;
}

.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

.stButton>button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    background: linear-gradient(to right, #6366f1, #ec4899);
    color: white;
    font-size: 22px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}

.result-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🩺 AI Diabetes Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Smart Healthcare Analytics Dashboard</div>',
    unsafe_allow_html=True
)

# Main card
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)

    glucose = st.number_input("Glucose Level", 0, 300, 120)

    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)

    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin Level", 0, 900, 80)

    bmi = st.number_input("BMI", 0.0, 70.0, 25.5)

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        0.0,
        3.0,
        0.5
    )

    age = st.number_input("Age", 1, 120, 30)

st.write("")

# Prediction button
if st.button("🔍 Analyze Diabetes Risk"):

    input_data = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree,
        'Age': age
    }])

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    risk = probability[0][1] * 100

    st.write("")

    # Result UI
    if prediction[0] == 1:
        st.markdown(
            f'''
            <div class="result-box"
            style="background:#fee2e2;color:#b91c1c;">
            ⚠️ HIGH DIABETES RISK<br>
            {risk:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''
            <div class="result-box"
            style="background:#dcfce7;color:#166534;">
            ✅ LOW DIABETES RISK<br>
            {risk:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.write("")

    # Progress
    st.progress(int(risk))

    # Charts Section
    st.subheader("📊 Health Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        labels = ['Risk', 'Safe']
        values = [risk, 100-risk]

        fig1, ax1 = plt.subplots()

        ax1.pie(
            values,
            labels=labels,
            autopct='%1.1f%%'
        )

        ax1.set_title("Diabetes Risk Distribution")

        st.pyplot(fig1)

    with chart_col2:

        features = ['Glucose', 'BMI', 'BloodPressure', 'Age']

        values = [
            glucose,
            bmi,
            blood_pressure,
            age
        ]

        fig2, ax2 = plt.subplots()

        ax2.bar(features, values)

        ax2.set_title("Health Parameters")

        st.pyplot(fig2)

    # Insights
    st.subheader("🧠 AI Health Insights")

    if glucose > 140:
        st.warning("High glucose level detected")

    if bmi > 30:
        st.warning("BMI indicates obesity risk")

    if blood_pressure > 90:
        st.warning("High blood pressure detected")

    if age > 45:
        st.info("Age increases diabetes probability")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("")
st.caption("Powered by Streamlit + Machine Learning + AI Analytics")