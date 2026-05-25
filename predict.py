import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input data with column names
data = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 20,
    'Insulin': 80,
    'BMI': 25.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 30
}])

# Scale input
scaled_data = scaler.transform(data)

# Prediction
prediction = model.predict(scaled_data)

# Probability
probability = model.predict_proba(scaled_data)

# Output
if prediction[0] == 1:
    print("Person may have diabetes")
else:
    print("Person is non-diabetic")

print(f"Diabetes Risk: {probability[0][1] * 100:.2f}%")