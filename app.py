import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# ==========================
# MODEL ARCHITECTURE
# ==========================

class ChurnModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# ==========================
# LOAD MODEL & SCALER
# ==========================

model = ChurnModel(29)

model.load_state_dict(
    torch.load(
        "model/customer_churn_model2.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

scaler = joblib.load("model/scaler2.pkl")

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn.")

# ==========================
# CUSTOMER INFO
# ==========================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

# ==========================
# PHONE SERVICES
# ==========================

st.header("Phone Services")

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes"]
)

# ==========================
# INTERNET SERVICES
# ==========================

st.header("Internet Services")

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber Optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No Internet Service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No Internet Service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No Internet Service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No Internet Service"]
)

# ==========================
# STREAMING SERVICES
# ==========================

st.header("Streaming Services")

stream_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No Internet Service"]
)

stream_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No Internet Service"]
)

# ==========================
# BILLING
# ==========================

st.header("Billing Information")

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank Transfer",
        "Credit Card",
        "Electronic Check",
        "Mailed Check"
    ]
)

# ==========================
# PREDICTION
# ==========================

if st.button("Predict Churn", use_container_width=True):

    features = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,

        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,

        'OnlineSecurity_No internet service': 0,
        'OnlineSecurity_Yes': 0,

        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,

        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,

        'TechSupport_No internet service': 0,
        'TechSupport_Yes': 0,

        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,

        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,

        'Contract_One year': 0,
        'Contract_Two year': 0,

        'PaperlessBilling_Yes': 0,

        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0,
    }

    # Internet Service
    if internet_service == "Fiber Optic":
        features['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        features['InternetService_No'] = 1

    # Online Security
    if online_security == "Yes":
        features['OnlineSecurity_Yes'] = 1
    elif online_security == "No Internet Service":
        features['OnlineSecurity_No internet service'] = 1

    # Online Backup
    if online_backup == "Yes":
        features['OnlineBackup_Yes'] = 1
    elif online_backup == "No Internet Service":
        features['OnlineBackup_No internet service'] = 1

    # Device Protection
    if device_protection == "Yes":
        features['DeviceProtection_Yes'] = 1
    elif device_protection == "No Internet Service":
        features['DeviceProtection_No internet service'] = 1

    # Tech Support
    if tech_support == "Yes":
        features['TechSupport_Yes'] = 1
    elif tech_support == "No Internet Service":
        features['TechSupport_No internet service'] = 1

    # Streaming TV
    if stream_tv == "Yes":
        features['StreamingTV_Yes'] = 1
    elif stream_tv == "No Internet Service":
        features['StreamingTV_No internet service'] = 1

    # Streaming Movies
    if stream_movies == "Yes":
        features['StreamingMovies_Yes'] = 1
    elif stream_movies == "No Internet Service":
        features['StreamingMovies_No internet service'] = 1

    # Contract
    if contract == "One year":
        features['Contract_One year'] = 1
    elif contract == "Two year":
        features['Contract_Two year'] = 1

    # Paperless Billing
    if paperless == "Yes":
        features['PaperlessBilling_Yes'] = 1

    # Payment Method
    if payment == "Credit Card":
        features['PaymentMethod_Credit card (automatic)'] = 1
    elif payment == "Electronic Check":
        features['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed Check":
        features['PaymentMethod_Mailed check'] = 1

    input_df = pd.DataFrame([features])

    scaled = scaler.transform(input_df)

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits).item()

    st.divider()

    if probability > 0.6:
        st.error(
            f"⚠️ High Churn Risk ({probability:.2%})"
        )

    elif probability > 0.3:
        st.warning(
            f"🟠 Medium Churn Risk ({probability:.2%})"
        )

    else:
        st.success(
            f"🟢 Low Churn Risk ({probability:.2%})"
        )

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )