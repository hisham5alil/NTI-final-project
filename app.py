import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import pickle

from xgboost import XGBClassifier, XGBRegressor

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Income & Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================

st.markdown("""
<style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Global font settings */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header gradient banner */
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .hero-header h1 {
        color: #ffffff !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-header p {
        color: #e0e6ed !important;
        font-size: 1.1rem;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    
    /* Primary Action Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    /* Section Container Styling */
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD INCOME MODEL
# ============================================================

@st.cache_resource
def load_income_model():

    model = XGBClassifier()
    model.load_model("model.json")

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("label_encoders.pkl", "rb") as file:
        label_encoders = pickle.load(file)

    return model, scaler, label_encoders


# ============================================================
# LOAD CAR PRICE MODEL
# ============================================================

@st.cache_resource
def load_car_model():

    model = XGBRegressor()
    model.load_model("car_price_model.json")

    with open("car_price_encoder.pkl", "rb") as file:
        artifacts = pickle.load(file)

    return model, artifacts


income_model, scaler, label_encoders = load_income_model()
car_model, car_artifacts = load_car_model()

encoding_maps = car_artifacts["encoding_maps"]
global_mean = car_artifacts["global_mean"]
categorical_cols = car_artifacts["categorical_cols"]


# ============================================================
# INCOME PREDICTION
# ============================================================

def predict_income(user_data):

    X_user = pd.DataFrame([user_data])

    # Encode categorical columns
    for col, le in label_encoders.items():

        X_user[col] = le.transform(
            X_user[col].astype(str)
        )

    # Keep the same column order used during training
    X_user = X_user[
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country"
        ]
    ]

    # Scaling
    X_user = pd.DataFrame(
        scaler.transform(X_user),
        columns=X_user.columns
    )

    # Probability
    probability = income_model.predict_proba(X_user)[0][1]

    return probability


# ============================================================
# CAR PRICE PREDICTION
# ============================================================

def predict_car_price(car_data):

    X_car = pd.DataFrame([car_data])

    # Target encoding
    for col in categorical_cols:

        X_car[col] = (
            X_car[col]
            .map(encoding_maps[col])
            .fillna(global_mean)
        )

    # Make sure all columns are numeric
    X_car = X_car.astype(float)

    # Prediction
    predicted_price = car_model.predict(X_car)[0]

    return predicted_price


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-header">
    <h1>🚀 AI Income & Car Price Predictor</h1>
    <p>Predict income probability first, then estimate matching car market price using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 1: INCOME PREDICTION
# ============================================================

st.markdown("### 👤 Step 1: Income Prediction")
st.caption("Enter demographic and financial details to calculate income probability.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=17,
        max_value=100,
        value=35
    )

    workclass = st.selectbox(
        "Workclass",
        label_encoders["workclass"].classes_
    )

    education = st.selectbox(
        "Education",
        label_encoders["education"].classes_
    )

    education_num = st.number_input(
        "Education Number",
        min_value=1,
        max_value=16,
        value=13
    )

with col2:
    marital_status = st.selectbox(
        "Marital Status",
        label_encoders["marital-status"].classes_
    )

    occupation = st.selectbox(
        "Occupation",
        label_encoders["occupation"].classes_
    )

    relationship = st.selectbox(
        "Relationship",
        label_encoders["relationship"].classes_
    )

    race = st.selectbox(
        "Race",
        label_encoders["race"].classes_
    )

with col3:
    sex = st.selectbox(
        "Sex",
        label_encoders["sex"].classes_
    )

    native_country = st.selectbox(
        "Native Country",
        label_encoders["native-country"].classes_
    )

    fnlwgt = st.number_input(
        "Final Weight",
        min_value=0,
        value=180000
    )

st.markdown("#### 💰 Financial Indicators & Hours")

col1, col2, col3 = st.columns(3)

with col1:
    capital_gain = st.number_input(
        "Capital Gain",
        min_value=0,
        value=0
    )

with col2:
    capital_loss = st.number_input(
        "Capital Loss",
        min_value=0,
        value=0
    )

with col3:
    hours_per_week = st.number_input(
        "Hours per Week",
        min_value=1,
        max_value=100,
        value=40
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "Calculate Income Probability",
    type="primary",
    use_container_width=True
):
    user_data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    probability = predict_income(user_data)
    st.session_state["income_probability"] = probability


# ============================================================
# DISPLAY INCOME RESULT
# ============================================================

if "income_probability" in st.session_state:

    probability = st.session_state["income_probability"]

    st.markdown("---")
    st.subheader("📊 Calculated Income Probability")

    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(
            "Income Probability",
            f"{probability:.2%}"
        )

    with res_col2:
        st.progress(float(probability))
        if probability >= 0.5:
            st.success("High income probability detected (>50K/year). Proceed to check higher-tier cars below.")
        else:
            st.info("Moderate income probability detected. Proceed to check matching car price options below.")


# ============================================================
# SECTION 2: CAR PRICE PREDICTION
# ============================================================

st.markdown("---")

st.markdown("### 🚘 Step 2: Car Price Prediction")
st.caption("Select the specifications of the car you wish to evaluate based on the income profile above.")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox(
        "Brand",
        sorted(encoding_maps["brand"].keys())
    )

    model_name = st.selectbox(
        "Model",
        sorted(encoding_maps["model"].keys())
    )

    year = st.number_input(
        "Year",
        min_value=1990,
        max_value=2026,
        value=2020
    )

with col2:
    condition = st.selectbox(
        "Condition",
        sorted(encoding_maps["condition"].keys())
    )

    kilo_meter = st.number_input(
        "Kilometers",
        min_value=0,
        value=80000
    )

    color = st.selectbox(
        "Color",
        sorted(encoding_maps["color"].keys())
    )

with col3:
    power_trans = st.selectbox(
        "Transmission",
        sorted(encoding_maps["power_trans"].keys())
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        sorted(encoding_maps["fuel_type"].keys())
    )

    location = st.selectbox(
        "Location",
        sorted(encoding_maps["location"].keys())
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "Predict Car Price",
    type="primary",
    use_container_width=True
):
    car_data = {
        "brand": brand,
        "model": model_name,
        "year": year,
        "condition": condition,
        "kilo_meter": kilo_meter,
        "color": color,
        "power_trans": power_trans,
        "fuel_type": fuel_type,
        "location": location
    }

    predicted_price = predict_car_price(car_data)
    st.session_state["predicted_price"] = predicted_price


# ============================================================
# DISPLAY CAR PRICE RESULT
# ============================================================

if "predicted_price" in st.session_state:

    predicted_price = st.session_state["predicted_price"]

    st.markdown("---")
    st.subheader("💵 Estimated Car Price")

    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(
            "Predicted Price",
            f"{predicted_price:,.0f} EGP"
        )
    
    with res_col2:
        st.success("The estimated car price has been calculated successfully based on market specifications.")