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
    page_icon="⚡",
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

    /* Section Cards */
    div[data-testid="stExpander"], div.stCard {
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
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
    
    /* Divider spacing */
    hr {
        margin: 2.5rem 0;
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
# HERO TITLE HEADER
# ============================================================

st.markdown("""
<div class="hero-header">
    <h1>🚀 AI Income & Car Price Predictor</h1>
    <p>Advanced Machine Learning portal for Demographics Analysis & Vehicle Valuation</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TAB-BASED LAYOUT FOR BETTER NAVIGATION
# ============================================================

tab1, tab2 = st.tabs(["👤 Income Prediction", "🚘 Car Price Prediction"])


# ============================================================
# TAB 1: INCOME SECTION
# ============================================================

with tab1:
    st.markdown("### 📋 Demographic & Work Profile")
    st.caption("Provide the user attributes below to estimate high-income probability.")

    with st.container():
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

    st.markdown("---")
    st.markdown("### 💰 Financial Indicators & Workload")

    col1, col2, col3 = st.columns(3)

    with col1:
        capital_gain = st.number_input(
            "Capital Gain ($)",
            min_value=0,
            value=0
        )

    with col2:
        capital_loss = st.number_input(
            "Capital Loss ($)",
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

    # DISPLAY INCOME RESULT
    if "income_probability" in st.session_state:
        probability = st.session_state["income_probability"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📊 **Prediction Results**")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(
                label="High Income Probability (>50K)",
                value=f"{probability:.2%}"
            )
        with res_col2:
            st.progress(float(probability))
            if probability >= 0.5:
                st.success("High probability of earning >$50K/year based on demographic indicators.")
            else:
                st.warning("Moderate or lower probability of earning >$50K/year based on input profile.")


# ============================================================
# TAB 2: CAR PRICE SECTION
# ============================================================

with tab2:
    st.markdown("### 🚗 Vehicle Specifications")
    st.caption("Select your vehicle attributes to calculate estimated market value.")

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
            "Year of Manufacture",
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
            "Mileage (Kilometers)",
            min_value=0,
            value=80000
        )

        color = st.selectbox(
            "Exterior Color",
            sorted(encoding_maps["color"].keys())
        )

    with col3:
        power_trans = st.selectbox(
            "Transmission Type",
            sorted(encoding_maps["power_trans"].keys())
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(encoding_maps["fuel_type"].keys())
        )

        location = st.selectbox(
            "Location / Region",
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

    # DISPLAY CAR PRICE
    if "predicted_price" in st.session_state:
        predicted_price = st.session_state["predicted_price"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.success("💵 **Valuation Complete**")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(
                label="Estimated Market Price",
                value=f"{predicted_price:,.0f} EGP"
            )
        with res_col2:
            st.caption("Estimation considers current market trends, vehicle condition, and mileage deprecation.")