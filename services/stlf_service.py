import streamlit as st
import pandas as pd
from database.mongo import stlf_col
from services.hf_model_loader import load_stlf_model
from pymongo.errors import ServerSelectionTimeoutError, AutoReconnect
import warnings

# Silence sklearn / TF warnings
warnings.filterwarnings("ignore")

# ===============================
# LOAD MODEL (ONCE & SAFE)
# ===============================
@st.cache_resource
def get_model():
    try:
        return load_stlf_model()
    except Exception as e:
        st.error("❌ Failed to load STLF model")
        st.exception(e)
        st.stop()

model = get_model()

FEATURES = [
    'hour', 'dayofweek', 'month', 'is_weekend',
    'lag_1', 'lag_2', 'lag_24', 'roll24_mean'
]

# ===============================
# STREAMLIT TAB
# ===============================
def stlf_tab():
    st.header("⏱ Short-Term Load Forecast (STLF)")

    # --------------------------------
    # SECTION 1: MANUAL INPUT (FORM)
    # --------------------------------
    st.subheader("🧮 Manual Prediction (Form Input)")

    with st.form("stlf_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            hour = st.number_input("Hour (0–23)", 0, 23, 12)
            dayofweek = st.number_input("Day of Week (0=Sun)", 0, 6, 1)

        with col2:
            month = st.number_input("Month (1–12)", 1, 12, 1)
            is_weekend = st.selectbox("Is Weekend?", [0, 1])

        with col3:
            lag_1 = st.number_input("Lag 1 Load", value=100.0)
            lag_2 = st.number_input("Lag 2 Load", value=98.0)

        with col4:
            lag_24 = st.number_input("Lag 24 Load", value=95.0)
            roll24_mean = st.number_input("Rolling 24 Mean", value=97.0)

        submitted = st.form_submit_button("⏳ Predict Load")

    if submitted:
        input_df = pd.DataFrame([[
            hour, dayofweek, month, is_weekend,
            lag_1, lag_2, lag_24, roll24_mean
        ]], columns=FEATURES)

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🔮 Predicted Load: **{round(float(prediction), 2)}**")
        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)

    st.divider()

    # --------------------------------
    # SECTION 2: DATABASE BASED STLF
    # --------------------------------
    st.subheader("📂 STLF Using Database Data")

    try:
        data = list(
            stlf_col.find({}, {"_id": 0}).limit(200)
        )
    except (ServerSelectionTimeoutError, AutoReconnect):
        st.warning("🔄 Database connection dropped, retrying later")
        return

    if not data:
        st.info("No STLF data found in database")
        st.info("Upload data from 📂 Data Upload tab")
        return

    df = pd.DataFrame(data)
    st.dataframe(df.head())

    if st.button("📈 Predict Using DB Records"):
        try:
            X = df[FEATURES]
            df["STLF_Prediction"] = model.predict(X)

            st.success("Prediction completed for database records")
            st.dataframe(df)
        except Exception as e:
            st.error("❌ Batch prediction failed")
            st.exception(e)
