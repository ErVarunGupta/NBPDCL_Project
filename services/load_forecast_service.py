import streamlit as st
import pandas as pd
import numpy as np
from database.mongo import load_col
from services.hf_model_loader import load_load_forecast_model
from sklearn.preprocessing import MinMaxScaler
from pymongo.errors import NetworkTimeout, AutoReconnect

# ------------------ MODEL LOAD ------------------

@st.cache_resource
def get_model():
    return load_load_forecast_model()

model = get_model()

# ------------------ HELPER FUNCTION ------------------

def create_sequences(data, time_steps=24):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)

# ------------------ STREAMLIT TAB ------------------

def load_forecast_tab():
    st.header("📈 Load Forecast (LSTM – Short Term)")

    # Load data from MongoDB
    try:
        data = list(
            load_col.find({}, {"_id": 0}).limit(200)
        )
    except (NetworkTimeout, AutoReconnect):
        st.warning("🔄 Database timeout. Please retry.")
        return
    if not data:
        st.warning("No data available in database")
        return

    df = pd.DataFrame(data)
    st.subheader("Input Data Preview")
    st.dataframe(df.head())

    # -------- SELECT LOAD COLUMN --------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.error("No numeric load column found")
        return

    load_col_name = st.selectbox(
        "Select Load Column for Forecasting",
        numeric_cols
    )

    # -------- FORECAST BUTTON --------
    if st.button("⚡ Forecast Load (LSTM)"):

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[[load_col_name]])

        # Sequence creation
        time_steps = 24   # short-term (24 hours)
        X = create_sequences(scaled_data, time_steps)

        if len(X) == 0:
            st.error("Not enough data for LSTM prediction")
            return

        # Prediction
        preds = model.predict(X)

        # Inverse scaling
        preds = scaler.inverse_transform(preds)

        # Result dataframe
        result_df = df.iloc[time_steps:].copy()
        result_df["Predicted_Load"] = preds.flatten()

        st.subheader(" Forecast Result")
        st.dataframe(result_df.tail())
