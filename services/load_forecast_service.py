import streamlit as st
import pandas as pd
import tensorflow as tf
from database.mongo import load_col
from services.hf_model_loader import load_load_forecast_model

@st.cache_resource
def get_model():
    return load_load_forecast_model()

model = get_model()

def load_forecast_tab():
    st.header("📈 Load Forecast (LSTM)")

    data = list(load_col.find())
    if not data:
        st.warning("No data available")
        return

    df = pd.DataFrame(data)
    st.dataframe(df.head())

    if st.button("⚡ Forecast Load"):
        preds = model.predict(df.select_dtypes(include="number"))
        df["Predicted_Load"] = preds.flatten()
        st.dataframe(df)
