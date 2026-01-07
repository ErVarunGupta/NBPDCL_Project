import streamlit as st
import pandas as pd
import joblib
from database.mongo import energy_col
from services.hf_model_loader import load_energy_model

@st.cache_resource
def get_model():
    return load_energy_model()

model_data = get_model()
model = model_data["model"]
features = model_data["features"]

def energy_tab():
    st.header("👤 Energy Usage Analysis")

    data = list(energy_col.find())
    if not data:
        st.warning("No data found")
        return

    df = pd.DataFrame(data)
    st.dataframe(df.head())

    if st.button("📊 Predict Energy"):
        preds = model.predict(df[features])
        df["Predicted_Wh"] = preds[:, 0]
        st.dataframe(df)
