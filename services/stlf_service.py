import streamlit as st
import pandas as pd
import joblib
from database.mongo import stlf_col
from services.hf_model_loader import load_stlf_model

@st.cache_resource
def get_model():
    return load_stlf_model()

model = get_model()

def stlf_tab():
    st.header("⏱ Short-Term Load Forecast")

    data = list(stlf_col.find())
    if not data:
        st.warning("No STLF data found")
        return

    df = pd.DataFrame(data)
    st.dataframe(df.head())

    if st.button("⏳ Predict Next Hours"):
        preds = model.predict(df.select_dtypes(include="number"))
        df["STLF_Prediction"] = preds
        st.dataframe(df)
