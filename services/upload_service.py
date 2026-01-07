import streamlit as st
import pandas as pd
from database.mongo import energy_col, load_col, stlf_col, theft_col

def upload_tab():
    st.header("📂 Upload Dataset to Database")

    dataset_type = st.selectbox(
        "Select Dataset Type",
        [
            "Energy Usage",
            "Load Forecast",
            "Short-Term Load Forecast",
            "Theft Detection"
        ]
    )

    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.dataframe(df.head())

        if st.button("🚀 Upload"):
            if dataset_type == "Energy Usage":
                energy_col.insert_many(df.to_dict("records"))
            elif dataset_type == "Load Forecast":
                load_col.insert_many(df.to_dict("records"))
            elif dataset_type == "Short-Term Load Forecast":
                stlf_col.insert_many(df.to_dict("records"))
            elif dataset_type == "Theft Detection":
                theft_col.insert_many(df.to_dict("records"))

            st.success(f"✅ {len(df)} records inserted")
