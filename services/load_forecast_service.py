
# ====================================
import streamlit as st
import pandas as pd
import numpy as np
from database.mongo import load_col
from services.hf_model_loader import load_load_forecast_model
from sklearn.preprocessing import MinMaxScaler
from pymongo.errors import NetworkTimeout, AutoReconnect

# =================================================
# MODEL LOAD (ONCE)
# =================================================
@st.cache_resource
def get_model():
    return load_load_forecast_model()

model = get_model()

# =================================================
# SEQUENCE BUILDER
# =================================================
def create_sequences(data, time_steps=24):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)

# =================================================
# MAIN TAB
# =================================================
def load_forecast_tab():
    st.header("📈 Load Forecast (MSN-wise, 500 rows at a time)")

    # ---------------- SESSION STATE ----------------
    if "page" not in st.session_state:
        st.session_state.page = 0

    PAGE_SIZE = 500

    # ---------------- PAGINATION CONTROLS ----------------
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("⬅ Previous", disabled=st.session_state.page == 0):
            st.session_state.page -= 1

    with col2:
        if st.button("Next ➡"):
            st.session_state.page += 1

    with col3:
        st.info(f"Page: {st.session_state.page + 1}")

    # ---------------- SAFE SKIP ----------------
    skip = max(st.session_state.page * PAGE_SIZE, 0)

    # ---------------- LOAD DATA ----------------
    try:
        cursor = (
            load_col
            .find({}, {"_id": 0})
            .sort("ts", 1)
            .skip(skip)
            .limit(PAGE_SIZE)
        )
        data = list(cursor)

    except (NetworkTimeout, AutoReconnect):
        st.error("❌ Database connection issue")
        return

    if not data:
        st.warning("No more data available")
        return

    df = pd.DataFrame(data)

    st.subheader("📂 Loaded Data (preview)")
    st.dataframe(df.head(10))

    # ---------------- MSN DROPDOWN ----------------
    if "msn" not in df.columns:
        st.error("Column `msn` not found in data")
        return

    msn_list = sorted(df["msn"].dropna().unique().tolist())

    if not msn_list:
        st.warning("No MSN IDs found on this page")
        return

    selected_msn = st.selectbox(
        "Select Meter (MSN)",
        msn_list,
        key=f"msn_select_{st.session_state.page}"
    )

    # ---------------- FILTER MSN DATA ----------------
    msn_df = df[df["msn"] == selected_msn].copy()

    if len(msn_df) < 24:
        st.warning("Not enough records for this meter (need ≥ 24)")
        return

    # ---------------- TARGET LOAD COLUMN ----------------
    LOAD_COLUMN = "wh_imp"

    if LOAD_COLUMN not in msn_df.columns:
        st.error("Required column `wh_imp` not found in data")
        return

    # ---------------- PREDICT ----------------
    if st.button("⏳ Predict Load"):

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(msn_df[[LOAD_COLUMN]])

        X = create_sequences(scaled_data, 24)

        if len(X) == 0:
            st.error("Sequence generation failed")
            return

        preds = model.predict(X, verbose=0)
        preds = scaler.inverse_transform(preds)

        predicted_value = round(float(preds[-1][0]), 2)

        st.success(f"🎯 Predicted Load (Wh): {predicted_value}")

        # ---------------- RESULT TABLE ----------------
        result_df = msn_df.iloc[24:].copy()
        result_df["Predicted_Wh"] = preds.flatten()

        st.subheader("📈 Prediction Details")
        st.dataframe(
            result_df[["ts", "wh_imp", "Predicted_Wh"]].tail(10)
        )
