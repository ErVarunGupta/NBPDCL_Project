import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
from pymongo.errors import NetworkTimeout, AutoReconnect

from database.mongo import theft_col
from utils.sequence_builder import create_sequences
from services.hf_model_loader import (
    load_lstm_ae,
    load_cnn_lstm_ae,
    load_theft_scaler
)

# ==================================================
# LOAD MODELS (CACHED)
# ==================================================
@st.cache_resource
def load_models():
    return (
        load_lstm_ae(),
        load_cnn_lstm_ae(),
        load_theft_scaler()
    )

lstm_ae, cnn_lstm_ae, scaler = load_models()

# ==================================================
# CONSTANTS
# ==================================================
FEATURES = [
    "v_r", "v_y", "v_b",
    "i_r", "i_y", "i_b",
    "wh_imp", "wh_exp",
    "lsfrequency"
]

SEQ_LEN = 48
TH_ANOMALY = 0.25
TH_THEFT = 0.40

# ==================================================
# SCORE FUNCTIONS
# ==================================================
def anomaly_score(X_seq):
    recon = lstm_ae.predict(X_seq, verbose=0)
    return np.mean((X_seq - recon) ** 2, axis=(1, 2))

def theft_score(X_seq):
    recon = cnn_lstm_ae.predict(X_seq, verbose=0)
    return np.mean((X_seq - recon) ** 2, axis=(1, 2))

def decide(anom, theft):
    if theft > TH_THEFT:
        return "⚠️ THEFT"
    elif anom > TH_ANOMALY:
        return "⚠️ ANOMALY"
    return "✅ NORMAL"

# ==================================================
# STREAMLIT TAB
# ==================================================
def theft_tab():
    st.header("🚨 Theft & Anomaly Detection")

    try:
        data = list(
            theft_col.find({}, {"_id": 0}).limit(200)
        )
    except (NetworkTimeout, AutoReconnect):
        st.warning("🔄 Database timeout. Please retry.")
        return

    if not data:
        st.info("No theft meter data found")
        st.info("Upload data from 📂 Data Upload tab")
        return

    df = pd.DataFrame(data)
    st.dataframe(df.head())

    if st.button("🔍 Run Theft Detection"):

        results = []

        for meter_id, mdf in df.groupby("msn"):

            if len(mdf) <= SEQ_LEN:
                continue

            mdf = mdf.sort_values("ts")

            X = mdf[FEATURES]
            X_scaled = scaler.transform(X)

            X_seq = create_sequences(X_scaled, SEQ_LEN)
            if len(X_seq) == 0:
                continue

            anom = anomaly_score(X_seq).mean()
            theft = theft_score(X_seq).mean()

            status = decide(anom, theft)

            results.append({
                "msn": meter_id,
                "anomaly_score": round(anom, 4),
                "theft_score": round(theft, 4),
                "status": status
            })

        results_df = pd.DataFrame(results)
        st.session_state["theft_results"] = results_df

    # ==================================================
    # RESULTS
    # ==================================================
    if "theft_results" in st.session_state:

        res = st.session_state["theft_results"]

        st.subheader("📊 Detection Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("⚠️ THEFT", (res["status"] == "⚠️ THEFT").sum())
        c2.metric("⚠️ ANOMALY", (res["status"] == "⚠️ ANOMALY").sum())
        c3.metric("✅ NORMAL", (res["status"] == "✅ NORMAL").sum())

        st.subheader("📋 Meter-wise Results")
        st.dataframe(res)

        st.subheader("📈 Status Distribution")
        st.bar_chart(res["status"].value_counts())
