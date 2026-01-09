
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from pymongo.errors import NetworkTimeout, AutoReconnect

from database.mongo import theft_col
from utils.sequence_builder import create_sequences
from services.hf_model_loader import (
    load_lstm_ae,
    load_cnn_lstm_ae,
    load_theft_scaler
)

# ==================================================
# WARNINGS OFF (sklearn / TF / numpy)
# ==================================================
warnings.filterwarnings("ignore")

# ==================================================
# LOAD MODELS (ONCE – SAFE)
# ==================================================
@st.cache_resource
def load_models():
    try:
        lstm_ae = load_lstm_ae()
        cnn_lstm_ae = load_cnn_lstm_ae()
        scaler = load_theft_scaler()
        return lstm_ae, cnn_lstm_ae, scaler
    except Exception as e:
        st.error("❌ Failed to load theft detection models")
        st.exception(e)
        st.stop()

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
# SCORING FUNCTIONS
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

    if "page" not in st.session_state:
        st.session_state.page = 1

    PAGE_SIZE = 500

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous"):
            if st.session_state.page > 1:
                st.session_state.page -= 1

    with col3:
        if st.button("Next ➡️"):
            st.session_state.page += 1

    st.caption(f"Page {st.session_state.page}")

    skip = max(st.session_state.page * PAGE_SIZE, 0)

    try:
        data = list(
            theft_col.find({}, {"_id": 0})
            .skip(skip)
            .limit(PAGE_SIZE)
        )
    except (NetworkTimeout, AutoReconnect):
        st.warning("🔄 Database timeout")
        return

    if not data:
        st.info("No more data available")
        return

    df = pd.DataFrame(data)
    st.dataframe(df)

    # ----------------------------------------------
    # RUN DETECTION
    # ----------------------------------------------
    if st.button("🔍 Run Theft Detection"):

        progress = st.progress(0)
        status_text = st.empty()

        results = []
        meter_ids = df["msn"].unique()
        total = len(meter_ids)

        for idx, meter_id in enumerate(meter_ids):

            status_text.text(
                f"Processing meter {idx + 1}/{total} : {meter_id}"
            )

            mdf = df[df["msn"] == meter_id].sort_values("ts")

            if len(mdf) <= SEQ_LEN:
                continue

            try:
                X = mdf[FEATURES]
                X_scaled = scaler.transform(X)

                X_seq = create_sequences(X_scaled, SEQ_LEN)
                if len(X_seq) == 0:
                    continue

                anom = anomaly_score(X_seq).mean()
                theft = theft_score(X_seq).mean()

                results.append({
                    "msn": meter_id,
                    "anomaly_score": round(float(anom), 4),
                    "theft_score": round(float(theft), 4),
                    "status": decide(anom, theft)
                })

            except Exception:
                continue  # corrupted meter → skip safely

            progress.progress((idx + 1) / total)

        results_df = pd.DataFrame(results)
        st.session_state["theft_results"] = results_df

        st.success("✅ Theft detection completed")

    # ----------------------------------------------
    # RESULTS SECTION (SHOW ALL METERS)
    # ----------------------------------------------
    if "theft_results" in st.session_state:

        res = st.session_state["theft_results"]

        st.subheader("📊 Detection Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("⚠️ THEFT", (res["status"] == "⚠️ THEFT").sum())
        c2.metric("⚠️ ANOMALY", (res["status"] == "⚠️ ANOMALY").sum())
        c3.metric("✅ NORMAL", (res["status"] == "✅ NORMAL").sum())

        # ------------------------------------------
        # FULL METER STATUS TABLE
        # ------------------------------------------
        st.subheader("📋 Meter-wise Status")
        st.dataframe(
            res.sort_values("status"),
            use_container_width=True
        )

        # ------------------------------------------
        # STATUS DISTRIBUTION
        # ------------------------------------------
        st.subheader("📈 Status Distribution")
        st.bar_chart(res["status"].value_counts())
