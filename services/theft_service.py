
# # =====================================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import warnings
# from pymongo.errors import NetworkTimeout, AutoReconnect

# from database.mongo import theft_col
# from utils.sequence_builder import create_sequences
# from services.hf_model_loader import (
#     load_lstm_ae,
#     load_cnn_lstm_ae,
#     load_theft_scaler
# )

# # ==================================================
# # WARNINGS OFF (sklearn / TF / numpy)
# # ==================================================
# warnings.filterwarnings("ignore")

# # ==================================================
# # LOAD MODELS (ONCE – SAFE)
# # ==================================================
# @st.cache_resource
# def load_models():
#     try:
#         lstm_ae = load_lstm_ae()
#         cnn_lstm_ae = load_cnn_lstm_ae()
#         scaler = load_theft_scaler()
#         return lstm_ae, cnn_lstm_ae, scaler
#     except Exception as e:
#         st.error("❌ Failed to load theft detection models")
#         st.exception(e)
#         st.stop()

# lstm_ae, cnn_lstm_ae, scaler = load_models()

# # ==================================================
# # CONSTANTS
# # ==================================================
# FEATURES = [
#     "v_r", "v_y", "v_b",
#     "i_r", "i_y", "i_b",
#     "wh_imp", "wh_exp",
#     "lsfrequency"
# ]

# SEQ_LEN = 48
# TH_ANOMALY = 0.25
# TH_THEFT = 0.40

# # ==================================================
# # SCORING FUNCTIONS
# # ==================================================
# def anomaly_score(X_seq):
#     recon = lstm_ae.predict(X_seq, verbose=0)
#     return np.mean((X_seq - recon) ** 2, axis=(1, 2))

# def theft_score(X_seq):
#     recon = cnn_lstm_ae.predict(X_seq, verbose=0)
#     return np.mean((X_seq - recon) ** 2, axis=(1, 2))

# def decide(anom, theft):
#     if theft > TH_THEFT:
#         return "⚠️ THEFT"
#     elif anom > TH_ANOMALY:
#         return "⚠️ ANOMALY"
#     return "✅ NORMAL"

# # ==================================================
# # STREAMLIT TAB
# # ==================================================
# def theft_tab():
#     st.header("🚨 Theft & Anomaly Detection")

#     if "page" not in st.session_state:
#         st.session_state.page = 1

#     PAGE_SIZE = 500

#     col1, col2, col3 = st.columns([1, 2, 1])

#     with col1:
#         if st.button("⬅️ Previous"):
#             if st.session_state.page > 1:
#                 st.session_state.page -= 1

#     with col3:
#         if st.button("Next ➡️"):
#             st.session_state.page += 1

#     st.caption(f"Page {st.session_state.page}")

#     skip = max(st.session_state.page * PAGE_SIZE, 0)

#     try:
#         data = list(
#             theft_col.find({}, {"_id": 0})
#             .skip(skip)
#             .limit(PAGE_SIZE)
#         )
#     except (NetworkTimeout, AutoReconnect):
#         st.warning("🔄 Database timeout")
#         return

#     if not data:
#         st.info("No more data available")
#         return

#     df = pd.DataFrame(data)
#     st.dataframe(df)

#     # ----------------------------------------------
#     # RUN DETECTION
#     # ----------------------------------------------
#     if st.button("🔍 Run Theft Detection"):

#         progress = st.progress(0)
#         status_text = st.empty()

#         results = []
#         meter_ids = df["msn"].unique()
#         total = len(meter_ids)

#         for idx, meter_id in enumerate(meter_ids):

#             status_text.text(
#                 f"Processing meter {idx + 1}/{total} : {meter_id}"
#             )

#             mdf = df[df["msn"] == meter_id].sort_values("ts")

#             if len(mdf) <= SEQ_LEN:
#                 continue

#             try:
#                 X = mdf[FEATURES]
#                 X_scaled = scaler.transform(X)

#                 X_seq = create_sequences(X_scaled, SEQ_LEN)
#                 if len(X_seq) == 0:
#                     continue

#                 anom = anomaly_score(X_seq).mean()
#                 theft = theft_score(X_seq).mean()

#                 results.append({
#                     "msn": meter_id,
#                     "anomaly_score": round(float(anom), 4),
#                     "theft_score": round(float(theft), 4),
#                     "status": decide(anom, theft)
#                 })

#             except Exception:
#                 continue  # corrupted meter → skip safely

#             progress.progress((idx + 1) / total)

#         results_df = pd.DataFrame(results)
#         st.session_state["theft_results"] = results_df

#         st.success("✅ Theft detection completed")

#     # ----------------------------------------------
#     # RESULTS SECTION (SHOW ALL METERS)
#     # ----------------------------------------------
#     if "theft_results" in st.session_state:

#         res = st.session_state["theft_results"]

#         st.subheader("📊 Detection Summary")

#         c1, c2, c3 = st.columns(3)
#         c1.metric("⚠️ THEFT", (res["status"] == "⚠️ THEFT").sum())
#         c2.metric("⚠️ ANOMALY", (res["status"] == "⚠️ ANOMALY").sum())
#         c3.metric("✅ NORMAL", (res["status"] == "✅ NORMAL").sum())

#         # ------------------------------------------
#         # FULL METER STATUS TABLE
#         # ------------------------------------------
#         st.subheader("📋 Meter-wise Status")
#         st.dataframe(
#             res.sort_values("status"),
#             use_container_width=True
#         )

#         # ------------------------------------------
#         # STATUS DISTRIBUTION
#         # ------------------------------------------
#         st.subheader("📈 Status Distribution")
#         st.bar_chart(res["status"].value_counts())


# ====================================================



# =====================================================
# Power Theft & Anomaly Detection (FINAL – FIXED)
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

warnings.filterwarnings("ignore")

# =====================================================
# LOAD MODELS (ONCE)
# =====================================================
@st.cache_resource
def load_models():
    lstm_ae = load_lstm_ae()
    cnn_lstm_ae = load_cnn_lstm_ae()
    scaler = load_theft_scaler()
    return lstm_ae, cnn_lstm_ae, scaler

lstm_ae, cnn_lstm_ae, scaler = load_models()

# =====================================================
# CONSTANTS
# =====================================================
FEATURES = [
    "v_r", "v_y", "v_b",
    "i_r", "i_y", "i_b",
    "wh_imp", "wh_exp",
    "lsfrequency"
]

SEQ_LEN = 48

ANOMALY_RATIO_TH = 0.15
THEFT_RATIO_TH   = 0.20
ZERO_CURRENT_TH  = 0.30

# =====================================================
# HELPERS
# =====================================================
def ae_error(model, X_seq):
    recon = model.predict(X_seq, verbose=0)
    return np.mean((X_seq - recon) ** 2, axis=(1, 2))

def decide(anom_ratio, theft_ratio, zero_current_ratio):
    if theft_ratio > THEFT_RATIO_TH and zero_current_ratio > ZERO_CURRENT_TH:
        return "⚠️ THEFT"
    elif anom_ratio > ANOMALY_RATIO_TH:
        return "⚠️ ANOMALY"
    return "✅ NORMAL"

def format_indian_time(ts_array):
    return [
        pd.to_datetime(ts, unit="ms")
        .tz_localize("UTC")
        .tz_convert("Asia/Kolkata")
        .strftime("%d-%m-%Y %H:%M")
        for ts in ts_array
    ]

# =====================================================
# STREAMLIT TAB
# =====================================================
def theft_tab():
    st.header("🚨 Power Theft & Anomaly Detection")

    PAGE_SIZE = 500
    if "page" not in st.session_state:
        st.session_state.page = 1

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous") and st.session_state.page > 1:
            st.session_state.page -= 1

    with col3:
        if st.button("Next ➡️"):
            st.session_state.page += 1

    st.caption(f"Page {st.session_state.page}")

    skip = (st.session_state.page) * PAGE_SIZE

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
        st.info("No data available")
        return

    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["ts"])
    st.dataframe(df, use_container_width=True)

    # =================================================
    # RUN DETECTION
    # =================================================
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

            # -----------------------------------------
            # INSUFFICIENT DATA CASE
            # -----------------------------------------
            if len(mdf) <= SEQ_LEN:
                results.append({
                    "msn": meter_id,
                    "status": "⚪ INSUFFICIENT DATA",
                    "anomaly_ratio": 0.0,
                    "theft_ratio": 0.0,
                    "zero_current_ratio": 0.0,
                    "anomaly_timestamps": [],
                    "theft_timestamps": []
                })
                progress.progress((idx + 1) / total)
                continue

            try:
                X = mdf[FEATURES]
                X_scaled = scaler.transform(X)
                X_seq = create_sequences(X_scaled, SEQ_LEN)

                if len(X_seq) == 0:
                    continue

                seq_end_times = mdf["ts"].iloc[SEQ_LEN:].astype("int64") // 10**6

                anom_errors  = ae_error(lstm_ae, X_seq)
                theft_errors = ae_error(cnn_lstm_ae, X_seq)

                anom_th  = np.percentile(anom_errors, 99.7)
                theft_th = np.percentile(theft_errors, 99.9)

                anom_flags  = anom_errors  > anom_th
                theft_flags = theft_errors > theft_th

                anom_ratio  = np.mean(anom_flags)
                theft_ratio = np.mean(theft_flags)

                zero_current_ratio = (
                    (mdf[["i_r", "i_y", "i_b"]].abs().sum(axis=1) < 0.05).mean()
                )

                status = decide(anom_ratio, theft_ratio, zero_current_ratio)

                # -----------------------------------------
                # TIMESTAMPS (ONLY IF NOT NORMAL)
                # -----------------------------------------
                if status == "✅ NORMAL":
                    anom_times = []
                    theft_times = []
                else:
                    anom_times  = format_indian_time(seq_end_times[anom_flags][:5])
                    theft_times = format_indian_time(seq_end_times[theft_flags][:5])

                results.append({
                    "msn": meter_id,
                    "status": status,
                    "anomaly_ratio": round(float(anom_ratio), 3),
                    "theft_ratio": round(float(theft_ratio), 3),
                    "zero_current_ratio": round(float(zero_current_ratio), 3),
                    "anomaly_timestamps": anom_times,
                    "theft_timestamps": theft_times
                })

            except Exception:
                continue

            progress.progress((idx + 1) / total)

        st.session_state["theft_results"] = pd.DataFrame(results)
        st.success("✅ Theft detection completed")

    # =================================================
    # RESULTS
    # =================================================
    if "theft_results" in st.session_state:

        res = st.session_state["theft_results"]

        st.subheader("📊 Detection Summary")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("⚠️ THEFT", (res["status"] == "⚠️ THEFT").sum())
        c2.metric("⚠️ ANOMALY", (res["status"] == "⚠️ ANOMALY").sum())
        c3.metric("✅ NORMAL", (res["status"] == "✅ NORMAL").sum())
        c4.metric("⚪ INSUFFICIENT", (res["status"] == "⚪ INSUFFICIENT DATA").sum())

        st.subheader("📋 Meter-wise Status")
        st.dataframe(res, use_container_width=True)

        st.subheader("📈 Status Distribution")
        st.bar_chart(res["status"].value_counts())

