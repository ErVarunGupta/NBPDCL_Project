# import streamlit as st
# import pandas as pd
# import numpy as np
# from database.mongo import load_col
# from services.hf_model_loader import load_load_forecast_model
# from sklearn.preprocessing import MinMaxScaler
# from pymongo.errors import NetworkTimeout, AutoReconnect

# # ------------------ MODEL LOAD ------------------

# @st.cache_resource
# def get_model():
#     return load_load_forecast_model()

# model = get_model()

# # ------------------ HELPER FUNCTION ------------------

# def create_sequences(data, time_steps=24):
#     X = []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#     return np.array(X)

# # ------------------ STREAMLIT TAB ------------------

# def load_forecast_tab():
#     st.header("📈 Load Forecast (LSTM – Short Term)")

#     # Load data from MongoDB
#     try:
#         data = list(
#             load_col.find({}, {"_id": 0}).limit(200)
#         )
#     except (NetworkTimeout, AutoReconnect):
#         st.warning("🔄 Database timeout. Please retry.")
#         return
#     if not data:
#         st.warning("No data available in database")
#         return

#     df = pd.DataFrame(data)
#     st.subheader("Input Data Preview")
#     st.dataframe(df.head())

#     # -------- SELECT LOAD COLUMN --------
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()

#     if not numeric_cols:
#         st.error("No numeric load column found")
#         return

#     load_col_name = st.selectbox(
#         "Select Load Column for Forecasting",
#         numeric_cols
#     )

#     # -------- FORECAST BUTTON --------
#     if st.button("⚡ Forecast Load (LSTM)"):

#         # Scaling
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(df[[load_col_name]])

#         # Sequence creation
#         time_steps = 24   # short-term (24 hours)
#         X = create_sequences(scaled_data, time_steps)

#         if len(X) == 0:
#             st.error("Not enough data for LSTM prediction")
#             return

#         # Prediction
#         preds = model.predict(X)

#         # Inverse scaling
#         preds = scaler.inverse_transform(preds)

#         # Result dataframe
#         result_df = df.iloc[time_steps:].copy()
#         result_df["Predicted_Load"] = preds.flatten()

#         st.subheader(" Forecast Result")
#         st.dataframe(result_df.tail())


# ============================

# import streamlit as st
# import pandas as pd
# import numpy as np
# from database.mongo import load_col
# from services.hf_model_loader import load_load_forecast_model
# from sklearn.preprocessing import MinMaxScaler
# from pymongo.errors import NetworkTimeout, AutoReconnect

# # ================== MODEL LOAD ==================

# @st.cache_resource
# def get_model():
#     return load_load_forecast_model()

# model = get_model()

# # ================== HELPER FUNCTIONS ==================

# def create_sequences(data, time_steps):
#     X = []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#     return np.array(X)


# def recursive_forecast(model, last_sequence, scaler, future_steps):
#     preds = []
#     current_seq = last_sequence.copy()

#     for _ in range(future_steps):
#         pred = model.predict(
#             current_seq.reshape(1, -1, 1),
#             verbose=0
#         )
#         preds.append(pred[0][0])

#         # slide window
#         current_seq = np.roll(current_seq, -1)
#         current_seq[-1] = pred

#     preds = scaler.inverse_transform(
#         np.array(preds).reshape(-1, 1)
#     )
#     return preds.flatten()

# # ================== STREAMLIT TAB ==================

# def load_forecast_tab():
#     st.header("📈 Load Forecast (LSTM – Short Term)")

#     # -------- CONSUMER DROPDOWN (msn_id) --------
#     try:
#         consumers = load_col.distinct("msn_id")
#     except (NetworkTimeout, AutoReconnect):
#         st.warning("🔄 Database timeout. Please retry.")
#         return

#     if not consumers:
#         st.error("❌ No consumers (msn_id) found in database")
#         return

#     selected_consumer = st.selectbox(
#         "👤 Select Consumer (msn_id)",
#         consumers
#     )

#     # -------- LOAD DATA FOR SELECTED CONSUMER --------
#     data = list(
#         load_col.find(
#             {"msn_id": selected_consumer},
#             {"_id": 0}
#         ).sort("ts", 1)
#     )

#     if not data:
#         st.error("❌ No data found for selected consumer")
#         return

#     df = pd.DataFrame(data)
#     df["ts"] = pd.to_datetime(df["ts"])

#     st.subheader("📋 Consumer Data Preview")
#     st.dataframe(df.head())

#     # -------- SELECT LOAD COLUMN --------
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()

#     if not numeric_cols:
#         st.error("❌ No numeric columns available for forecasting")
#         return

#     load_col_name = st.selectbox(
#         "⚡ Select Load Column (Recommended: wh_imp)",
#         numeric_cols,
#         index=numeric_cols.index("wh_imp") if "wh_imp" in numeric_cols else 0
#     )

#     # -------- HISTORY & FORECAST OPTIONS --------
#     history_hours = st.selectbox(
#         "⏮️ Select Past Data Window (Hours)",
#         [24, 48, 72]
#     )

#     forecast_hours = st.selectbox(
#         "⏭️ Select Forecast Horizon (Hours)",
#         [24, 48]
#     )

#     # -------- FORECAST BUTTON --------
#     if st.button("⚡ Forecast Load"):

#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(df[[load_col_name]])

#         if len(scaled_data) < history_hours:
#             st.error("❌ Not enough data for selected history window")
#             return

#         last_sequence = scaled_data[-history_hours:]

#         preds = recursive_forecast(
#             model=model,
#             last_sequence=last_sequence,
#             scaler=scaler,
#             future_steps=forecast_hours
#         )

#         # -------- FUTURE TIMESTAMPS --------
#         future_index = pd.date_range(
#             start=df["ts"].iloc[-1],
#             periods=forecast_hours + 1,
#             freq="H"
#         )[1:]

#         forecast_df = pd.DataFrame({
#             "Timestamp": future_index,
#             "Predicted_Load": preds
#         })

#         st.subheader("📊 Forecast Result")
#         st.dataframe(forecast_df)

#         st.line_chart(
#             forecast_df.set_index("Timestamp")["Predicted_Load"]
#         )

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
    st.header("📈 Load Forecast (500 rows → MSN-wise)")

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

    # 🔥 SAFE SKIP (NEVER NEGATIVE)
    skip = max(st.session_state.page * PAGE_SIZE, 0)

    # ---------------- LOAD 500 ROWS ----------------
    try:
        cursor = (
            load_col
            .find({}, {"_id": 0})
            .sort("timestamp", 1)
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

    st.subheader("📂 Loaded Data (current page)")
    st.dataframe(df.head(10))

    # ---------------- MSN DROPDOWN ----------------
    if "msn_id" not in df.columns:
        st.error("Column `msn_id` not found in data")
        return

    msn_list = sorted(df["msn_id"].dropna().unique().tolist())

    if not msn_list:
        st.warning("No MSN IDs found in this page")
        return

    selected_msn = st.selectbox(
        "Select MSN ID (from these 500 rows)",
        msn_list,
        key=f"msn_select_{st.session_state.page}"  # 🔥 reset on page change
    )

    # ---------------- FILTER DATA ----------------
    msn_df = df[df["msn_id"] == selected_msn].copy()

    if len(msn_df) < 24:
        st.warning("Selected MSN has less than 24 records on this page")
        return

    # ---------------- LOAD COLUMN ----------------
    numeric_cols = msn_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["msn_id"]]

    if not numeric_cols:
        st.error("No numeric load column found for selected MSN")
        return

    # load_col_name = st.selectbox(
    #     "Select Load Column",
    #     numeric_cols,
    #     key=f"load_col_{st.session_state.page}"
    # )

    # ---------------- PREDICT ----------------
    if st.button("⏳ Predict Load"):

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(msn_df[["msn"]])

        X = create_sequences(scaled_data, 24)

        if len(X) == 0:
            st.error("Not enough data after sequence creation")
            return

        preds = model.predict(X, verbose=0)
        preds = scaler.inverse_transform(preds)

        predicted_value = round(float(preds[-1][0]), 2)

        st.success(f"🎯 Predicted Load: {predicted_value}")

        # ---------------- RESULT TABLE ----------------
        result_df = msn_df.iloc[24:].copy()
        result_df["Predicted_Load"] = preds.flatten()

        st.subheader("📈 Prediction Details")
        st.dataframe(result_df.tail(10))
