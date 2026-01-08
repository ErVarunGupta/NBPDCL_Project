import streamlit as st
import pandas as pd
import time
from pymongo.errors import NetworkTimeout, AutoReconnect
from database.mongo import energy_col, load_col, stlf_col, theft_col

BATCH_SIZE = 500  # SAFE FOR ATLAS

COLLECTION_MAP = {
    "Energy Usage": energy_col,
    "Load Forecast": load_col,
    "Short-Term Load Forecast": stlf_col,
    "Theft Detection": theft_col,
}

def upload_tab():
    st.header("📂 Upload Dataset to Database")

    dataset_type = st.selectbox(
        "Select Dataset Type",
        list(COLLECTION_MAP.keys())
    )

    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if not file:
        return

    # -------------------------
    # LOAD FILE
    # -------------------------
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📄 Preview")
    st.dataframe(df.head())
    st.info(f"Total records: {len(df)}")

    if st.button("🚀 Upload to Database"):
        records = df.to_dict("records")
        collection = COLLECTION_MAP[dataset_type]

        total = len(records)
        inserted = 0

        progress = st.progress(0)
        status = st.empty()

        try:
            for i in range(0, total, BATCH_SIZE):
                batch = records[i : i + BATCH_SIZE]

                collection.insert_many(batch)
                inserted += len(batch)

                progress.progress(inserted / total)
                status.text(f"Inserted {inserted}/{total} records")

                time.sleep(0.2)  # Prevent Atlas socket kill

            st.success(f"✅ Successfully inserted {inserted} records into `{dataset_type}`")

        except (NetworkTimeout, AutoReconnect) as e:
            st.error("❌ Upload failed due to network timeout")
            st.warning(f"⚠️ Inserted {inserted} records before failure")
            st.exception(e)
