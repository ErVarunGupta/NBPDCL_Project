import streamlit as st
import pandas as pd
import joblib
import io
from pymongo.errors import NetworkTimeout, AutoReconnect

from database.mongo import energy_col
from services.hf_model_loader import load_energy_model


# ==================================================
# LOAD MODEL (ONCE)
# ==================================================
@st.cache_resource
def get_energy_assets():
    data = load_energy_model()
    return data["model"], data["features"], data["cust_baseline"]

model, features, baseline = get_energy_assets()


# ==================================================
# ENERGY TAB FUNCTION
# ==================================================
def energy_tab():
    st.header("👤 Energy Usage Analysis")

    # ---------------------------
    # LOAD DATA FROM DB
    # ---------------------------
    try:
        data = list(
            energy_col.find({}, {"_id": 0}).limit(200)
        )
    except (NetworkTimeout, AutoReconnect):
        st.warning("🔄 Database timeout. Please retry.")
        return
    
    if not data:
        st.warning("No energy usage data found in database")
        return

    df_db = pd.DataFrame(data)
    st.subheader("📂 Raw Data (from Database)")
    st.dataframe(df_db.head())

    # ---------------------------
    # CONTROL PANEL (INSIDE TAB)
    # ---------------------------
    st.subheader("📊 Control Panel")

    available_msns = baseline["msn_id"].unique()
    selected_msn = st.selectbox("Select Customer (MSN)", available_msns)

    col1, col2, col3 = st.columns(3)

    with col1:
        target_year = st.number_input("Forecast Year", 2026, 2050, 2027)

    with col2:
        month_options = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        selected_month = st.selectbox(
            "Select Month",
            list(month_options.keys()),
            format_func=lambda x: month_options[x]
        )

    with col3:
        unit_price = st.slider("Price per kWh (₹)", 0.0, 30.0, 7.5)

    compare_mode = st.checkbox("Enable Comparison Mode")
    comp_msn = None
    if compare_mode:
        comp_msn = st.selectbox(
            "Compare With",
            [m for m in available_msns if m != selected_msn]
        )

    # ---------------------------
    # FORECAST FUNCTION
    # ---------------------------
    def generate_forecast(msn_id, year, month):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        dates = pd.date_range(start, end, freq="D")

        df = pd.DataFrame({"date": dates, "msn_id": msn_id})
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear
        df["quarter"] = df["date"].dt.quarter

        cust = baseline[baseline["msn_id"] == msn_id].iloc[0]
        for f in [
            "wh_imp_lag1",
            "wh_imp_roll7",
            "vah_imp_lag1",
            "vah_imp_roll7",
        ]:
            df[f] = cust[f]

        preds = model.predict(df[features])
        df["Wh"] = preds[:, 0]
        df["VAh"] = preds[:, 1]
        df["PF"] = (df["Wh"] / df["VAh"]).fillna(1).clip(0, 1)
        df["Cost"] = (df["Wh"] / 1000) * unit_price

        return df

    # ---------------------------
    # RUN FORECAST
    # ---------------------------
    main_df = generate_forecast(selected_msn, target_year, selected_month)

    st.divider()
    st.subheader(
        f"🏢 Energy Forecast: {month_options[selected_month]} {target_year}"
    )

    tab1, tab2, tab3 = st.tabs(
        ["🚀 Usage Analytics", "⚖️ Comparison", "📂 Export"]
    )

    # ==========================
    # TAB 1: ANALYTICS
    # ==========================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Monthly Total", f"{round(main_df['Wh'].sum()/1000,2)} kWh")
        c2.metric("Peak Load", f"{round(main_df['Wh'].max(),1)} Wh")
        c3.metric("Avg Power Factor", f"{round(main_df['PF'].mean(),2)}")
        c4.metric("Estimated Cost", f"₹{round(main_df['Cost'].sum(),2)}")

        st.subheader("Daily Energy Usage (Wh)")
        st.area_chart(main_df.set_index("date")["Wh"])

        st.subheader("Power Factor Trend")
        st.line_chart(main_df.set_index("date")["PF"])

    # ==========================
    # TAB 2: COMPARISON
    # ==========================
    with tab2:
        if not compare_mode:
            st.info("Enable comparison mode to compare customers")
        else:
            comp_df = generate_forecast(comp_msn, target_year, selected_month)

            compare_chart = pd.DataFrame({
                selected_msn: main_df["Wh"].values,
                comp_msn: comp_df["Wh"].values
            }, index=main_df["date"])

            st.line_chart(compare_chart)

    # ==========================
    # TAB 3: EXPORT
    # ==========================
    with tab3:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            main_df.to_excel(writer, index=False, sheet_name="Forecast")

        st.download_button(
            "📊 Download Excel Report",
            data=output.getvalue(),
            file_name=f"Energy_Report_{selected_msn}_{target_year}.xlsx"
        )

        st.dataframe(main_df)

    # ---------------------------
    # LEADERBOARD
    # ---------------------------
    st.divider()
    st.subheader("🏆 Annual Leaderboard (Projected)")

    if st.button("Generate Ranking"):
        leaderboard = []
        for msn in baseline["msn_id"].unique():
            row = baseline[baseline["msn_id"] == msn].iloc[0]
            annual = (row["wh_imp_roll7"] * 365) / 1000
            leaderboard.append({
                "MSN": msn,
                "Annual_kWh": round(annual, 2)
            })

        lb_df = pd.DataFrame(leaderboard).sort_values(
            "Annual_kWh", ascending=False
        ).head(5)

        st.table(lb_df)
