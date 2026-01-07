import streamlit as st

from services.upload_service import upload_tab
from services.energy_usage_service import energy_tab
from services.load_forecast_service import load_forecast_tab
from services.stlf_service import stlf_tab
from services.theft_service import theft_tab

st.set_page_config(
    page_title="⚡ Smart Energy AI Dashboard",
    layout="wide"
)

st.title("⚡ Smart Energy AI Dashboard")

tabs = st.tabs([
    "📂 Data Upload",
    "⏱ Short-Term Load Forecast",
    "👤 Energy Usage Analysis",
    "📈 Load Forecast (LSTM)",
    "🚨 Theft Detection"
])

with tabs[0]:
    upload_tab()

with tabs[1]:
    stlf_tab()

with tabs[2]:
    energy_tab()

with tabs[3]:
    load_forecast_tab()

with tabs[4]:
    theft_tab()
