
from pymongo import MongoClient
import os
import streamlit as st

# MONGO_URI = os.getenv("mongodb://localhost:27017/")
MONGO_URI = st.secrets["MONGO_URI"]

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)
db = client["smart_energy_ai"]

energy_col = db["energy_usage"]
load_col = db["load_forecast"]
stlf_col = db["stlf_forecast"]
theft_col = db["theft_detection"]
