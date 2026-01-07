
from pymongo import MongoClient
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")    #for local system
MONGO_URI = st.secrets["MONGO_URI"]     # for app deployment on streamlit

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)
db = client["smart_energy_ai"]

energy_col = db["energy_usage"]
load_col = db["load_forecast"]
stlf_col = db["stlf_forecast"]
theft_col = db["theft_detection"]
