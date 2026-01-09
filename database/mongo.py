
from pymongo import MongoClient
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_db():
    MONGO_URI = os.getenv("MONGO_URI")    #for local system
    # MONGO_URI = st.secrets["MONGO_URI"]     # for app deployment on streamlit

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=20000,
        maxPoolSize=10,
        retryWrites=True,
        retryReads=True,
    )

    # Force initial connection
    client.admin.command("ping")
    return client["smart_energy_ai"]

db = get_db()

energy_col = db["energy_usage"]
load_col   = db["load_forecast"]
stlf_col   = db["stlf_forecast"]
theft_col  = db["theft_detection"]

