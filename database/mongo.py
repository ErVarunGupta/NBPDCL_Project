
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)
db = client["smart_energy_ai"]

energy_col = db["energy_usage"]
load_col = db["load_forecast"]
stlf_col = db["stlf_forecast"]
theft_col = db["theft_detection"]
