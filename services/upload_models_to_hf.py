from huggingface_hub import HfApi, upload_file
import os

REPO_ID = "varungupta0994/NBPDCL_Models"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = D:\NBPDCL Internship\Final Project

files = [
    os.path.join(BASE_DIR, "models", "lstm_autoencoder.keras"),
    os.path.join(BASE_DIR, "models", "cnn_lstm_autoencoder.keras"),
    os.path.join(BASE_DIR, "models", "lstm_load_forecast_model.keras"),
    os.path.join(BASE_DIR, "models", "energy_forecast_model.pkl"),
    os.path.join(BASE_DIR, "models", "rf_load_model.joblib"),
    os.path.join(BASE_DIR, "models", "stlf_model.joblib"),
    os.path.join(BASE_DIR, "scalers", "theft_scaler.pkl"),
    os.path.join(BASE_DIR, "scalers", "load_scaler.pkl"),
]

api = HfApi()

for f in files:
    print("Uploading:", f)
    upload_file(
        path_or_fileobj=f,
        path_in_repo=os.path.basename(f),
        repo_id=REPO_ID,
        repo_type="model"
    )

print("✅ All models uploaded successfully")
