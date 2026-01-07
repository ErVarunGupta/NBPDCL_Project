import tensorflow as tf
import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "varungupta0994/NBPDCL_Models"

def load_lstm_ae():
    path = hf_hub_download(REPO_ID, "lstm_autoencoder.keras")
    return tf.keras.models.load_model(path)

def load_cnn_lstm_ae():
    path = hf_hub_download(REPO_ID, "cnn_lstm_autoencoder.keras")
    return tf.keras.models.load_model(path)

def load_energy_model():
    path = hf_hub_download(REPO_ID, "energy_forecast_model.pkl")
    return joblib.load(path)

def load_load_forecast_model():
    path = hf_hub_download(REPO_ID, "lstm_load_forecast_model.keras")
    return tf.keras.models.load_model(path)

def load_stlf_model():
    path = hf_hub_download(REPO_ID, "stlf_model.joblib")
    return joblib.load(path)

def load_rf_model():
    path = hf_hub_download(REPO_ID, "rf_load_model.joblib")
    return joblib.load(path)

def load_theft_scaler():
    path = hf_hub_download(REPO_ID, "theft_scaler.pkl")
    return joblib.load(path)

def load_load_scaler():
    path = hf_hub_download(REPO_ID, "load_scaler.pkl")
    return joblib.load(path)
