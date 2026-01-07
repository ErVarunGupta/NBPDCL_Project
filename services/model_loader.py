import joblib
import tensorflow as tf

def load_models():
    return {
        # STLF
        "rf_load": joblib.load("models/rf_load_model.joblib"),
        "stlf": joblib.load("models/stlf_model.joblib"),

        # Theft detection
        "lstm_ae": tf.keras.models.load_model("models/lstm_autoencoder.keras"),
        "cnn_lstm_ae": tf.keras.models.load_model("models/cnn_lstm_autoencoder.keras"),
        "theft_scaler": joblib.load("scalers/theft_scaler.pkl"),

        # Deep load forecast
        "lstm_forecast": tf.keras.models.load_model(
            "models/lstm_load_forecast_model.keras"
        ),
        "load_scaler": joblib.load("scalers/load_scaler.pkl"),

        # Energy usage
        "energy_model": joblib.load("models/energy_forecast_model.pkl"),
    }
