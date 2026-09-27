import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "fytopod_lstm_model.keras"
SEQ_LEN    = 12
N_FEATURES = 4  # moisture, temp, light, ph


def build_lstm(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES) -> Model:
    inputs = layers.Input(shape=(seq_len, n_features), name="sensor_input")
    x = layers.LSTM(32, dropout=0.2, name="lstm_layer")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="stress_output")(x)
    model = Model(inputs, output, name="FytoPod_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")]
    )
    return model


def load_lstm() -> Model:
    return tf.keras.models.load_model(MODEL_PATH)


def predict_stress(model: Model, sequence: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    sequence: shape (1, 12, 4)
    Returns stress probability and binary prediction.
    """
    prob = float(model.predict(sequence, verbose=0)[0][0])
    return {
        "stress_probability": round(prob, 4),
        "stress_predicted":   prob >= threshold,
        "early_warning":      prob >= threshold,
        "alert": "Stress predicted — consider watering soon!"
                 if prob >= threshold else "Plant conditions normal"
    }