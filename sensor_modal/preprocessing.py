import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

SCALER_PATH  = Path(__file__).parent.parent / "models" / "fytopod_scaler.pkl"
FEATURE_COLS = ["moisture", "temperature", "light", "ph", "humidity", "nitrogen"]
SEQ_LEN      = 12


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.dropna(subset=FEATURE_COLS)
    return df


def scale_features(df: pd.DataFrame, fit: bool = True) -> tuple:
    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[FEATURE_COLS])
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        scaled = scaler.transform(df[FEATURE_COLS])
    return scaled, scaler


def create_sequences(scaled: np.ndarray, labels: np.ndarray,
                     seq_len: int = SEQ_LEN) -> tuple:
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X), np.array(y)


def prepare_data(csv_path: str, seq_len: int = SEQ_LEN,
                 test_size: float = 0.2) -> dict:
    df     = load_data(csv_path)
    scaled, scaler = scale_features(df, fit=True)
    labels = df["stress_in_3h"].values
    X, y   = create_sequences(scaled, labels, seq_len)
    split  = int(len(X) * (1 - test_size))
    return {
        "X_train": X[:split], "y_train": y[:split],
        "X_val":   X[split:], "y_val":   y[split:],
        "scaler":  scaler,
    }


def get_latest_sequence(csv_path: str,
                        seq_len: int = SEQ_LEN) -> np.ndarray:
    df     = load_data(csv_path)
    scaled, _ = scale_features(df, fit=False)
    n_feat = len(FEATURE_COLS)
    return scaled[-seq_len:].reshape(1, seq_len, n_feat)