from pathlib import Path
from preprocessing import prepare_data
from lstm_model import build_lstm, MODEL_PATH
import tensorflow as tf

DATA_PATH = Path(__file__).parent.parent / "data" / "sample" / "sensor_data_6months.csv"

def train(epochs: int = 20, batch_size: int = 32):
    data = prepare_data(str(DATA_PATH))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]

    # Class weights for imbalanced stress events
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weights = {0: 1.0, 1: n_neg / (n_pos + 1e-6)}
    print(f"Class weights: {class_weights}")

    model = build_lstm()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", patience=4,
            restore_best_weights=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, factor=0.3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_recall",
            save_best_only=True, mode="max", verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


if __name__ == "__main__":
    model, history = train()
    print(f"Model saved to {MODEL_PATH}")