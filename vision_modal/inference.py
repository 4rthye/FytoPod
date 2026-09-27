import tensorflow as tf
import numpy as np
import json
import cv2
from pathlib import Path

# Load model and class indices
MODEL_PATH = Path(__file__).parent.parent / "models" / "fytopod_final.keras"
CLASS_INDEX_PATH = Path(__file__).parent.parent / "models" / "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)
    # Invert: index -> class name
    idx_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_path: str) -> dict:
    input_array = preprocess_image(image_path)

    # Dummy sensor sequence (zeros if no real sensor data)
    sensor_input = np.zeros((1, 12, 6), dtype=np.float32)

    preds = model.predict([input_array, sensor_input], verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    disease_class = idx_to_class[pred_idx]

    return {
        "disease_class": disease_class,
        "confidence": confidence,
        "pred_idx": pred_idx,
        "all_probs": preds[0].tolist()
    }