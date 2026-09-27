import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = Path(__file__).parent.parent / "models" / "lifespan_rf.pkl"

def generate_training_data(n_samples: int = 1500, seed: int = 42) -> tuple:
    np.random.seed(seed)
    severity   = np.random.uniform(0, 1, n_samples)
    stress     = np.random.uniform(0, 1, n_samples)
    age        = np.random.uniform(10, 120, n_samples)
    treatment  = np.random.randint(0, 2, n_samples)
    watering   = np.random.uniform(0, 1, n_samples)
    light      = np.random.uniform(0, 1, n_samples)

    target = (
        60
        - severity  * 25
        - stress    * 10
        - (age / 120) * 15
        + treatment * 18
        + watering  * 8
        + light     * 5
        + np.random.normal(0, 3, n_samples)
    )
    target = np.clip(target, 5, 90)

    X = np.column_stack([severity, stress, age, treatment, watering, light])
    return X, target


def train_and_save():
    X, y = generate_training_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved to {MODEL_PATH}")
    return model


def load_model():
    if not MODEL_PATH.exists():
        print("Model not found, training now...")
        return train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_lifespan(disease_severity: float, stress_score: float,
                     plant_age: float, watering_score: float = 0.7,
                     light_score: float = 0.7) -> dict:
    rf = load_model()

    X_untreated = np.array([[disease_severity, stress_score, plant_age,
                              0, watering_score, light_score]])
    X_treated   = np.array([[disease_severity * 0.4, stress_score * 0.5, plant_age,
                              1, watering_score, light_score]])

    days_untreated = float(rf.predict(X_untreated)[0])
    days_treated   = float(rf.predict(X_treated)[0])

    return {
        "days_without_treatment": round(days_untreated, 1),
        "days_with_treatment":    round(days_treated, 1),
        "treatment_benefit_days": round(days_treated - days_untreated, 1)
    }