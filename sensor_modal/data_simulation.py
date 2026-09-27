import numpy as np
import pandas as pd
from pathlib import Path

HOURS = 4320
SEED  = 42
np.random.seed(SEED)


def simulate_sensor_data(hours: int = HOURS) -> pd.DataFrame:
    t           = np.arange(hours)
    season      = np.sin(2 * np.pi * t / (24 * 30))
    hour_of_day = t % 24

    # Soil moisture
    moisture = 55 + 15 * season + np.random.normal(0, 3, hours)
    watering_events = np.random.choice(hours, size=60, replace=False)
    for w in watering_events:
        moisture[w:w+6] += np.linspace(20, 0, 6)
    moisture = np.clip(moisture, 10, 95)

    # Temperature
    temperature = (
        22 + 5 * season
        + 4 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
        + np.random.normal(0, 0.8, hours)
    )
    temperature = np.clip(temperature, 12, 35)

    # Light
    light = np.maximum(0, 400 * np.sin(np.pi * hour_of_day / 24))
    light += np.random.normal(0, 15, hours)
    light = np.clip(light, 0, 500)

    # pH
    ph = 6.5 + 0.3 * season + np.random.normal(0, 0.15, hours)
    ph = np.clip(ph, 5.5, 7.5)

    # Humidity
    humidity = 60 + 10 * season + np.random.normal(0, 4, hours)
    humidity = np.clip(humidity, 30, 95)

    # Nitrogen
    nitrogen = 40 + 8 * season + np.random.normal(0, 2, hours)
    nitrogen = np.clip(nitrogen, 15, 65)

    # Noise spikes ±1.2%
    for arr in [moisture, temperature, light, ph, humidity, nitrogen]:
        spike_idx = np.random.choice(hours, size=30, replace=False)
        arr[spike_idx] *= np.random.uniform(0.988, 1.012, size=30)

    df = pd.DataFrame({
        "timestamp":   pd.date_range("2024-01-01", periods=hours, freq="h"),
        "moisture":    moisture,
        "temperature": temperature,
        "light":       light,
        "ph":          ph,
        "humidity":    humidity,
        "nitrogen":    nitrogen,
    })

    # Dropout periods — linear interpolation
    dropout_starts = np.random.choice(hours - 10, size=15, replace=False)
    for start in dropout_starts:
        length = np.random.randint(2, 8)
        end    = min(start + length, hours - 1)
        for col in ["moisture", "temperature", "light", "ph", "humidity", "nitrogen"]:
            df.loc[start:end, col] = np.nan
    df = df.interpolate(method="linear")

    # Stress labels
    df["stressed_now"] = (
        (df["moisture"]    < 30) |
        (df["temperature"] > 30) |
        (df["light"]       < 50) |
        (df["ph"]          < 5.8) |
        (df["ph"]          > 7.2)
    ).astype(int)

    df["stress_in_3h"] = df["stressed_now"].shift(-3).fillna(0).astype(int)

    return df


if __name__ == "__main__":
    df  = simulate_sensor_data()
    out = Path(__file__).parent.parent / "data" / "sample" / "sensor_data_6months.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.head())
    print(f"\nStress events: {df['stress_in_3h'].sum()} / {len(df)}")