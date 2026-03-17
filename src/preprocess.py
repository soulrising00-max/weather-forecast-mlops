import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

REGIONS = ["technopark", "thampanoor"]
WINDOW = 48
HORIZON = 24
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

all_data = []

for region in REGIONS:
    df = pd.read_csv(f"data/raw/{region}.csv")
    df = df.dropna(subset=["temperature_2m"])
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df = df.drop(columns=["time"])
    all_data.append((region, df))

# Fit scaler on technopark data, apply to both
features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "hour",
    "dayofweek",
]
scaler = MinMaxScaler()
scaler.fit(all_data[0][1][features])
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved.")

for region, df in all_data:
    scaled = scaler.transform(df[features])

    X, y = [], []
    for i in range(len(scaled) - WINDOW - HORIZON + 1):
        X.append(scaled[i : i + WINDOW])
        y.append(scaled[i + WINDOW : i + WINDOW + HORIZON, 0])  # temperature only

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    np.save(f"data/processed/X_train_{region}.npy", X_train)
    np.save(f"data/processed/X_test_{region}.npy", X_test)
    np.save(f"data/processed/y_train_{region}.npy", y_train)
    np.save(f"data/processed/y_test_{region}.npy", y_test)

    print(f"{region}: X_train={X_train.shape}, X_test={X_test.shape}")
