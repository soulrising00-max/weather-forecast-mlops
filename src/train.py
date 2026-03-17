import numpy as np, json, os, subprocess
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date
import joblib

os.makedirs("models", exist_ok=True)

metrics = {}
scaler = joblib.load("models/scaler.pkl")

for region in ["technopark", "thampanoor"]:
    X_train = np.load(f"data/processed/X_train_{region}.npy")
    y_train = np.load(f"data/processed/y_train_{region}.npy")
    X_test = np.load(f"data/processed/X_test_{region}.npy")
    y_test = np.load(f"data/processed/y_test_{region}.npy")

    model = keras.Sequential(
        [
            keras.layers.LSTM(
                64, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2
            ),
            keras.layers.Dense(24),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mae")
    cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[cb],
        verbose=1,
    )

    model.save(f"models/{region}_model.keras")

    # Evaluate — inverse transform temperature column only
    y_pred_scaled = model.predict(X_test)

    # Reconstruct full feature array to inverse transform
    dummy = np.zeros(
        (y_pred_scaled.shape[0] * y_pred_scaled.shape[1], scaler.n_features_in_)
    )
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred_actual = scaler.inverse_transform(dummy)[:, 0].reshape(y_pred_scaled.shape)

    dummy2 = np.zeros((y_test.shape[0] * y_test.shape[1], scaler.n_features_in_))
    dummy2[:, 0] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy2)[:, 0].reshape(y_test.shape)

    mae = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())
    rmse = mean_squared_error(y_test_actual.flatten(), y_pred_actual.flatten()) ** 0.5

    metrics[f"mae_{region}"] = round(mae, 4)
    metrics[f"rmse_{region}"] = round(rmse, 4)
    print(f"{region} — MAE: {mae:.4f}°C  RMSE: {rmse:.4f}°C")

# Write metrics.json
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Write version.json
try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )
except Exception:
    sha = "unknown"

version = {
    "version": date.today().strftime("%Y%m%d"),
    "trained_on": date.today().isoformat(),
    "git_sha": sha,
    "mae_technopark": metrics["mae_technopark"],
    "rmse_technopark": metrics["rmse_technopark"],
    "mae_thampanoor": metrics["mae_thampanoor"],
    "rmse_thampanoor": metrics["rmse_thampanoor"],
}
with open("version.json", "w") as f:
    json.dump(version, f, indent=2)

print("Saved metrics.json and version.json")
