# import streamlit as st, json
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import keras
# import joblib, requests
# from datetime import datetime, timedelta, timezone

# with open("version.json") as f:
#     v = json.load(f)

# LOCATIONS = {
#     "technopark": {"lat": 8.5574, "lon": 76.8800},
#     "thampanoor": {"lat": 8.4875, "lon": 76.9525},
# }
# FEATURES = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
# WINDOW = 48
# HORIZON = 24


# # def fetch_recent(lat, lon):
# #     end = datetime.now(timezone.utc)
# #     start = end - timedelta(hours=WINDOW + 1)
# #     url = "https://api.open-meteo.com/v1/forecast"
# #     r = requests.get(
# #         url,
# #         params={
# #             "latitude": lat,
# #             "longitude": lon,
# #             "hourly": ",".join(FEATURES),
# #             "start_date": start.strftime("%Y-%m-%d"),
# #             "end_date": end.strftime("%Y-%m-%d"),
# #             "timezone": "Asia/Kolkata",
# #             "forecast_days": 2,
# #         },
# #         timeout=30,
# #     )
# #     r.raise_for_status()
# #     data = r.json()["hourly"]
# #     df = pd.DataFrame(data)
# #     df["time"] = pd.to_datetime(df["time"])
# #     df = df.dropna(subset=["temperature_2m"])
# #     return df.tail(WINDOW).reset_index(drop=True)
# def fetch_recent(lat, lon):
#     end = datetime.now(timezone.utc).date()
#     start = end - timedelta(days=3)
#     url = "https://archive-api.open-meteo.com/v1/archive"
#     r = requests.get(
#         url,
#         params={
#             "latitude": lat,
#             "longitude": lon,
#             "hourly": ",".join(FEATURES),
#             "start_date": start.isoformat(),
#             "end_date": end.isoformat(),
#             "timezone": "Asia/Kolkata",
#         },
#         timeout=30,
#     )
#     r.raise_for_status()
#     data = r.json()["hourly"]
#     df = pd.DataFrame(data)
#     df["time"] = pd.to_datetime(df["time"])
#     df = df.dropna(subset=["temperature_2m"])
#     return df.tail(WINDOW).reset_index(drop=True)


# def make_forecast(df, model, scaler):
#     df["hour"] = df["time"].dt.hour
#     df["dayofweek"] = df["time"].dt.dayofweek
#     features = FEATURES + ["hour", "dayofweek"]
#     scaled = scaler.transform(df[features])
#     X = scaled[-WINDOW:].reshape(1, WINDOW, len(features))
#     y_scaled = model.predict(X, verbose=0)[0]  # shape (24,)
#     dummy = np.zeros((HORIZON, scaler.n_features_in_))
#     dummy[:, 0] = y_scaled
#     y_actual = scaler.inverse_transform(dummy)[:, 0]
#     return y_actual


# scaler = joblib.load("models/scaler.pkl")

# st.set_page_config(page_title="Weather Forecast — Trivandrum")
# st.caption(
#     f"Model v{v['version']} | Trained: {v['trained_on']} | RMSE: {v['rmse_technopark']:.2f}°C"
# )
# st.title("Weather Forecast — Technopark & Thampanoor")

# tab1, tab2 = st.tabs(["Technopark", "Thampanoor"])

# for tab, region in zip([tab1, tab2], ["technopark", "thampanoor"]):
#     with tab:
#         coords = LOCATIONS[region]
#         with st.spinner("Fetching latest data and running forecast..."):
#             try:
#                 df_recent = fetch_recent(coords["lat"], coords["lon"])
#                 model = keras.saving.load_model(f"models/{region}_model.keras")
#                 forecast = make_forecast(df_recent, model, scaler)

#                 # Build forecast timestamps (next 24 hours)
#                 last_time = df_recent["time"].iloc[-1]
#                 forecast_times = [
#                     last_time + timedelta(hours=i + 1) for i in range(HORIZON)
#                 ]
#                 df_forecast = pd.DataFrame(
#                     {"time": forecast_times, "Forecast (°C)": forecast}
#                 )

#                 # Actuals (last 48 hours)
#                 df_actual = df_recent[["time", "temperature_2m"]].copy()
#                 df_actual = df_actual.rename(columns={"temperature_2m": "Actual (°C)"})

#                 # Plot
#                 fig = px.line(
#                     df_forecast,
#                     x="time",
#                     y="Forecast (°C)",
#                     title=f"{region.capitalize()} — 24-hour forecast",
#                 )
#                 fig.add_scatter(
#                     x=df_actual["time"],
#                     y=df_actual["Actual (°C)"],
#                     mode="lines",
#                     name="Actual (°C)",
#                     line=dict(dash="dash", color="gray"),
#                 )
#                 fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
#                 st.plotly_chart(fig, use_container_width=True)

#                 # Sidebar stats
#                 with st.sidebar:
#                     st.subheader(f"{region.capitalize()}")
#                     st.metric("Min forecast", f"{forecast.min():.1f}°C")
#                     st.metric("Max forecast", f"{forecast.max():.1f}°C")
#                     st.metric(
#                         "Avg humidity",
#                         f"{df_recent['relative_humidity_2m'].mean():.0f}%",
#                     )
#                     st.metric(
#                         "Wind speed", f"{df_recent['wind_speed_10m'].mean():.1f} km/h"
#                     )

#             except Exception as e:
#                 st.error(f"Error: {e}")
import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Weather Forecast", page_icon="🌦️")

v = json.load(open("version.json"))
st.caption(
    f"Model v{v['version']} | Trained: {v['trained_on']} | "
    f"RMSE Technopark: {v['rmse_technopark']:.2f}°C | "
    f"RMSE Thampanoor: {v['rmse_thampanoor']:.2f}°C"
)

st.title("🌦️ Weather Forecast — Thiruvananthapuram")

cache = json.load(open("forecast_cache.json"))

LOCATIONS = {
    "Technopark": "technopark",
    "Thampanoor": "thampanoor",
}

tabs = st.tabs(list(LOCATIONS.keys()))

for tab, (loc_name, key) in zip(tabs, LOCATIONS.items()):
    with tab:
        data = cache[key]
        actuals = data["actuals"]
        actual_times = pd.to_datetime(data["times"])
        forecast = data["forecast"]
        last_time = actual_times[-1]
        forecast_times = [last_time + timedelta(hours=i + 1) for i in range(24)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual_times, y=actuals, name="Actual", line=dict(color="steelblue")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_times,
                y=forecast,
                name="Forecast",
                line=dict(color="tomato", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{loc_name} — 24h Forecast",
            xaxis_title="Time",
            yaxis_title="Temp (°C)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Min Forecast", f"{min(forecast):.1f}°C")
        col2.metric("Max Forecast", f"{max(forecast):.1f}°C")
        col3.metric("Avg Humidity", f"{data['humidity']:.0f}%")
