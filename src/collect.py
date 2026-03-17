# import requests, pandas as pd
# from datetime import datetime, timedelta
# from pathlib import Path

# LOCATIONS = {
#     "technopark":  {"lat": 8.5574, "lon": 76.8800},
#     "thampanoor":  {"lat": 8.4875, "lon": 76.9525},
# }
# VARIABLES = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
# END = datetime.utcnow().date() - timedelta(days=1)
# START = END - timedelta(days=180)

# for name, coords in LOCATIONS.items():
#     out = Path(f"data/raw/{name}.csv")
#     url = "https://archive-api.open-meteo.com/v1/archive"
#     params = {
#         "latitude": coords["lat"], "longitude": coords["lon"],
#         "start_date": str(START), "end_date": str(END),
#         "hourly": VARIABLES, "timezone": "Asia/Kolkata"
#     }
#     r = requests.get(url, params=params, timeout=30)
#     r.raise_for_status()
#     data = r.json()["hourly"]
#     df = pd.DataFrame(data)
#     if out.exists():
#         existing = pd.read_csv(out)
#         df = pd.concat([existing, df]).drop_duplicates("time").reset_index(drop=True)
#     df.to_csv(out, index=False)
#     print(f"Saved {len(df)} rows → {out}")
import requests, pandas as pd, os
from datetime import date, timedelta

LOCATIONS = {
    "technopark":  {"lat": 8.5574, "lon": 76.8800},
    "thampanoor":  {"lat": 8.4875, "lon": 76.9525},
}
VARIABLES = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"

def fetch(name, lat, lon):
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=180)
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "hourly": VARIABLES,
        "start_date": start, "end_date": end,
        "timezone": "Asia/Kolkata"
    })
    df = pd.DataFrame(r.json()["hourly"])
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{name}.csv", index=False)
    print(f"Saved {len(df)} rows for {name}")

for name, coords in LOCATIONS.items():
    fetch(name, **coords)