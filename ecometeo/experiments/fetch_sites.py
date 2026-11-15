"""Скачивает почасовую погоду 2022 (Open-Meteo archive, ERA5) для городов-кандидатов Урала.
Запуск: python experiments/fetch_sites.py   -> data/weather_sites/<город>.json"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

SITES = {
    "chelyabinsk": (55.15, 61.38), "magnitogorsk": (53.42, 59.05), "zlatoust": (55.17, 59.67),
    "asha": (55.00, 57.27), "novotroitsk": (51.20, 58.30), "pervouralsk": (56.91, 59.94),
    "polevskoy": (56.44, 60.19), "revda": (56.80, 59.93), "ekaterinburg": (56.84, 60.61),
    "miass": (55.05, 60.11), "satka": (55.05, 59.03), "troitsk": (54.08, 61.56),
    "beloretsk": (53.96, 58.40), "kamensk": (56.42, 61.93), "n_tagil": (57.92, 59.97),
    "serov": (59.60, 60.57), "ufa": (54.74, 55.97), "kurgan": (55.44, 65.34),
    "perm": (58.01, 56.25), "tyumen": (57.15, 65.53), "izhevsk": (56.85, 53.20),
    "tula": (54.19, 37.62), "st_oskol": (51.30, 37.84), "vyksa": (55.32, 42.17),
}
VARS = ("temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,surface_pressure,precipitation,snowfall,"
        "snow_depth,cloud_cover,cloud_cover_low,wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m,"
        "wind_gusts_10m,shortwave_radiation,boundary_layer_height,apparent_temperature,soil_temperature_0_to_7cm,"
        "et0_fao_evapotranspiration,vapour_pressure_deficit")
out = Path(__file__).resolve().parent.parent / "data" / "weather_sites"
out.mkdir(parents=True, exist_ok=True)
for name, (lat, lon) in SITES.items():
    f = out / f"{name}.json"
    if f.exists():
        continue
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": "2021-12-25", "end_date": "2022-12-31",
                                "hourly": VARS, "timezone": "Asia/Yekaterinburg"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=60) as r:
                f.write_bytes(r.read()); print("ok", name, flush=True); break
        except Exception as e:
            print("retry", name, e, flush=True); time.sleep(15 * (attempt + 1))
    time.sleep(2)
