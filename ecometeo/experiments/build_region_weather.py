"""Собирает почасовую погоду «в среднем по Челябинской области» (Южный Урал) из 7 городов.
Источник: реанализ ERA5 через Open-Meteo Historical Weather API (файлы data/weather_sites/, см. fetch_sites.py).
Запуск: python experiments/build_region_weather.py -> data/weather_region_2022_hourly.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ecometeo.weather_live import POINTS

SCALAR = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "boundary_layer_height", "apparent_temperature"]
frames = []
for name in POINTS:
    w = json.loads((ROOT / "data" / "weather_sites" / f"{name}.json").read_text(encoding="utf-8"))["hourly"]
    d = pd.DataFrame({k: w[k] for k in SCALAR + ["wind_direction_10m"]}, index=pd.to_datetime(w["time"])).astype(float)
    a = np.deg2rad(d.pop("wind_direction_10m")); d["wx"], d["wy"] = np.cos(a), np.sin(a)
    frames.append(d.loc["2022-01-01":"2022-12-31 23:00"])
m = sum(frames) / len(frames)
m["wind_direction_10m"] = (np.degrees(np.arctan2(m.pop("wy"), m.pop("wx"))) % 360).round(1)   # направление — как средний вектор
out = {"source": "ERA5 via Open-Meteo Historical Weather API; mean of " + ", ".join(POINTS), "timezone": "Asia/Yekaterinburg",
       "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in m.index], **{c: m[c].round(2).tolist() for c in m.columns}}}
(ROOT / "data" / "weather_region_2022_hourly.json").write_text(json.dumps(out), encoding="utf-8")
print(len(m), "часов;", m.isna().sum().sum(), "пропусков")
