"""Сетка точек ERA5 (шаг 0,25°) вокруг Екатеринбурга — Первоуральска: янв–июль 2022, только признаки модели."""
import json, time, urllib.parse, urllib.request
from pathlib import Path
VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m,boundary_layer_height,apparent_temperature"
out = Path(__file__).resolve().parent.parent / "data" / "weather_sites" / "grid"; out.mkdir(parents=True, exist_ok=True)
pts = [(round(55.75 + 0.25 * i, 2), round(58.75 + 0.25 * j, 2)) for i in range(9) for j in range(12)]
for k in range(0, len(pts), 12):
    chunk = [p for p in pts[k:k + 12] if not (out / f"g_{p[0]}_{p[1]}.json").exists()]
    if not chunk: continue
    q = urllib.parse.urlencode({"latitude": ",".join(str(p[0]) for p in chunk), "longitude": ",".join(str(p[1]) for p in chunk),
                                "start_date": "2021-12-30", "end_date": "2022-07-31", "hourly": VARS, "timezone": "Asia/Yekaterinburg"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=120) as r:
                res = json.loads(r.read())
            res = res if isinstance(res, list) else [res]
            for p, one in zip(chunk, res):
                (out / f"g_{p[0]}_{p[1]}.json").write_text(json.dumps(one), encoding="utf-8")
            print("ok", chunk[0], len(res), flush=True); break
        except Exception as e:
            print("retry", e, flush=True); time.sleep(65)
    time.sleep(20)
