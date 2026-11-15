"""Карта точности по сетке ERA5 (0,25°): где погода лучше всего объясняет пыль."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X0 = base_table()
WX = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
CORE = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log"]
rows = []
for f in sorted((ROOT / "data" / "weather_sites" / "grid").glob("g_*.json")):
    lat, lon = map(float, f.stem.split("_")[1:])
    w = json.loads(f.read_text(encoding="utf-8"))["hourly"]
    d = pd.DataFrame(w); d["time"] = pd.to_datetime(d["time"]); d = d.set_index("time").astype(float)
    a = np.deg2rad(d.wind_direction_10m); d["wx"], d["wy"] = np.cos(a), np.sin(a)
    d = (d + d.shift(1)) / 2
    o = pd.DataFrame({"T_air": d.temperature_2m, "RH_air": d.relative_humidity_2m, "P_air": d.pressure_msl, "wind": d.wind_speed_10m,
                      "blh_log": np.log(d.boundary_layer_height.clip(lower=10)), "wd_cos": d.wx, "wd_sin": d.wy})
    r = evaluate(X0.drop(columns=WX).join(o, on="ts"), PROCESS + WINDOW + CORE, verbose=False)
    rows.append({"lat": lat, "lon": lon, "mape": r["mape"], "R2": r["R2"], "strict": r["strict"], "rev": r["strict_rev"]})
t = pd.DataFrame(rows)
for c in ("mape", "R2", "strict", "rev"):
    print(f"\n{c} (строки — широта, столбцы — долгота)"); print(t.pivot(index="lat", columns="lon", values=c).sort_index(ascending=False).round(3 if c != "mape" else 2).to_string())
print("\nлучшие точки:"); print(t.sort_values("mape").head(8).round(3).to_string(index=False))
