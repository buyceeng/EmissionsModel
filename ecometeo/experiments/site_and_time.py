"""Чья погода лучше объясняет пыль (поиск площадки по данным) и верно ли совмещено время."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X0 = base_table()
WX = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
PW = PROCESS + WINDOW
print("=== 0. текущая модель (файл проекта) ===")
evaluate(X0, FEATURES, name="как в проекте")
print("\n=== 1. Города: те же признаки, погода другого города ===")
rows = []
for f in sorted((ROOT / "data" / "weather_sites").glob("*.json")):
    o, _ = site_weather(f.stem)
    X = X0.drop(columns=WX).join(o, on="ts")
    r = evaluate(X, PW + WX, verbose=False); rows.append({"город": f.stem, **{k: round(v, 3) for k, v in r.items() if k != "pred"}})
t = pd.DataFrame(rows).sort_values("R2", ascending=False); print(t.to_string(index=False))
print("\n=== 2. Сдвиг времени погоды (ч), топ-города ===")
for s in list(t["город"][:4]) + ["chelyabinsk"]:
    line = []
    for sh in (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6):
        o, _ = site_weather(s, sh)
        r = evaluate(X0.drop(columns=WX).join(o, on="ts"), PW + WX, verbose=False)
        line.append(f"{sh:+d}:{r['R2']:.3f}/{r['mape']:.2f}")
    print(f"{s:14s}", "  ".join(line), flush=True)
