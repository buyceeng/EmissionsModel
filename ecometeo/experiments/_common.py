"""Общий стенд для экспериментов: та же честная проверка, что в models.evaluate_leave_one_month_out."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ecometeo.features import FEATURES, PROCESS, WINDOW

def ridge(a=1.0): return make_pipeline(StandardScaler(), Ridge(alpha=a))

def site_weather(name, shift_h=0):
    """Погода города: значения для часа, кончающегося в ts (среднее ts-1 и ts), сдвиг shift_h часов."""
    w = json.loads((ROOT / "data" / "weather_sites" / f"{name}.json").read_text(encoding="utf-8"))["hourly"]
    d = pd.DataFrame(w); d["time"] = pd.to_datetime(d["time"]); d = d.set_index("time").astype(float)
    for h in ("10m", "100m"):
        a = np.deg2rad(d.pop(f"wind_direction_{h}"))
        d[f"wx{h}"], d[f"wy{h}"] = np.cos(a), np.sin(a)
    d = (d + d.shift(1)) / 2
    d.index = d.index + pd.Timedelta(hours=shift_h)
    o = pd.DataFrame({"T_air": d.temperature_2m, "RH_air": d.relative_humidity_2m, "P_air": d.pressure_msl,
                      "wind": d.wind_speed_10m, "blh_log": np.log(d.boundary_layer_height.clip(lower=10)),
                      "T_app": d.apparent_temperature})
    n = np.hypot(d.wx10m, d.wy10m).replace(0, 1)
    o["wd_cos"], o["wd_sin"] = d.wx10m / n, d.wy10m / n
    return o, d

def evaluate(X, feats, mk=ridge, name="", verbose=True, target="dust_ok"):
    ok = X.status.eq("OK") & X[target].notna() & X[feats].notna().all(axis=1)
    d = X[ok]; y = d[target]; m = d.ts.dt.month
    p = pd.Series(index=d.index, dtype=float)
    for k in sorted(m.unique()):
        p[m == k] = mk().fit(d.loc[m != k, feats], y[m != k]).predict(d.loc[m == k, feats])
    tr, te = m <= 4, m >= 5
    p2 = mk().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    p3 = mk().fit(d.loc[te, feats], y[te]).predict(d.loc[tr, feats])
    r2 = lambda a, b: 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    ape = (y - p).abs() / y * 100
    res = dict(n=len(d), mape=ape.mean(), le10=(ape <= 10).mean() * 100, R2=r2(y, p), strict=r2(y[te], p2), strict_rev=r2(y[tr], p3))
    if verbose:
        print(f"{name:58s} n={res['n']:4d} ошибка={res['mape']:5.2f}% ≤10%={res['le10']:3.0f}% R2={res['R2']:.3f} зима→лето={res['strict']:.3f} лето→зима={res['strict_rev']:.3f}", flush=True)
    res["pred"] = p
    return res

def base_table():
    return pd.read_pickle(ROOT / "artifacts" / "dataset.pkl").copy()
