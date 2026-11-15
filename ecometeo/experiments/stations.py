"""Наблюдения метеостанций Южного Урала (NOAA ISD-Lite) против модельной погоды ERA5 (Open-Meteo).
Данные: python experiments/fetch_isd.py"""
import sys, gzip; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X = base_table()
ISD = ROOT / "data" / "isd"
ST = pd.read_csv(ISD / "stations.csv", dtype=str)
WXC = ["T_air", "RH_air", "wind", "wd_sin", "wd_cos", "T_app"]


def station(usaf, wban):
    """Почасовой ряд станции в местном времени (UTC+5), значения для часа, кончающегося в ts."""
    d = pd.read_csv(gzip.open(ISD / f"{usaf}-{wban}-2022.gz", "rt"), sep=r"\s+", header=None,
                    names=["y", "m", "d", "h", "T", "Td", "P", "wd", "ws", "sky", "p1", "p6"]).replace(-9999, np.nan)
    d.index = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h)) + pd.Timedelta(hours=5)
    o = pd.DataFrame({"T": d["T"] / 10, "Td": d.Td / 10, "P": d.P / 10, "ws": d.ws / 10 * 3.6}, index=d.index)   # км/ч, как у Open-Meteo
    a = np.deg2rad(d.wd); o["wx"], o["wy"] = np.cos(a) * (d.ws > 0), np.sin(a) * (d.ws > 0)
    o = o[~o.index.duplicated()].reindex(pd.date_range("2022-01-01", "2022-12-31 23:00", freq="h"))
    cover = o["T"].notna().mean()
    o = o.interpolate(limit=5, limit_area="inside")
    o = (o + o.shift(1)) / 2
    return o, cover


def to_features(o):
    T, Td = o["T"], o["Td"]
    rh = (100 * np.exp(17.625 * Td / (243.04 + Td)) / np.exp(17.625 * T / (243.04 + T))).clip(0, 100)
    n = np.hypot(o.wx, o.wy).replace(0, np.nan)
    f = pd.DataFrame({"T_air": T, "RH_air": rh, "wind": o.ws, "wd_cos": (o.wx / n).fillna(0), "wd_sin": (o.wy / n).fillna(0)})
    e = rh / 100 * 6.105 * np.exp(17.27 * T / (237.7 + T)); f["T_app"] = T + 0.33 * e - 0.70 * f.wind / 3.6 - 4.0
    return f


base = evaluate(X, FEATURES, name="ERA5, Челябинск (текущая модель)")
X0 = X.drop(columns=WXC)
S = {}
print("\n=== Одна станция: T, влажность, ветер — наблюдения; давление и слой перемешивания — ERA5 ===")
for _, r in ST.iterrows():
    o, cover = station(r.usaf, r.wban); S[f"{r['name']} {r.usaf}"] = o
    Z = X0.join(to_features(o), on="ts")
    ok = Z.status.eq("OK"); miss = Z.loc[ok, "T_air"].isna().mean()
    evaluate(Z, FEATURES, name=f"{r['name']} ({r.lat},{r.lon}) сроков {cover:.0%}, пропусков {miss:.0%}")
print("\n=== Сводка по региону: среднее по станциям ===")
def mean_of(keys):
    return to_features(pd.concat([S[k] for k in keys]).groupby(level=0).mean())
groups = {"все 8 станций Южного Урала": list(S), "Челябинская обл. (без Георгиевки и Комсомольца)": [k for k in S if not k.startswith(("GEORG", "KOMSOM"))],
          "Челябинск: Баландино + город": [k for k in S if k.startswith(("BALAND", "CHELYAB"))]}
for nm, keys in groups.items():
    Z = X0.join(mean_of(keys), on="ts"); evaluate(Z, FEATURES, name=nm)
    Z2 = X.join(mean_of(keys).add_prefix("st_"), on="ts")
    evaluate(Z2, FEATURES + ["st_T_air", "st_RH_air", "st_wind", "st_wd_sin", "st_wd_cos"], name="   ERA5 Челябинск + эта сводка вместе")
    evaluate(Z2, FEATURES + ["st_T_air"], name="   ERA5 Челябинск + только её температура")
print("\n=== ERA5 Челябинск + температура отдельной станции ===")
for k, o in S.items():
    Z = X.copy(); Z["st_T"] = o["T"].reindex(Z.ts).values
    evaluate(Z, FEATURES + ["st_T"], name=k)
print("\nсправка: корреляция температуры станции с ERA5-Челябинск (все часы года) и средняя разница, °C")
for k, o in S.items():
    t = o["T"].reindex(X.ts).values; print(f"  {k:28s} r={pd.Series(t).corr(X.T_air):.3f}  Δ={np.nanmean(t - X.T_air):+.2f}")
