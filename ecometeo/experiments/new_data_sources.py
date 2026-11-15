import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT, WXD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ROOT)
from ecometeo.features import FEATURES
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)

X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl").copy()
def hourly(path, key="hourly"):
    w = json.load(open(path, encoding="utf-8"))[key]
    df = pd.DataFrame(w); df["time"] = pd.to_datetime(df.time); df = df.set_index("time").astype(float)
    df = (df + df.shift(1)) / 2          # час, кончающийся в ts
    return df
E = hourly(WXD + "/extra.json"); A = hourly(WXD + "/aq.json")
P("CAMS: доля непустых по месяцам (pm10):", A.pm10.notna().groupby(A.index.month).mean().round(2).to_dict())
P("ERA5 extra: доля непустых BLH:", E.boundary_layer_height.notna().mean().round(2))
for c in E.columns: X["e_" + c] = E[c].reindex(X.ts).values
for c in A.columns: X["a_" + c] = A[c].reindex(X.ts).values
X["blh_log"] = np.log(X["e_boundary_layer_height"].clip(lower=10))
# температура: память на 1-3 суток, суточный ход
for h in (24, 72):
    X[f"T_roll{h}"] = X["T_air"].rolling(h, min_periods=1).mean()
X["T_d24"] = X["T_air"] - X["T_air"].shift(24)
X["P_d3"] = X["P_air"] - X["P_air"].shift(3)
# абсолютная влажность и плотность воздуха
T = X["T_air"]; RH = X["RH_air"]
X["abs_hum"] = 6.112 * np.exp(17.67 * T / (T + 243.5)) * RH * 2.1674 / (273.15 + T)
X["air_dens"] = X["P_air"] * 100 / (287.05 * (273.15 + T))
# газы того же прибора (одновременно с пылью)
GAS = ["NO_ok", "NO2_ok", "SO2_ok", "CO_ok", "O2_ok", "flow_ok"]
for g in GAS:
    X[g + "_l1"] = X[g].shift(1)

def ridge(a=1.0): return make_pipeline(StandardScaler(), Ridge(alpha=a))
def hgb(): return HistGradientBoostingRegressor(max_iter=400, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1., random_state=0)

def ev(feats, mk=ridge, log=False, name=""):
    ok = X.status.eq("OK") & X["dust_ok"].notna() & X[feats].notna().all(axis=1)
    d = X[ok]; y = d["dust_ok"]; yt = np.log(y) if log else y; m = d.ts.dt.month
    p = pd.Series(index=d.index, dtype=float)
    for k in sorted(m.unique()):
        tr, te = m != k, m == k
        p[te] = mk().fit(d.loc[tr, feats], yt[tr]).predict(d.loc[te, feats])
    if log: p = np.exp(p)
    tr, te = m <= 4, m >= 5
    p2 = mk().fit(d.loc[tr, feats], yt[tr]).predict(d.loc[te, feats]); p2 = np.exp(p2) if log else p2
    ape = (y - p).abs() / y * 100
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2t = 1 - ((y[te] - p2) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    P(f"{name:62s} n={len(d):4d}  ошибка={ape.mean():5.2f}%  ≤10%={(ape<=10).mean()*100:3.0f}%  R2={r2:.3f}  строгая R2={r2t:.3f}")
    return ape.mean()

BASE = FEATURES
EXW = ["e_boundary_layer_height", "blh_log", "e_apparent_temperature", "e_vapour_pressure_deficit", "e_cloud_cover_low",
       "e_soil_temperature_0_to_7cm", "e_wet_bulb_temperature_2m", "e_direct_radiation", "e_is_day"]
MEM = ["T_roll24", "T_roll72", "T_d24", "P_d3", "abs_hum", "air_dens"]
AQ = ["a_pm10", "a_pm2_5", "a_dust", "a_carbon_monoxide", "a_nitrogen_dioxide", "a_sulphur_dioxide"]

P("\n=== B. Газы того же прибора (известны одновременно с пылью) ===")
ev(BASE, name="текущая модель")
ev(BASE + GAS, name="+ NO, NO2, SO2, CO, O2, расход (этот час)")
for g in GAS:
    ev(BASE + [g], name=f"  + {g}")
ev(BASE + [g + "_l1" for g in GAS], name="+ газы прошлого часа")
P("\n=== C. Комбинации ===")
W2 = BASE + EXW + MEM
BW = BASE + ["blh_log", "e_apparent_temperature", "e_soil_temperature_0_to_7cm", "abs_hum"]
ev(BW, name="Ridge: база + лучшие погодные (BLH, ощущ. t, почва, абс.влажн.)")
ev(W2, lambda: ridge(10), name="Ridge α=10: расширенная погода")
ev(W2, hgb, name="Бустинг: расширенная погода")
ev(BW + GAS, name="Ridge: лучшие погодные + газы прибора")
ev(W2 + GAS, lambda: ridge(10), name="Ridge α=10: расширенная погода + газы прибора")
ev(W2 + GAS, hgb, name="Бустинг: расширенная погода + газы прибора")
