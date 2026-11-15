import sys, json, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT, WX, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ROOT)
from ecometeo.features import FEATURES, TARGET, WEATHER, training_rows
from ecometeo.models import MLPEnsemble
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)

X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl")

# ---------- extra features ----------
lom = [c for c in X.columns if str(c).endswith("_lom")]
dsp = [c for c in X.columns if str(c).endswith("_dsp") and c not in ("timedelta_dsp", "electric_power_dsp", "o2_dsp", "carbon_dsp")]
akp = [c for c in X.columns if str(c).endswith("_akp") and c not in ("timedelta_akp", "electric_power_akp", "argon_akp")]
wire = [c for c in X.columns if str(c).endswith("_akp_wire")]
X["dsp_add_t"] = X[dsp].clip(lower=0).sum(axis=1) / 1000
X["akp_add_t"] = X[akp].clip(lower=0).sum(axis=1) / 1000
X["wire_t"] = X[wire].clip(lower=0).sum(axis=1) / 1000
for c in ("o2_dsp", "carbon_dsp", "gas", "argon_akp", "electric_power_akp", "timedelta_akp"):
    X["x_" + c] = X[c].clip(lower=0)
top_lom = X[lom].sum().sort_values(ascending=False).index[:8].tolist()
for i, c in enumerate(top_lom):
    X[f"lom{i}_share"] = X[c].clip(lower=0) / (X[lom].clip(lower=0).sum(axis=1) + 1)
MAT = ["dsp_add_t", "akp_add_t", "wire_t"] + ["x_" + c for c in ("o2_dsp", "carbon_dsp", "gas", "argon_akp", "electric_power_akp", "timedelta_akp")]
LOMMIX = [f"lom{i}_share" for i in range(len(top_lom))]
hr = X.ts.dt.hour
X["h_sin"], X["h_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
X["dow_we"] = (X.ts.dt.dayofweek >= 5).astype(float)
TIME = ["h_sin", "h_cos", "dow_we"]
X["flow_x"] = X["flow_ok"]

# hourly weather
w = json.load(open(WX, encoding="utf-8"))["hourly"]
W = pd.DataFrame(w); W["time"] = pd.to_datetime(W.time); W = W.set_index("time")
W["wd_sin"] = np.sin(np.deg2rad(W.wind_direction_10m)); W["wd_cos"] = np.cos(np.deg2rad(W.wind_direction_10m))
# value for the hour ending at ts = mean of start and end instants
Wm = (W + W.shift(1)) / 2
HW = Wm.reindex(X.ts.values)
HW.index = X.index
X["T_h"] = HW.temperature_2m; X["RH_h"] = HW.relative_humidity_2m; X["P_h"] = HW.pressure_msl; X["wind_h"] = HW.wind_speed_10m
X["dew_h"] = HW.dew_point_2m; X["rad_h"] = HW.shortwave_radiation; X["cloud_h"] = HW.cloud_cover
X["snow_h"] = HW.snow_depth; X["prec_h"] = HW.precipitation; X["wdsin_h"] = HW.wd_sin; X["wdcos_h"] = HW.wd_cos; X["gust_h"] = HW.wind_gusts_10m
X["T_h24"] = X["T_h"].rolling(24, min_periods=1).mean()
WEATHER_H = ["T_h", "RH_h", "P_h", "wind_h"]
WEATHER_HX = WEATHER_H + ["dew_h", "rad_h", "cloud_h", "snow_h", "prec_h", "wdsin_h", "wdcos_h", "gust_h"]
X["dust_l1"] = X["dust_ok"].shift(1)

base_noW = [f for f in FEATURES if f not in WEATHER]
rows = training_rows(X) & X[WEATHER_HX].notna().all(axis=1)
d = X[rows].copy(); y = d[TARGET]; month = d.ts.dt.month
P("rows", len(d), "months", sorted(month.unique()))

# ---------- models ----------
def ridge(): return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
def spline(): return make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), Ridge(alpha=3.0))
def poly2(): return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=30.0))
def hgb(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
def rf(): return RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0)
class Hybrid:
    """linear + boosting on residuals (boosting only learns what linear missed)"""
    def fit(self, X, y):
        self.l = ridge().fit(X, y); self.b = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=80, l2_regularization=3.0, random_state=0).fit(X, y - self.l.predict(X)); return self
    def predict(self, X): return self.l.predict(X) + self.b.predict(X)
def mlp(): return MLPEnsemble(n_models=5)
def mlp_big(): return MLPEnsemble(n_models=3, hidden=(64, 32), alpha=1e-3)

def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.abs(y - p).mean()

def evaluate(mk, feats):
    pr = pd.Series(index=d.index, dtype=float)
    for m in sorted(month.unique()):
        tr, te = month != m, month == m
        pr[te] = mk().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    r2, mae = scores(y, pr)
    tr, te = month <= 4, month >= 5
    p2 = mk().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    r2t, _ = scores(y[te], p2)
    return r2, mae, r2t, pr

def row(name, mk, feats):
    t0 = time.time(); r2, mae, r2t, pr = evaluate(mk, feats)
    P(f"{name:58s} LOMO R2={r2:6.3f} MAE={mae*1000:5.2f}e-3 | JanApr->MayJul R2={r2t:6.3f}  ({time.time()-t0:.0f}s)")
    return pr

F = base_noW + WEATHER_H + ["wdsin_h", "wdcos_h"]
for nm, msk in [("янв-апр", month <= 4), ("май-июл", month >= 5), ("нечёт. месяцы", month % 2 == 1), ("чёт. месяцы", month % 2 == 0)]:
    m = ridge().fit(d.loc[msk, F], y[msk]); co = dict(zip(F, m[-1].coef_ / m[0].scale_))
    ang = np.degrees(np.arctan2(co["wdsin_h"], co["wdcos_h"])) % 360
    P(f"{nm:14s} ветер из ~{ang:3.0f}°, амплитуда {np.hypot(co['wdsin_h'], co['wdcos_h'])*1000:4.1f}e-3 | T_h {co['T_h']*1000:.2f}e-3/°C | power {co['power']*1000:.3f}e-3")
# per-month LOMO R2: hourly4 vs +wind dir
for nm, fs in [("суточная", FEATURES), ("почасовая", base_noW + WEATHER_H), ("почас.+напр.ветра", F)]:
    pr = pd.Series(index=d.index, dtype=float)
    for mm in sorted(month.unique()):
        tr, te = month != mm, month == mm
        pr[te] = ridge().fit(d.loc[tr, fs], y[tr]).predict(d.loc[te, fs])
    P(f"{nm:20s} r по месяцам: " + " ".join(f"{np.corrcoef(y[month==mm], pr[month==mm])[0,1]:.2f}" for mm in sorted(month.unique())))
# share of each wind sector by half
sec = pd.cut(np.degrees(np.arctan2(d.wdsin_h, d.wdcos_h)) % 360, [0, 90, 180, 270, 360], labels=["С-В", "В-Ю", "Ю-З", "З-С"])
P((pd.crosstab(month <= 4, sec, normalize="index").round(2)).to_string())
P("средний остаток базовой почасовой модели по секторам ветра (1e-3):")
pr = pd.Series(index=d.index, dtype=float); fs = base_noW + WEATHER_H
for mm in sorted(month.unique()):
    tr, te = month != mm, month == mm
    pr[te] = ridge().fit(d.loc[tr, fs], y[tr]).predict(d.loc[te, fs])
P(((y - pr).groupby([month <= 4, sec]).mean().unstack() * 1000).round(1).to_string())
