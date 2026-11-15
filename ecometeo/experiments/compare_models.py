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

# ---------- 1. residual diagnostics ----------
P("\n=== 1. Нелинейность: остатки линейной модели (LOMO) по децилям признака ===")
pr_lin = row("Ridge base", ridge, FEATURES)
res = y - pr_lin
for f in ["power", "td_min", "scrap_t", "T_air", "RH_air", "P_air", "wind", "cycle_hour"]:
    q = pd.qcut(d[f].rank(method="first"), 10, labels=False)
    g = res.groupby(q).mean() * 1000
    P(f"  {f:10s} средний остаток по децилям (1e-3): " + " ".join(f"{v:+5.1f}" for v in g.values))
P(f"  std остатка = {res.std()*1000:.1f}e-3")
# T x power interaction check
tq = pd.qcut(d.T_air, 3, labels=["холод", "средне", "тепло"]); pq = pd.qcut(d.power.rank(method="first"), 3, labels=["P low", "P mid", "P hi"])
P("  средний остаток (1e-3) температура × мощность:\n" + (res.groupby([tq, pq]).mean().unstack() * 1000).round(1).to_string())

# ---------- 2. model families on base features ----------
P("\n=== 2. Модели на текущих признаках ===")
row("Сплайны (GAM-подобная) + Ridge", spline, FEATURES)
row("Полином 2-й степени (все взаимодействия) + Ridge", poly2, FEATURES)
row("Градиентный бустинг (HGB)", hgb, FEATURES)
row("Случайный лес", rf, FEATURES)
row("Гибрид: Ridge + бустинг по остаткам", Hybrid, FEATURES)
row("MLP ансамбль (текущая нейросеть)", mlp, FEATURES)
row("MLP большая 64-32, слабая регуляризация", mlp_big, FEATURES)

# ---------- 3. feature sets ----------
P("\n=== 3. Дополнительные признаки (Ridge / бустинг / гибрид) ===")
sets = {
    "base": FEATURES,
    "base без погоды": base_noW,
    "+ время суток/выходные": FEATURES + TIME,
    "+ материалы ДСП/АКП, кислород, углерод, газ": FEATURES + MAT,
    "+ состав лома (доли 8 главных видов)": FEATURES + LOMMIX,
    "погода почасовая вместо суточной": base_noW + WEATHER_H,
    "погода почасовая + суточная": FEATURES + WEATHER_H,
    "погода почасовая расширенная (ветер-направление, солнце, снег...)": base_noW + WEATHER_HX + ["T_air"],
    "всё вместе (без расхода)": FEATURES + TIME + MAT + LOMMIX + WEATHER_HX,
}
for nm, fs in sets.items():
    row(f"Ridge  | {nm}", ridge, fs)
    row(f"HGB    | {nm}", hgb, fs)
    row(f"Гибрид | {nm}", Hybrid, fs)

P("\n=== 4. Признаки с того же прибора (не для прогноза, только для монитора) ===")
row("Ridge  | base + расход газа (flow)", ridge, FEATURES + ["flow_x"])
d = d[d.dust_l1.notna()]; y = d[TARGET]; month = d.ts.dt.month
P(f"  (дальше только {len(d)} строк, где известна пыль прошлого часа)")
row("Ridge  | base (на этих же строках)", ridge, FEATURES)
row("Ridge  | base + пыль прошлого часа (авторегрессия)", ridge, FEATURES + ["dust_l1"])
P(f"  corr(T_air суточная, T_h почасовая) = {d.T_air.corr(d.T_h):.3f}")
