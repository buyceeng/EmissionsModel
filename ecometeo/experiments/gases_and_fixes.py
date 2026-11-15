import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
from ecometeo.features import FEATURES, WEATHER, PROCESS, WINDOW
from ecometeo.models import make_linear
from sklearn.ensemble import HistGradientBoostingRegressor
out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)

X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl").copy()
# --- новые медленные признаки (без замеров пыли) ---
for c in ("power", "scrap_t"):
    for sp in (6, 12, 24):
        X[f"{c}_ema{sp}"] = X[c].ewm(span=sp).mean()
for sp in (6, 24):
    X[f"T_ema{sp}"] = X["T_air"].ewm(span=sp).mean()
rep = X["mode"].eq("ОСТАНОВ")
last_rep = X["ts"].where(rep).ffill()
X["h_after_repair"] = ((X["ts"] - last_rep).dt.total_seconds() / 3600).clip(upper=72).fillna(72)
X["after_repair24"] = (X["h_after_repair"] <= 24).astype(float) * (~rep)
X["stop_share24"] = (X["power"] <= 0).astype(float).rolling(24, min_periods=1).mean()
SLOW = [c for c in X.columns if "_ema" in c] + ["h_after_repair", "after_repair24", "stop_share24"]
X["doy"] = X.ts.dt.dayofyear

def lomo(target, feats, mk=make_linear, rows=None):
    ok = X[target].notna() & X[feats].notna().all(axis=1) & X.status.eq("OK")
    d = X[ok]; y = d[target]; m = d.ts.dt.month
    p = pd.Series(index=d.index, dtype=float)
    for k in sorted(m.unique()):
        tr, te = m != k, m == k
        p[te] = mk().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    e = y - p
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    tr, te = m <= 4, m >= 5
    p2 = mk().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    r2t = 1 - ((y[te] - p2) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    mape = (e.abs() / y.abs().clip(lower=1e-9)).mean() * 100
    return r2, r2t, mape, (e.abs() / y.abs() <= .1).mean() * 100, d.index, p

P("=== 1. Пыль: можно ли уменьшить ошибку новыми признаками ===")
for nm, fs in [("текущая модель", FEATURES), ("+ сглаженная мощность/лом/температура", FEATURES + [c for c in SLOW if "_ema" in c]),
               ("+ часы после ремонта, доля остановов за сутки", FEATURES + ["h_after_repair", "after_repair24", "stop_share24"]),
               ("+ всё медленное", FEATURES + SLOW)]:
    r2, r2t, mape, w10, *_ = lomo("dust_ok", fs)
    P(f"  {nm:48s} R2={r2:.3f}  строгая={r2t:.3f}  ошибка={mape:.1f}%  ≤10%: {w10:.0f}% часов")

P("\n=== 2. Пыль после ремонта: средний остаток текущей модели по часам после пуска ===")
cv = pd.read_csv(ROOT + "/artifacts/cv_predictions.csv", parse_dates=["ts"]).set_index("ts")
res = (cv.y - cv.linear)
h = X.set_index("ts")["h_after_repair"].reindex(res.index)
P((res.groupby(pd.cut(h, [0, 6, 12, 24, 48, 72.1])).agg(["mean", "count"]) * [1000, 1]).round(1).to_string())

P("\n=== 3. Поправка по последнему замеру (для монитора и прогноза на ближайшие часы) ===")
r = res.asfreq("h")
y = cv.y.asfreq("h"); p = cv.linear.asfreq("h")
for hz in (1, 3, 6, 12, 24):
    rho = r.autocorr(hz)
    corr = p + rho * r.shift(hz)
    m = corr.notna() & y.notna()
    e0 = ((y - p)[m].abs() / y[m]).mean() * 100; e1 = ((y - corr)[m].abs() / y[m]).mean() * 100
    P(f"  известен замер {hz:2d} ч назад: ошибка {e0:.1f}% -> {e1:.1f}%  (коэф. переноса {rho:.2f})")

P("\n=== 4. Другие вещества и расход: что объясняют процесс и погода ===")
PROC = PROCESS + WINDOW
for t, nm in [("flow_ok", "Расход газа"), ("NO_ok", "NO"), ("NO2_ok", "NO2"), ("SO2_ok", "SO2"), ("CO_ok", "CO"), ("O2_ok", "O2"), ("dust_ok", "Пыль (для сравнения)")]:
    a = lomo(t, PROC); b = lomo(t, WEATHER); c = lomo(t, FEATURES); g = lomo(t, FEATURES, lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0))
    s = X.loc[X.status.eq("OK"), t]
    P(f"  {nm:22s} R2: процесс={a[0]:6.2f}  погода={b[0]:6.2f}  процесс+погода={c[0]:6.2f} (строгая {c[1]:6.2f}, ошибка {c[2]:5.1f}%)  бустинг={g[0]:6.2f} | среднее {s.mean():.3g}, разброс {s.std()/s.mean()*100:.0f}%")

P("\n=== 5. Реагируют ли газы на остановы печи ===")
d = X[X.status.eq("OK")]
for t in ("flow_ok", "dust_ok", "NO_ok", "NO2_ok", "SO2_ok", "CO_ok", "O2_ok"):
    w, s = d.loc[d.power > 0, t].mean(), d.loc[d.power <= 0, t].mean()
    dd = d[t].diff(); dp = d.power.diff()
    P(f"  {t:8s} работа {w:9.3f}  останов {s:9.3f}  ({(s/w-1)*100:+5.1f}%)   corr(Δмощн, Δвещ)={dp.corr(dd):+.2f}")
