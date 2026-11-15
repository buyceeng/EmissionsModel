import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT, WXD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ROOT)
from ecometeo.features import FEATURES
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)
X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl").copy()
def hourly(p):
    w = pd.DataFrame(json.load(open(p, encoding="utf-8"))["hourly"]); w["time"] = pd.to_datetime(w.time)
    w = w.set_index("time").astype(float); return (w + w.shift(1)) / 2
H = hourly(WXD + "/hourly.json"); E = hourly(WXD + "/extra.json")
X["blh_log"] = np.log(E.boundary_layer_height.reindex(X.ts).clip(lower=10).values)
X["app_T"] = E.apparent_temperature.reindex(X.ts).values
X["soil_T"] = E.soil_temperature_0_to_7cm.reindex(X.ts).values
X["sun"] = E.sunshine_duration.reindex(X.ts).values / 3600
X["cloud"] = H.cloud_cover.reindex(X.ts).values
X["cloud_low"] = E.cloud_cover_low.reindex(X.ts).values
X["snowfall"] = H.snowfall.reindex(X.ts).values
T, RH = X.T_air, X.RH_air
X["abs_hum"] = 6.112 * np.exp(17.67 * T / (T + 243.5)) * RH * 2.1674 / (273.15 + T)
lime = {"Известь_dsp": "lime1", "Известь 2сорт_dsp": "lime2", "Известь 2сорт_lom": "lime2_lom"}
for c, n in lime.items():
    X[n] = X[c].clip(lower=0) / 1000
    X[n + "_24"] = X[n].rolling(24, min_periods=1).mean()
    X[n + "_72"] = X[n].rolling(72, min_periods=1).mean()
for c in ("sun", "cloud", "cloud_low", "snowfall"):
    X[c + "_24"] = X[c].rolling(24, min_periods=1).mean()

def ridge(a=1.0): return make_pipeline(StandardScaler(), Ridge(alpha=a))
def ev(feats, name, a=1.0):
    ok = X.status.eq("OK") & X.dust_ok.notna() & X[feats].notna().all(axis=1)
    d = X[ok]; y = d.dust_ok; m = d.ts.dt.month
    p = pd.Series(index=d.index, dtype=float)
    for k in sorted(m.unique()):
        p[m == k] = ridge(a).fit(d.loc[m != k, feats], y[m != k]).predict(d.loc[m == k, feats])
    tr, te = m <= 4, m >= 5
    p2 = ridge(a).fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    ape = (y - p).abs() / y * 100
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(); r2t = 1 - ((y[te] - p2) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    # стабильность по половинам: вторая проверка — июн-июл -> ян-апр
    tr2, te2 = m >= 5, m <= 4
    p3 = ridge(a).fit(d.loc[tr2, feats], y[tr2]).predict(d.loc[te2, feats])
    r2b = 1 - ((y[te2] - p3) ** 2).sum() / ((y[te2] - y[te2].mean()) ** 2).sum()
    P(f"{name:58s} ошибка={ape.mean():5.2f}%  ≤10%={(ape<=10).mean()*100:3.0f}%  R2={r2:.3f}  строгая: зима→лето {r2t:.3f}, лето→зима {r2b:.3f}")
    return ape.mean()
B = FEATURES
WB = ["blh_log", "app_T", "soil_T", "abs_hum"]
ev(B, "текущая модель")
ev(B + WB, "+ пограничный слой, ощущ. t, почва, абс. влажность")
for grp, fs in [("известь (час)", ["lime1", "lime2", "lime2_lom"]), ("известь (сутки)", ["lime1_24", "lime2_24", "lime2_lom_24"]),
                ("известь (3 сут)", ["lime1_72", "lime2_72", "lime2_lom_72"]), ("облака/солнце/снег (час)", ["sun", "cloud", "cloud_low", "snowfall"]),
                ("облака/солнце/снег (сутки)", ["sun_24", "cloud_24", "cloud_low_24", "snowfall_24"])]:
    ev(B + fs, f"+ {grp}")
BEST = B + WB + ["lime1_24", "lime2_24", "lime2_lom_24", "cloud_24", "sun_24", "snowfall_24"]
ev(BEST, "+ погода расш. + известь (сутки) + облака/солнце/снег (сутки)")
ev(BEST, "  то же, α=5", 5)
ev(B + WB + ["cloud_24", "sun_24", "snowfall_24"], "+ погода расш. + облака/солнце/снег (сутки)")
ev(B + WB + ["lime1_24", "lime2_24", "lime2_lom_24"], "+ погода расш. + известь (сутки)")
# коэффициенты извести
ok = X.status.eq("OK") & X.dust_ok.notna()
m = ridge().fit(X.loc[ok, BEST], X.loc[ok, "dust_ok"])
co = dict(zip(BEST, m[-1].coef_ / m[0].scale_))
P({k: round(v * 1000, 3) for k, v in co.items() if k.startswith(("lime", "cloud", "sun", "snow", "blh", "app", "soil", "abs"))})
P("средняя добавка извести, т/ч (сутки):", X.loc[ok, ["lime1_24", "lime2_24", "lime2_lom_24"]].mean().round(3).to_dict())
P("\n=== набор, доступный онлайн ===")
ev(B + ["blh_log"], "+ пограничный слой")
ev(B + ["blh_log", "app_T"], "+ пограничный слой, ощущ. t")
ev(B + ["blh_log", "app_T", "abs_hum"], "+ пограничный слой, ощущ. t, абс. влажность")
