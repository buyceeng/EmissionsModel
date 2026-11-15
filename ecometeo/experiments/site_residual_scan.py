"""Что осталось в ошибке после перехода на погоду верной площадки."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
SITE = sys.argv[1] if len(sys.argv) > 1 else "pervouralsk"
X0 = base_table()
WX = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
o, d = site_weather(SITE); X = X0.drop(columns=WX).join(o, on="ts")
X["vpd"] = 0.6108 * np.exp(17.27 * X.T_air / (X.T_air + 237.3)) * (1 - X.RH_air / 100)
F = PROCESS + WINDOW + ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "vpd"]
r = evaluate(X, F, name=f"{SITE}: ядро+BLH+VPD (расчётный)")
res = (X.dust_ok - r["pred"]); X["res"] = res
s = res.copy(); s.index = X.ts; s = s.asfreq("h")
print("автокорреляция ошибки:", " ".join(f"лаг{k}:{s.autocorr(k):.2f}" for k in (1, 2, 3, 6, 12, 24, 48)))
day = X.ts.dt.floor("D"); dm = res.groupby(day).mean(); print(f"доля суточного сдвига в дисперсии ошибки: {dm.reindex(day).values[res.notna()].var() / res.var():.0%}; std сут. сдвига {dm.std():.4f}")
print("ошибка по месяцам, %:", (res.abs() / X.dust_ok * 100).groupby(X.ts.dt.month).mean().round(2).to_dict())
num = [c for c in X0.columns if X0[c].dtype.kind == "f" and c not in ("dust", "dust_ok", "dust_g_h")]
E = d.reindex(X.ts).add_prefix("w_"); E.index = X.index
Z = pd.concat([X0[num].clip(lower=0), E], axis=1)
ok = res.notna()
rows = []
for c in Z.columns:
    z = Z[c]
    if z[ok].std() == 0 or z[ok].notna().sum() < 500: continue
    zd = z.groupby(day).mean(); n_d = ok.groupby(day).sum()
    good = n_d[n_d >= 12].index
    rows.append({"столбец": c, "r_час": np.corrcoef(z[ok].fillna(0), res[ok])[0, 1], "r_сутки": zd[good].corr(dm[good]),
                 "r_сутки_ранг": zd[good].corr(dm[good], method="spearman")})
t = pd.DataFrame(rows).set_index("столбец"); t["max"] = t.abs().max(axis=1)
print(t.sort_values("max", ascending=False).head(25).round(3).to_string())
