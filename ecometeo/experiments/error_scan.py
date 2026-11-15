import sys, json, numpy as np, pandas as pd
ROOT, WXD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)
X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl").set_index("ts")
cv = pd.read_csv(ROOT + "/artifacts/cv_predictions.csv", parse_dates=["ts"]).set_index("ts")
res = (cv.y - cv.linear)
# все сырые числовые столбцы + почасовая погода (полная) + расширенная
H = json.load(open(WXD + "/hourly.json", encoding="utf-8"))["hourly"]; H = pd.DataFrame(H); H["time"] = pd.to_datetime(H.time); H = H.set_index("time")
E = json.load(open(WXD + "/extra.json", encoding="utf-8"))["hourly"]; E = pd.DataFrame(E); E["time"] = pd.to_datetime(E.time); E = E.set_index("time")
W = H.join(E).drop(columns=["wind_direction_10m"]).astype(float)
W["wd_sin"] = np.sin(np.deg2rad(H.wind_direction_10m)); W["wd_cos"] = np.cos(np.deg2rad(H.wind_direction_10m))
num = X.select_dtypes("number").drop(columns=[c for c in X.columns if c.endswith("_ok") or c in ("dust", "dust_g_h", "NO_g_h", "SO2_g_h", "CO_g_h")], errors="ignore")
num = num.loc[:, num.std() > 0]
ALL = num.join(W.add_prefix("w_"), how="left")
ok = res.index
# 1) почасово: корреляция остатка с каждым столбцом (и со сглаженным за 24 ч)
rows = []
for c in ALL.columns:
    s = ALL[c]
    r1 = res.corr(s.reindex(ok))
    r24 = res.corr(s.rolling(24, min_periods=6).mean().reindex(ok))
    r72 = res.corr(s.rolling(72, min_periods=12).mean().reindex(ok))
    rows.append((c, r1, r24, r72))
t = pd.DataFrame(rows, columns=["столбец", "r_час", "r_сутки", "r_3сут"]).set_index("столбец")
t["max"] = t.abs().max(axis=1)
P("Что связано с ошибкой модели (порог случайности ~0.05 по часам, ~0.2 по суткам):")
P(t.sort_values("max", ascending=False).head(30).round(3).to_string())
# 2) суточно
dres = res.groupby(res.index.floor("D")).mean()
dall = ALL.groupby(ALL.index.floor("D")).mean().reindex(dres.index)
dc = dall.corrwith(dres).dropna()
P("\nПо суткам (средний остаток за сутки ↔ среднее за сутки), топ-20:")
P(dc.reindex(dc.abs().sort_values(ascending=False).index).head(20).round(3).to_string())
P(f"(суток: {len(dres)})")
