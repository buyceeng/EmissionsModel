import sys, numpy as np, pandas as pd
ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)
X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl")
cv = pd.read_csv(ROOT + "/artifacts/cv_predictions.csv", parse_dates=["ts"])
d = X.set_index("ts")
cv = cv.set_index("ts")
cv["res"] = cv.y - cv.linear

# 1. the March week hour by hour (every 3h)
wk = cv.loc["2022-03-14":"2022-03-20 23:00"].join(d[["power", "scrap_t", "mode", "T_air", "wind", "wind_dir", "flow_ok", "status"]])
P("=== неделя 14–20.03, каждые 3 ч ===")
P(wk.iloc[::3][["y", "linear", "res", "power", "scrap_t", "mode", "T_air", "wind_dir", "flow_ok"]].round(3).to_string())
# full timeline around 16.03 evening and 18–19.03
P("\n=== 16.03 14:00 – 17.03 06:00 по часам ===")
w2 = d.loc["2022-03-16 14:00":"2022-03-17 06:00", ["power", "scrap_t", "mode", "status", "dust_ok", "T_air"]].join(cv[["linear", "res"]])
P(w2.round(3).to_string())
P("\n=== 18.03 10:00 – 19.03 06:00 ===")
w3 = d.loc["2022-03-18 10:00":"2022-03-19 06:00", ["power", "scrap_t", "mode", "status", "dust_ok", "T_air", "wind_dir", "flow_ok"]].join(cv[["linear", "res"]])
P(w3.round(3).to_string())

# 2. error structure: autocorrelation & slow vs fast
r = cv.res.asfreq("h")
P("\n=== автокорреляция ошибки (только годные часы подряд) ===")
P(" ".join(f"лаг{k}:{r.autocorr(k):.2f}" for k in (1, 2, 3, 6, 12, 24, 48)))
day = cv.res.groupby(cv.index.floor("D")).transform("mean")
P(f"дисперсия ошибки: всего {cv.res.var()*1e6:.1f}e-6; медленная (среднее за сутки) {day.var()*1e6:.1f}e-6 = {day.var()/cv.res.var()*100:.0f}%; быстрая (внутри суток) {(cv.res-day).var()*1e6:.1f}e-6")

# 3. how dust reacts to stops: measured vs model during stop hours
s = d.join(cv[["linear"]])
stop = (s.power <= 0) & s.dust_ok.notna()
work = (s.power > 0) & s.dust_ok.notna()
P(f"\nпыль замер: работа {s.dust_ok[work].mean():.4f}, останов {s.dust_ok[stop].mean():.4f}  | модель: работа {s.linear[work].mean():.4f}, останов {s.linear[stop].mean():.4f}")
# event study around stops start
starts = s.index[(s.power <= 0) & (s.power.shift(1) > 0)]
rows = []
for t in starts:
    seg = s.loc[t - pd.Timedelta("3h"): t + pd.Timedelta("6h")]
    if len(seg) == 10 and seg.dust_ok.notna().all():
        base_m, base_p = seg.dust_ok.iloc[:3].mean(), seg.linear.iloc[:3].mean()
        rows.append(np.r_[seg.dust_ok.values - base_m, seg.linear.values - base_p, seg.power.values])
if rows:
    a = np.array(rows)
    P(f"остановов с полными замерами: {len(a)}")
    P("час от начала останова: " + " ".join(f"{k:+d}" for k in range(-3, 7)))
    P("замер, Δ от до-останова: " + " ".join(f"{v*1000:+5.1f}" for v in a[:, :10].mean(0)))
    P("модель, Δ:               " + " ".join(f"{v*1000:+5.1f}" for v in a[:, 10:20].mean(0)))
    P("мощность:                " + " ".join(f"{v:5.1f}" for v in a[:, 20:].mean(0)))
