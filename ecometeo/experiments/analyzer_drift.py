import sys, numpy as np, pandas as pd
ROOT, OUT = sys.argv[1], sys.argv[2]
out = open(OUT, "w", encoding="utf-8")
def P(*a): print(*a, file=out, flush=True)
X = pd.read_pickle(ROOT + "/artifacts/dataset.pkl").set_index("ts")
cv = pd.read_csv(ROOT + "/artifacts/cv_predictions.csv", parse_dates=["ts"]).set_index("ts")
res = (cv.y - cv.linear).reindex(X.index)
ok = X.status.eq("OK")
seg = (ok != ok.shift()).cumsum().where(ok)        # непрерывные участки работы анализатора
X["seg"] = seg; X["res"] = res
g = X[ok].groupby("seg")
info = g.agg(start=("res", lambda s: s.index[0]), hours=("res", "size"), mean_res=("res", "mean"), std_res=("res", "std"))
P(f"участков непрерывной работы анализатора: {len(info)}; медиана длины {info.hours.median():.0f} ч")
segmean = X.loc[ok, "seg"].map(info.mean_res)
P(f"доля дисперсии ошибки, объяснённая сдвигом уровня на участке: {segmean.var()/res[ok].var()*100:.0f}%")
P(info.sort_values("start").head(40).round(4).to_string())

# шум прибора: изменения пыли час к часу в спокойной работе печи
d = X[ok]
steady = (d.power > 50) & (d.power.shift(1) > 50) & (d.index.to_series().diff() == pd.Timedelta("1h"))
dd = (d.dust_ok.diff().abs() / d.dust_ok * 100)[steady]
P(f"\nизменение замера пыли за час при ровной работе печи: медиана {dd.median():.1f}%, среднее {dd.mean():.1f}%")
# ошибка модели в первые часы после включения анализатора vs позже
X["h_on"] = X.loc[ok].groupby("seg").cumcount() + 1
for a, b in [(1, 6), (7, 24), (25, 72), (73, 10000)]:
    m = ok & X.h_on.between(a, b)
    P(f"час после включения анализатора {a}-{b}: средняя |ошибка| {(res[m].abs()/X.dust_ok[m]).mean()*100:.1f}%, средний сдвиг {res[m].mean()*1000:+.1f}e-3, часов {m.sum()}")
# ошибка по суткам: насколько дни отличаются
day = res[ok].groupby(res[ok].index.floor("D")).mean()
P(f"\nсредний сдвиг по суткам: std {day.std()*1000:.1f}e-3 (≈{day.std()/X.dust_ok.mean()*100:.1f}% от средней пыли)")
P("самые «смещённые» сутки:\n" + (day.sort_values().iloc[[0,1,2,-3,-2,-1]] * 1000).round(1).to_string())
