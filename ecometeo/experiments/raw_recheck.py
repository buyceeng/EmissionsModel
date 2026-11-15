"""Повторный разбор исходного data.xlsx: статусы прибора, «замёрзшие» значения, связи каналов."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
pd.set_option("display.width", 250)
X = base_table()
m = X.ts.dt.month
print("=== 1. Статус прибора по месяцам ===")
print(pd.crosstab(m, X.status).to_string())

print("\n=== 2. Часы со статусом FAULT: что именно сломано и жив ли канал пыли ===")
f = X[X.status == "FAULT"]
cond = {"газ>5000": (f[["NO", "SO2", "CO", "NO2"]] > 5000).any(axis=1), "O2>25": f.O2 > 25, "O2<5": f.O2 < 5, "H2O>100": f.H2O > 100}
print({k: int(v.sum()) for k, v in cond.items()})
print(f[["flow", "dust", "NO", "SO2", "CO", "NO2", "H2O", "O2"]].describe(percentiles=[.05, .5, .95]).round(3).to_string())
print("по месяцам: медиана пыли в FAULT / в OK")
print(pd.DataFrame({"fault_n": f.groupby(f.ts.dt.month).size(), "fault_dust_med": f.groupby(f.ts.dt.month).dust.median(),
                    "ok_dust_med": X[X.status == "OK"].groupby(m).dust.median()}).round(4).to_string())

print("\n=== 3. Годные часы: описательная статистика каналов ===")
ok = X[X.status == "OK"]
print(ok[["flow", "dust", "NO", "SO2", "CO", "NO2", "H2O", "O2"]].describe(percentiles=[.01, .05, .5, .95, .99]).round(4).to_string())

print("\n=== 4. «Замёрзшие» значения: доля часов, где значение точно равно предыдущему ===")
for c in ["flow", "dust", "NO", "SO2", "CO", "NO2", "H2O", "O2"]:
    same = (X[c] == X[c].shift()) & X.status.eq("OK")
    run = same.groupby((~same).cumsum()).cumsum()
    print(f"{c:5s} повторов {same.sum():5d}  макс. серия {int(run.max())}")

print("\n=== 5. Корреляции каналов прибора (годные часы): уровни / изменения за час ===")
C = ["dust", "flow", "NO", "SO2", "CO", "NO2", "H2O", "O2", "power", "T_air"]
print(ok[C].corr().round(2).to_string())
d = X[C].diff()[X.status.eq("OK") & X.status.shift().eq("OK")]
print(d.corr().round(2).to_string())

print("\n=== 6. Корреляции по Спирмену пыли с каналами внутри каждого месяца ===")
print(pd.DataFrame({k: g[C].corr(method="spearman")["dust"] for k, g in ok.groupby(ok.ts.dt.month)}).round(2).to_string())

print("\n=== 7. Пыль: уровни по месяцам, по часу суток, по дню недели ===")
print(ok.groupby(ok.ts.dt.month).dust.agg(["mean", "std", "min", "max"]).round(4).to_string())
print("час суток (отклонение от среднего суток, %):", ((ok.dust / ok.groupby(ok.ts.dt.floor("D")).dust.transform("mean") - 1) * 100).groupby(ok.ts.dt.hour).mean().round(1).to_dict())
print("день недели (0=пн), отклонение от скользящей 15-дн. нормы, %:", ((ok.dust / ok.set_index("ts").dust.rolling("15D").mean().values - 1) * 100).groupby(ok.ts.dt.dayofweek).mean().round(1).to_dict())
