"""Остановы печи: расписание, расход газа и «базовый» уровень пыли при нулевом расходе."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X = base_table(); ok = X.status.eq("OK")
st = X.power <= 0; rid = (st != st.shift()).cumsum()
runs = X[st].groupby(rid[st]).agg(start=("ts", "first"), hours=("ts", "size"), flow_min=("flow", "min"), flow_mean=("flow", "mean"), ok=("status", lambda s: (s == "OK").mean()))
runs["weekday"] = runs.start.dt.dayofweek; runs["hour"] = runs.start.dt.hour
long = runs[runs.hours >= 4]
print("=== 1. Остановы ≥ 4 ч за год:", len(long), "===")
print("день недели начала (0=пн):", long.weekday.value_counts().sort_index().to_dict(), "| час начала:", long.hour.value_counts().sort_index().to_dict())
print("интервал между остановами ≥ 4 ч, сут:", (long.start.diff().dt.total_seconds() / 86400).round(0).value_counts().sort_index().to_dict())
print(long.assign(start=long.start.dt.strftime("%m-%d %a %H:00")).round(0).to_string(index=False))

print("\n=== 2. Что показывает прибор, когда расход газа почти нулевой (годные часы) ===")
Xr = X[ok].copy(); day = Xr.ts.dt.floor("D")
ref = Xr[Xr.power > 50].groupby(day).agg(dust=("dust", "median"), NO=("NO", "median"), SO2=("SO2", "median"), CO=("CO", "median"), O2=("O2", "median"), flow=("flow", "median"))
for nm, g in {"останов, расход < 50": Xr[(Xr.power <= 0) & (Xr.flow < 50)], "останов, расход ≥ 250": Xr[(Xr.power <= 0) & (Xr.flow >= 250)]}.items():
    r = ref.reindex(g.ts.dt.floor("D").values)
    line = "  ".join(f"{c} {100 * np.nanmedian(g[c].values / r[c].values):.0f}%" for c in ["flow", "dust", "NO", "SO2", "CO", "O2"])
    print(f"{nm:24s} n={len(g):3d}  в % от медианы рабочих часов тех же суток: {line}")
g = Xr[(Xr.power <= 0) & (Xr.flow < 50)]
print("при расходе < 50: корреляция пыли с температурой воздуха r =", round(g.dust.corr(g.T_air), 2), "; пыль от", g.dust.min().round(3), "до", g.dust.max().round(3))
k = np.polyfit(g.T_air, g.dust, 1); k2 = np.polyfit(Xr[Xr.power > 50].T_air, Xr[Xr.power > 50].dust, 1)
print(f"наклон пыли по температуре: при нулевом расходе {k[0]*1000:.2f}e-3 на °C, при работе печи {k2[0]*1000:.2f}e-3 на °C")
