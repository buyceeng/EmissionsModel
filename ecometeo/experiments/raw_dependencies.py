"""Новые зависимости в исходных данных (погода — только Южный Урал).
A. Канал пыли в часы сбоя газоанализатора.  B. Производные показатели процесса.  C. Погода по региону."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
from numpy.linalg import lstsq
X = base_table()
X["pred"] = pd.read_csv(ROOT / "artifacts" / "monitor.csv").dust_linear.values


def slopes(d, y="dust"):
    """dust = a + b*T + c*power: наклоны по температуре и мощности."""
    A = np.c_[np.ones(len(d)), d.T_air, d.power]; k = lstsq(A, d[y].values, rcond=None)[0]
    return k[1] * 1000, k[2] * 1000


print("=== A. Пыль в часы сбоя газоанализатора (FAULT, в обучении не участвуют) ===")
ok, f = X[X.status == "OK"], X[X.status == "FAULT"]
for nm, d in {"годные, июнь–июль": ok[ok.ts.dt.month >= 6], "годные, все": ok, "сбой, авг–сен": f[f.ts.dt.month >= 8]}.items():
    bT, bP = slopes(d)
    k = d.index[np.isin(d.index - 1, d.index)]
    dd = pd.DataFrame({"dp": X.power[k].values - X.power[k - 1].values, "dd": X.dust[k].values - X.dust[k - 1].values, "dT": X.T_air[k].values - X.T_air[k - 1].values})
    print(f"{nm:20s} n={len(d):4d} пыль ср. {d.dust.mean():.4f}  наклон по T {bT:.2f}e-3/°C  по мощности {bP:.3f}e-3/МВт  "
          f"corr(Δмощн,Δпыль)={dd.dp.corr(dd.dd):+.2f}  corr(ΔT,Δпыль)={dd.dT.corr(dd.dd):+.2f}  расход ср. {d.flow.mean():.0f}")
e = f.dust - f.pred
print(f"модель (обучена на янв–июл) на часах сбоя: ошибка {(e.abs() / f.dust * 100).mean():.1f} %, сдвиг {e.mean() / f.dust.mean() * 100:+.1f} %, r={f.dust.corr(f.pred):.2f}")
for nm, g in {"расход < 250 (прибор в «половинном» режиме)": f[f.flow < 250], "расход ≥ 250": f[f.flow >= 250]}.items():
    e = g.dust - g.pred; print(f"   {nm:45s} n={len(g):3d} ошибка {(e.abs() / g.dust * 100).mean():.1f} %  сдвиг {e.mean() / g.dust.mean() * 100:+.1f} %")
st = f[(f.power <= 0)]; print("остановы печи внутри часов сбоя:", len(st), "ч; пыль в них", st.dust.mean().round(4), "против", f[f.power > 50].dust.mean().round(4), "при работе")

print("\n=== B. Производные показатели процесса (погода Челябинска, прежние признаки) ===")
lom = [c for c in X.columns if str(c).endswith("_lom")]; dsp = [c for c in X.columns if str(c).endswith("_dsp") and c not in ("timedelta_dsp", "electric_power_dsp", "o2_dsp", "carbon_dsp")]
M = X[lom + dsp].clip(lower=0)
grp = {"чугун": "угун", "известь": "звесть", "брикеты": "рикет", "стружка_пакеты": "тружка|пакет", "ферросплавы": "Mn|ФС|FeCr|ФХ|ФМН|Ферро|Мн |Ni|Никель|Мо",
       "углерод": "УСМ|Графит|КАУР|углерод", "алюминий": "люмин|АВ", "скрап": "крап"}
for g, pat in grp.items():
    X["g_" + g] = M.loc[:, M.columns.str.contains(pat)].sum(axis=1) / 1000
X["kwh_t"] = (X.power / X.scrap_t.replace(0, np.nan)).clip(upper=2).fillna(0)          # МВт·ч на тонну лома
X["o2_t"] = (X.o2_dsp.clip(lower=0) / X.scrap_t.replace(0, np.nan)).fillna(0).clip(upper=X.o2_dsp.max())
X["idle_min"] = 60 - X.td_min                                                            # минуты без тока в часе
X["n_types"] = (M[lom] > 0).sum(axis=1)                                                   # сколько видов шихты за час
for c in ["o2_dsp", "gas", "carbon_dsp"]:
    X["p_" + c] = X[c].clip(lower=0).fillna(0)
for c in ["power", "scrap_t", "td_min"] + ["g_" + g for g in grp]:
    X[c + "_m6"] = X[c].rolling(6, min_periods=1).mean(); X[c + "_m24"] = X[c].rolling(24, min_periods=1).mean()
X["stops24"] = (X.power <= 0).rolling(24, min_periods=1).mean(); X["stops72"] = (X.power <= 0).rolling(72, min_periods=1).mean()
X["hrs_since_stop"] = np.log1p(X.cycle_hour.clip(upper=200))
hol = pd.to_datetime(["2022-01-0%d" % d for d in range(1, 10)] + ["2022-02-23", "2022-03-06", "2022-03-07", "2022-03-08", "2022-05-01", "2022-05-02", "2022-05-03",
                      "2022-05-07", "2022-05-08", "2022-05-09", "2022-05-10", "2022-06-11", "2022-06-12", "2022-06-13"])
X["holiday"] = (X.ts - pd.Timedelta("1h")).dt.floor("D").isin(hol).astype(float)
X["TxP"] = X.T_air * X.power; X["Txstop"] = X.T_air * X.is_stop; X["T2"] = X.T_air ** 2
base = evaluate(X, FEATURES, name="текущая модель")
tests = {"+ группы материалов (час)": ["g_" + g for g in grp], "+ группы материалов (среднее 6 ч)": ["g_" + g + "_m6" for g in grp],
         "+ группы материалов (среднее 24 ч)": ["g_" + g + "_m24" for g in grp], "+ МВт·ч на тонну, O2 на тонну": ["kwh_t", "o2_t"],
         "+ кислород, газ, углерод в печь": ["p_o2_dsp", "p_gas", "p_carbon_dsp"], "+ число видов шихты": ["n_types"],
         "+ средние за 6 и 24 ч: мощность, лом, минуты": [c + s for c in ("power", "scrap_t", "td_min") for s in ("_m6", "_m24")],
         "+ доля остановов за 24 и 72 ч": ["stops24", "stops72"], "+ лог часов от пуска": ["hrs_since_stop"], "+ праздничные дни (произв. календарь)": ["holiday"],
         "+ T×мощность, T×останов": ["TxP", "Txstop"], "+ T²": ["T2"]}
for nm, add in tests.items():
    evaluate(X, FEATURES + add, name=nm)
for g in grp:
    evaluate(X, FEATURES + ["g_" + g, "g_" + g + "_m6", "g_" + g + "_m24"], name=f"   только «{g}» (час, 6 ч, 24 ч)")

print("\n=== C. Погода по региону Южный Урал (Open-Meteo / ERA5) ===")
W = {s: site_weather(s)[0] for s in ["chelyabinsk", "zlatoust", "miass", "satka", "asha", "troitsk", "magnitogorsk", "beloretsk", "novotroitsk"]}
WXC = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
def reg(names):
    o = sum(W[s] for s in names) / len(names); n = np.hypot(o.wd_cos, o.wd_sin).replace(0, 1); o["wd_cos"], o["wd_sin"] = o.wd_cos / n, o.wd_sin / n
    return o
X0 = X.drop(columns=WXC)
ALL = list(W); CHEL = ALL[:7]
evaluate(X0.join(W["chelyabinsk"], on="ts"), FEATURES, name="Челябинск (одна точка)")
evaluate(X0.join(reg(CHEL), on="ts"), FEATURES, name="среднее по 7 городам Челябинской области")
evaluate(X0.join(reg(ALL), on="ts"), FEATURES, name="среднее по 9 городам Южного Урала")
for extra in (["zlatoust"], ["zlatoust", "magnitogorsk"], ["zlatoust", "magnitogorsk", "troitsk", "asha"]):
    Z = X0.join(reg(ALL), on="ts")
    cols = []
    for s in extra:
        Z["T_" + s] = W[s].T_air.reindex(Z.ts).values - Z.T_air; cols.append("T_" + s)
    evaluate(Z, FEATURES + cols, name="среднее по региону + отклонение T: " + ", ".join(extra))
    evaluate(Z, FEATURES + cols, lambda: ridge(30), name="   то же, Ridge α=30")
