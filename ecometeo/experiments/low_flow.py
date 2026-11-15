"""Годные часы с аномально низким расходом газа: верить ли пыли в эти часы."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X = base_table(); r = evaluate(X, FEATURES, name="текущая модель, все годные часы")
X["res"] = X.dust_ok - r["pred"]; ok = X.status.eq("OK")
print("расход в годные часы, квантили:", X.flow[ok].quantile([.005, .01, .02, .03, .05, .1]).round(0).to_dict())
for lo, hi in ((0, 50), (50, 150), (150, 250), (250, 350), (350, 600)):
    g = X[ok & X.flow.between(lo, hi)]
    print(f"расход {lo:3d}–{hi:3d}: n={len(g):4d}  печь стоит {100 * (g.power <= 0).mean():3.0f}%  ошибка {(g.res.abs() / g.dust * 100).mean():5.1f}%  сдвиг {g.res.mean() / g.dust.mean() * 100:+5.1f}%")
low = ok & (X.flow < 50)
print("часы с расходом < 50 при работающей печи:", int((low & (X.power > 30)).sum()), "; даты:", sorted(set(X.ts[low].dt.strftime("%m-%d")))[:40])
Z = X.copy(); Z.loc[low, "status"] = "LOWFLOW"
evaluate(Z, FEATURES, name="без часов с расходом < 50")
X["TxP"], X["Txstop"] = X.T_air * X.power, X.T_air * X.is_stop
evaluate(X, FEATURES + ["TxP", "Txstop"], name="+ T×мощность, T×останов")
evaluate(X, FEATURES + ["TxP"], name="+ T×мощность")
evaluate(X, FEATURES + ["Txstop"], name="+ T×останов")
