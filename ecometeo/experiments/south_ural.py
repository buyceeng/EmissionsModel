"""Погода только Южного Урала (требование эксперта): какой город и какой набор признаков лучше."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X0 = base_table()
OLD = [c for c in ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app", "vpd", "blh", "wind_dir"] if c in X0]
CORE = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos"]
PW = PROCESS + WINDOW
SETS = {"ядро": CORE, "ядро+BLH": CORE + ["blh_log"], "ядро+BLH+ощущ. t (прежняя версия)": CORE + ["blh_log", "T_app"],
        "ядро+BLH+дефицит влажности": CORE + ["blh_log", "vpd"], "ядро+BLH+ощущ. t+дефицит влажности": CORE + ["blh_log", "T_app", "vpd"]}
SOUTH = ["chelyabinsk", "zlatoust", "miass", "satka", "asha", "troitsk", "magnitogorsk", "beloretsk", "novotroitsk"]


def table(o):
    X = X0.drop(columns=OLD).join(o, on="ts")
    X["vpd"] = 0.6108 * np.exp(17.27 * X.T_air / (X.T_air + 237.3)) * (1 - X.RH_air / 100)
    return X


W = {s: site_weather(s)[0] for s in SOUTH}
for s in SOUTH:
    print(f"\n=== {s} ===")
    for nm, f in SETS.items():
        evaluate(table(W[s]), PW + f, name=nm)
for nm, group in {"среднее по Челябинской области (Челябинск, Златоуст, Миасс, Сатка, Аша, Троицк, Магнитогорск)": SOUTH[:7],
                  "среднее Челябинск+Златоуст+Миасс": SOUTH[:3]}.items():
    o = sum(W[s] for s in group) / len(group)
    n = np.hypot(o.wd_cos, o.wd_sin).replace(0, 1); o["wd_cos"], o["wd_sin"] = o.wd_cos / n, o.wd_sin / n
    print(f"\n=== {nm} ===")
    for k, f in SETS.items():
        evaluate(table(o), PW + f, name=k)
