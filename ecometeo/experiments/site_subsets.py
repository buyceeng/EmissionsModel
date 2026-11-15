import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
X0 = base_table()
WX = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
PW = PROCESS + WINDOW
for s in ["ekaterinburg", "revda", "pervouralsk", "polevskoy"]:
    o, d = site_weather(s); X = X0.drop(columns=WX).join(o, on="ts")
    X["vpd"] = d.vapour_pressure_deficit.reindex(X.ts).values
    X["wsx"], X["wsy"] = X.wind * X.wd_cos, X.wind * X.wd_sin
    print("\n===", s)
    core = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos"]
    for nm, f in {"ядро": core, "ядро+BLH": core + ["blh_log"], "ядро+T_app": core + ["T_app"], "ядро+BLH+T_app (проект)": WX,
                  "ядро+BLH+VPD": core + ["blh_log", "vpd"], "ядро+VPD": core + ["vpd"], "ядро+BLH+вектор ветра": core + ["blh_log", "wsx", "wsy"],
                  "ядро+BLH+VPD+вектор": core + ["blh_log", "vpd", "wsx", "wsy"], "проект+VPD+вектор": WX + ["vpd", "wsx", "wsy"]}.items():
        evaluate(X, PW + f, name=nm)
