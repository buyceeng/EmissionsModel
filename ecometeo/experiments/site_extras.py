"""На погоде лучшего района: какие добавки ещё помогают."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
from _common import *
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
X0 = base_table()
WX = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
PW = PROCESS + WINDOW
SITES = ["ekaterinburg", "revda", "pervouralsk", "polevskoy"]
def table(site):
    o, d = site_weather(site)
    X = X0.drop(columns=WX + ["blh", "wind_dir"]).join(o, on="ts")
    e = d.reindex(X.ts)
    for c in ["dew_point_2m", "surface_pressure", "precipitation", "snowfall", "snow_depth", "cloud_cover", "cloud_cover_low",
              "wind_speed_100m", "wind_gusts_10m", "shortwave_radiation", "soil_temperature_0_to_7cm", "vapour_pressure_deficit",
              "et0_fao_evapotranspiration", "wx100m", "wy100m"]:
        X["e_" + c] = e[c].values
    T = d.temperature_2m
    for h in (6, 24, 72, 168):
        X[f"T_roll{h}"] = T.rolling(h, min_periods=1).mean().reindex(X.ts).values
    X["T_d24"] = (T - T.shift(24)).reindex(X.ts).values
    X["P_d3"] = (d.pressure_msl - d.pressure_msl.shift(3)).reindex(X.ts).values
    X["abs_hum"] = 6.112 * np.exp(17.67 * X.T_air / (X.T_air + 243.5)) * X.RH_air * 2.1674 / (273.15 + X.T_air)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * X.ts.dt.hour / 24), np.cos(2 * np.pi * X.ts.dt.hour / 24)
    X["weekend"] = (X.ts.dt.dayofweek >= 5).astype(float)
    X["wsx"], X["wsy"] = X.wind * X.wd_cos, X.wind * X.wd_sin        # вектор ветра (скорость × направление)
    X["T_pos"] = X.T_air.clip(lower=0)                                  # излом у 0 °C
    lom = [c for c in X0.columns if str(c).endswith("_lom")]
    for c in ["Известь_dsp", "Известь 2сорт_dsp", "Известь_lom", "Известь 2сорт_lom", "o2_dsp", "gas", "carbon_dsp"]:
        X["m_" + c] = X0[c].clip(lower=0).fillna(0).values
        X["m24_" + c] = X["m_" + c].rolling(24, min_periods=1).mean()
    return X
for s in SITES:
    X = table(s); B = PW + WX
    print(f"\n=== {s} ===")
    evaluate(X, B, name="база (признаки проекта)")
    evaluate(X, [c for c in B if c not in ("blh_log", "T_app")], name="без BLH и ощущ. t")
    for nm, add in {"+ ветер 100 м (вектор)": ["e_wx100m", "e_wy100m", "e_wind_speed_100m"], "+ вектор ветра (скор×напр)": ["wsx", "wsy"],
                    "+ порывы": ["e_wind_gusts_10m"], "+ точка росы": ["e_dew_point_2m"], "+ абс. влажность": ["abs_hum"],
                    "+ облачность": ["e_cloud_cover", "e_cloud_cover_low"], "+ радиация": ["e_shortwave_radiation"],
                    "+ почва": ["e_soil_temperature_0_to_7cm"], "+ снег": ["e_snow_depth", "e_snowfall"], "+ осадки": ["e_precipitation"],
                    "+ T среднее 6ч": ["T_roll6"], "+ T среднее 24ч": ["T_roll24"], "+ T среднее 72ч": ["T_roll72"], "+ T среднее 168ч": ["T_roll168"],
                    "+ ΔT за сутки, ΔP за 3ч": ["T_d24", "P_d3"], "+ время суток": ["hour_sin", "hour_cos"], "+ выходные": ["weekend"],
                    "+ излом T у 0°C": ["T_pos"], "+ VPD": ["e_vapour_pressure_deficit"],
                    "+ известь (час)": ["m_Известь_dsp", "m_Известь 2сорт_dsp", "m_Известь_lom", "m_Известь 2сорт_lom"],
                    "+ известь (сутки)": ["m24_Известь_dsp", "m24_Известь 2сорт_dsp", "m24_Известь_lom", "m24_Известь 2сорт_lom"],
                    "+ O2, газ, углерод в печь": ["m_o2_dsp", "m_gas", "m_carbon_dsp"]}.items():
        evaluate(X, B + add, name=nm)
    evaluate(X, B, lambda: ridge(10), name="Ridge α=10")
    evaluate(X, B, lambda: make_pipeline(StandardScaler(), HuberRegressor(alpha=1e-3, max_iter=500)), name="Huber (устойчивая к выбросам)")
    evaluate(X, B, lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1., random_state=0), name="Бустинг")
