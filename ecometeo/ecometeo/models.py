"""Модели пыли: линейная (основная) и нейросеть (ансамбль MLP) для сравнения.

Почему основная линейная: на честной проверке нейросеть, бустинг, лес,
сплайны и попарные взаимодействия не обогнали её (см. experiments/).
Зависимость пыли от работы печи и погоды линейная, а коэффициенты
линейной модели эколог может проверить сам.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, TARGET

# Понятные названия признаков для таблицы коэффициентов
FEATURE_RU = {
    "power": "мощность, на 1 МВт·ч", "td_min": "минуты под током, на 1 мин", "scrap_t": "лом, на 1 т",
    "is_stop": "печь стоит (1/0)", "first3": "первые 3 ч цикла (1/0)", "last1": "последний час цикла (1/0)",
    "T_air": "температура воздуха, на 1 °C", "RH_air": "влажность, на 1 %", "P_air": "давление, на 1 гПа",
    "wind": "скорость ветра, на 1 км/ч", "wd_sin": "ветер: восточная составляющая", "wd_cos": "ветер: северная составляющая",
    "blh_log": "слой перемешивания: на каждое удвоение высоты", "T_app": "ощущаемая температура, на 1 °C",
    "TxP": "температура × мощность, на 1 °C·МВт·ч",
}
for _k in (1, 2, 3):
    FEATURE_RU.update({f"power_l{_k}": f"мощность {_k} ч назад, на 1 МВт·ч",
                       f"td_min_l{_k}": f"минуты под током {_k} ч назад, на 1 мин",
                       f"scrap_t_l{_k}": f"лом {_k} ч назад, на 1 т"})


class MLPEnsemble:
    def __init__(self, n_models: int = 5, hidden=(8,), alpha: float = 0.3, seed: int = 0):
        self.models = [
            make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", alpha=alpha,
                             max_iter=1500, early_stopping=True, random_state=seed + k),
            )
            for k in range(n_models)
        ]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def make_linear():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


@dataclass
class DustModel:
    """Линейная модель (основная) + нейросеть для сравнения + разброс ошибки для тревог."""
    linear: object = field(default_factory=make_linear)
    nn: MLPEnsemble = field(default_factory=MLPEnsemble)
    resid_std: float = float("nan")
    features: list = field(default_factory=lambda: list(FEATURES))
    ranges: dict = field(default_factory=dict)     # диапазон признаков в обучении
    metrics: dict = field(default_factory=dict)
    ar_rho: float = 0.93

    def correction(self, last_resid, hours):
        """Добавка к прогнозу: последняя ошибка (замер − модель), затухающая за hours часов."""
        return last_resid * self.ar_rho ** np.asarray(hours, dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DustModel":
        Xf = X[self.features]
        self.linear.fit(Xf, y)
        self.nn.fit(Xf, y)
        self.resid_std = float(np.std(y - self.linear.predict(Xf)))
        self.ranges = {c: (float(Xf[c].min()), float(Xf[c].max())) for c in self.features}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Прогноз только для часов, где известны все признаки (иначе NaN)."""
        Xf = X[self.features]
        full = Xf.notna().all(axis=1)
        out = pd.DataFrame(np.nan, index=X.index, columns=["dust_linear", "dust_nn"])
        if full.any():
            out.loc[full, "dust_linear"] = self.linear.predict(Xf[full])
            out.loc[full, "dust_nn"] = self.nn.predict(Xf[full])
        return out

    def coefficients(self) -> pd.DataFrame:
        """Коэффициенты линейной модели в исходных единицах: на сколько мг/Нм³
        меняется пыль при росте признака на 1."""
        scaler, ridge = self.linear[0], self.linear[-1]
        coef = ridge.coef_ / scaler.scale_
        coef = np.where(np.array(self.features) == "blh_log", coef * np.log(2), coef)   # на удвоение высоты
        t = pd.DataFrame({"признак": [FEATURE_RU.get(f, f) for f in self.features], "мг/Нм³": coef})
        ws, wc = dict(zip(self.features, coef))["wd_sin"], dict(zip(self.features, coef))["wd_cos"]
        t = t[~pd.Series(self.features).isin(["wd_sin", "wd_cos"]).values]
        wind = pd.DataFrame({"признак": [f"ветер: больше пыли при ветре из ~{np.degrees(np.arctan2(ws, wc)) % 360:.0f}° "
                                         f"(разница между самым «пыльным» и противоположным направлением)"],
                             "мг/Нм³": [2 * np.hypot(ws, wc)]})
        return pd.concat([t, wind], ignore_index=True)

    def out_of_range(self, X: pd.DataFrame, cols=("T_air", "RH_air", "P_air", "wind", "power")) -> pd.Series:
        """True, если час выходит за диапазон обучения (прогноз — экстраполяция):
        погода, которой не было в обучении, или мощность печи выше максимума 2022 г."""
        bad = pd.Series(False, index=X.index)
        for c in cols:
            lo, hi = self.ranges.get(c, (-np.inf, np.inf))
            bad |= (X[c] < lo) | (X[c] > hi)
        return bad

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "DustModel":
        return joblib.load(path)


def _scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "r": float(np.corrcoef(y, p)[0, 1]),
        "R2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "MAE": float(np.abs(y - p).mean()),
    }


def evaluate_leave_one_month_out(X: pd.DataFrame, rows: pd.Series, n_models: int = 5) -> pd.DataFrame:
    """Честная проверка: учимся на всех месяцах, кроме одного, проверяем на нём.

    Плюс строгая проверка «R2_янв-апр→май-июл»: обучение на холодных месяцах,
    проверка на тёплых (модель должна выйти за знакомые температуры).
    Возвращает (таблица метрик по моделям, прогнозы каждого часа на месяце вне обучения).
    """
    d = X[rows]
    y = d[TARGET]
    month = d["ts"].dt.month
    makers = {"Линейная (основная)": make_linear, "Нейросеть (ансамбль MLP)": lambda: MLPEnsemble(n_models=n_models)}
    rows_out = []
    preds = pd.DataFrame({"ts": d["ts"], "y": y})   # прогноз каждого часа моделью, не видевшей его месяц
    for name, mk in makers.items():
        p = pd.Series(index=d.index, dtype=float)
        for m in sorted(month.unique()):
            tr, te = month != m, month == m
            p[te] = mk().fit(d.loc[tr, FEATURES], y[tr]).predict(d.loc[te, FEATURES])
        preds["linear" if name.startswith("Линейная") else "nn"] = p
        s = _scores(y, p)
        tr, te = month <= 4, month >= 5
        s["R2_янв-апр→май-июл"] = _scores(y[te], mk().fit(d.loc[tr, FEATURES], y[tr]).predict(d.loc[te, FEATURES]))["R2"]
        for m in sorted(month.unique()):
            s[f"r_м{m}"] = _scores(y[month == m], p[month == m])["r"]
        rows_out.append({"модель": name, **s})
    return pd.DataFrame(rows_out).set_index("модель"), preds
