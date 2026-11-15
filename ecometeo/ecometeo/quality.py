"""Показатели качества модели — понятным языком для технолога и эколога.

Считаются по честной проверке: каждый час предсказан моделью, которая
не видела его месяц (artifacts/cv_predictions.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quality(cv: pd.DataFrame, col: str = "linear") -> dict:
    y, p = cv["y"].to_numpy(), cv[col].to_numpy()
    err = p - y
    ape = np.abs(err) / y * 100
    sigma = float(np.std(err))
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),                  # мг/Нм³
        "mape": float(ape.mean()),                         # средняя ошибка, %
        "median_ape": float(np.median(ape)),               # типичная ошибка, %
        "p90_ape": float(np.percentile(ape, 90)),          # в 9 из 10 часов ошибка меньше
        "within10": float((ape <= 10).mean() * 100),       # % часов с ошибкой ≤ 10%
        "within20": float((ape <= 20).mean() * 100),
        "within2s": float((np.abs(err) <= 2 * sigma).mean() * 100),
        "bias": float(err.mean()),                         # систематический сдвиг
        "r": float(np.corrcoef(y, p)[0, 1]),
        "r2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "sigma": sigma,
        "mean_y": float(y.mean()),
        # «Правильно ли модель видит рост/спад пыли от часа к часу»
        "direction": float((np.sign(np.diff(p)) == np.sign(np.diff(y))).mean() * 100),
    }


def by_month(cv: pd.DataFrame, col: str = "linear") -> pd.DataFrame:
    m = cv.assign(month=pd.to_datetime(cv.ts).dt.month)
    rows = []
    for k, g in m.groupby("month"):
        q = quality(g, col)
        rows.append({"месяц": k, "часов": q["n"], "ср. ошибка, %": q["mape"],
                     "ошибка ≤10%, доля часов": q["within10"], "r": q["r"]})
    return pd.DataFrame(rows)


def correction_table(cv: pd.DataFrame, rho: float, horizons=(1, 3, 6, 12, 24)) -> pd.DataFrame:
    """Ошибка прогноза, если известен замер h часов назад (поправка ρ^h · последняя ошибка)."""
    s = cv.set_index("ts")[["y", "linear"]].asfreq("h")
    res = s.y - s.linear
    rows = []
    for h in horizons:
        corr = s.linear + rho ** h * res.shift(h)
        m = corr.notna() & s.y.notna()
        base = ((s.y - s.linear)[m].abs() / s.y[m]) * 100
        new = ((s.y - corr)[m].abs() / s.y[m]) * 100
        rows.append({"h": h, "base": float(base.mean()), "corr": float(new.mean()),
                     "within10": float((new <= 10).mean() * 100), "n": int(m.sum())})
    return pd.DataFrame(rows)


def grade(mape: float) -> tuple[str, str]:
    """Словесная оценка точности (для лампы на экране)."""
    if mape <= 10:
        return "ВЫСОКАЯ", "ok"
    if mape <= 20:
        return "СРЕДНЯЯ", "warn"
    return "НИЗКАЯ", "bad"
