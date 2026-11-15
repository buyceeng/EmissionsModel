"""Признаки для модели пыли."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROCESS = ["power", "td_min", "scrap_t", "is_stop", "first3", "last1"]
# почасовая погода + направление ветра + высота слоя перемешивания (лог) + ощущаемая температура
WEATHER = ["T_air", "RH_air", "P_air", "wind", "wd_sin", "wd_cos", "blh_log", "T_app"]
BLH_DEFAULT = 400.0   # м, типичная высота слоя перемешивания — если прогноза нет


def apparent_temperature(T, RH, wind_kmh):
    """Ощущаемая температура (формула Стедмана, как у Open-Meteo): запасной расчёт."""
    e = RH / 100 * 6.105 * np.exp(17.27 * T / (237.7 + T))
    return T + 0.33 * e - 0.70 * wind_kmh / 3.6 - 4.0
WINDOW = [f"{c}_l{k}" for k in (1, 2, 3) for c in ("power", "td_min", "scrap_t")]
# температура × мощность: в тепло печь добавляет к пыли больше, чем в мороз (experiments/low_flow_result.txt)
INTERACT = ["TxP"]
FEATURES = PROCESS + WEATHER + WINDOW + INTERACT
TARGET = "dust_ok"


def build_features(df: pd.DataFrame, lag_fill: str = "zero") -> pd.DataFrame:
    """Добавляет признаки режима и окно из трёх прошлых часов."""
    X = df.copy()
    X["is_stop"] = (X["power"] <= 0).astype(float)
    X["first3"] = ((X["cycle_hour"] >= 1) & (X["cycle_hour"] <= 3)).astype(float)
    X["last1"] = ((X["cycle_hour"] > 0) & (X["cycle_hour"] == X["cycle_len"])).astype(float)
    # направление ветра (откуда дует, °) -> две координаты, чтобы 359° и 1° были рядом
    X["wd_sin"] = np.sin(np.deg2rad(X["wind_dir"]))
    X["wd_cos"] = np.cos(np.deg2rad(X["wind_dir"]))
    # слой перемешивания: низкий (инверсия, НМУ) — пыли больше; берём логарифм
    blh = X["blh"] if "blh" in X else pd.Series(BLH_DEFAULT, index=X.index)
    X["blh_log"] = np.log(blh.astype(float).clip(lower=10))
    if "T_app" not in X or X["T_app"].isna().all():
        X["T_app"] = apparent_temperature(X["T_air"], X["RH_air"], X["wind"])
    X["TxP"] = X["T_air"] * X["power"]
    for k in (1, 2, 3):
        for c in ("power", "td_min", "scrap_t"):
            # до начала ряда: 0 (печь стояла) или первое значение (печь уже работала — для прогноза)
            fill = X[c].iloc[0] if lag_fill == "first" and len(X) else 0
            X[f"{c}_l{k}"] = X[c].shift(k).fillna(fill)
    return X


def training_rows(X: pd.DataFrame) -> pd.Series:
    """Только годные часы анализатора и полная погода."""
    return X["status"].eq("OK") & X[WEATHER].notna().all(axis=1) & X[TARGET].notna()
