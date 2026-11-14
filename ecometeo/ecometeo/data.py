"""Загрузка и очистка данных ЭкоМетео.

Одна строка = один час. Результат load_dataset() — таблица с очищенными
выбросами, статусом анализатора, режимами печи, циклами и погодой.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Переименование столбцов газоанализатора
EMISSIONS = {
    "Расход(Нкуб.м/час)": "flow",
    "Пыль(мг/Нкуб.м)": "dust",
    "NO(мг/Нкуб.м)": "NO",
    "SO2(мг/Нкуб.м)": "SO2",
    "CO(мг/Нкуб.м)": "CO",
    "NO2(мг/Нкуб.м)": "NO2",
    "H2O(%)": "H2O",
    "O2(%)": "O2",
}
GASES = ["flow", "dust", "NO", "SO2", "CO", "NO2", "O2"]


@dataclass
class Thresholds:
    """Пороги очистки и разметки. Все можно поменять при вызове."""
    gas_max: float = 5000.0      # мг/Нм³: выше — сбой анализатора
    o2_min: float = 5.0          # %: ниже — сбой (кроме нуля = прибор выключен)
    o2_max: float = 25.0         # %: выше — сбой (в воздухе 20.9%)
    h2o_max: float = 100.0       # %
    td_max: float = 60.0         # минут под током за час (эксперт: >60 — артефакт)
    repair_stop_h: int = 4       # останов >= N часов — ремонт
    ramp_power_share: float = 0.85
    ramp_min_scrap_t: float = 140.0
    ramp_max_h: int = 6


def _material_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    lom = [c for c in df.columns if str(c).endswith("_lom")]
    dsp = [c for c in df.columns if str(c).endswith("_dsp")
           and c not in ("timedelta_dsp", "electric_power_dsp", "o2_dsp", "carbon_dsp")]
    return lom, dsp


def load_raw(path: str | Path) -> pd.DataFrame:
    """Читает data.xlsx организаторов и строит метку времени.

    «Время» = 1..23, 0. Час 0 — это 24:00 тех же суток (проверено по
    непрерывности рядов). Метка = конец часа.
    """
    raw = pd.read_excel(path)
    raw = raw.loc[:, [c for c in raw.columns if not str(c).startswith("Unnamed")]].copy()
    raw["ts"] = raw["Дата"] + pd.to_timedelta(raw["Время"].replace(0, 24), unit="h")
    return raw.rename(columns=EMISSIONS)


def load_weather_hourly(path: str | Path) -> pd.DataFrame:
    """Почасовая погода (Open-Meteo archive; по умолчанию — среднее по Челябинской области) из JSON.

    Open-Meteo даёт мгновенные значения на начало часа. Для часа, который
    кончается в момент ts, берём среднее значений в ts−1ч и ts; направление
    ветра усредняем как вектор. Индекс — ts (конец часа), как в данных печи.
    """
    w = json.loads(Path(path).read_text(encoding="utf-8"))["hourly"]
    h = pd.DataFrame(
        {
            "T_air": w["temperature_2m"],
            "RH_air": w["relative_humidity_2m"],
            "P_air": w["pressure_msl"],
            "wind": w["wind_speed_10m"],
            "wd_x": np.cos(np.deg2rad(w["wind_direction_10m"])),
            "wd_y": np.sin(np.deg2rad(w["wind_direction_10m"])),
            # высота слоя перемешивания (м) и ощущаемая температура — если есть в файле
            "blh": w.get("boundary_layer_height", [np.nan] * len(w["time"])),
            "T_app": w.get("apparent_temperature", [np.nan] * len(w["time"])),
        },
        index=pd.to_datetime(w["time"]),
    ).astype(float)
    h = (h + h.shift(1)) / 2
    h["wind_dir"] = np.degrees(np.arctan2(h.pop("wd_y"), h.pop("wd_x"))) % 360
    return h


def load_weather(path: str | Path) -> pd.DataFrame:
    """Суточная погода (Open-Meteo, Челябинск) из JSON. Почасовой файл
    (с ключом «hourly») передаётся в load_weather_hourly."""
    w = json.loads(Path(path).read_text(encoding="utf-8"))
    if "hourly" in w:
        return load_weather_hourly(path)
    n = len(w["temperature_2m_mean"])
    start = w.get("start_date", "2022-01-01")
    return pd.DataFrame(
        {
            "T_air": w["temperature_2m_mean"],
            "RH_air": w["relative_humidity_2m_mean"],
            "P_air": w["pressure_msl_mean"],
            "wind": w["wind_speed_10m_mean"],
        },
        index=pd.date_range(start, periods=n, freq="D"),
    )


def clean(raw: pd.DataFrame, th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Очистка: статус анализатора, обрезка/обнуление ошибок процесса."""
    df = raw.copy()
    lom, dsp = _material_columns(df)
    df[lom + dsp] = df[lom + dsp].clip(lower=0)              # отрицательные массы -> 0
    df["power"] = df["electric_power_dsp"].clip(lower=0)      # МВт·ч за час
    df["td_min"] = df["timedelta_dsp"].clip(upper=th.td_max)  # минуты под током
    df["scrap_t"] = df[lom].sum(axis=1) / 1000                # лом, т

    off = (df[["flow", "dust", "CO", "O2"]] == 0).all(axis=1)
    fault = ~off & (
        (df[["NO", "SO2", "CO", "NO2"]] > th.gas_max).any(axis=1)
        | (df.O2 > th.o2_max) | (df.O2 < th.o2_min) | (df.H2O > th.h2o_max)
    )
    df["status"] = np.select([off, fault], ["OFF", "FAULT"], "OK")
    ok = df.status.eq("OK")
    df = df.copy()
    for g in GASES:
        df[g + "_ok"] = df[g].where(ok)   # только годные часы; остальное NaN
    # массовый выброс, г/ч
    for g in ["dust", "NO", "SO2", "CO"]:
        df[g + "_g_h"] = df[g + "_ok"] * df["flow_ok"] / 1000
    return df


def label_periods(df: pd.DataFrame, th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Режимы печи и циклы работы."""
    df = df.copy()
    P = df["power"]
    stop = P <= 0
    run_id = (stop != stop.shift()).cumsum()
    run_len = stop.groupby(run_id).transform("size")

    # цикл = непрерывная работа между остановками
    work = ~stop
    df["cycle"] = np.where(work, run_id, np.nan)
    df["cycle"] = df["cycle"].rank(method="dense")
    df["cycle_hour"] = df[work].groupby(run_id[work]).cumcount() + 1
    df["cycle_hour"] = df["cycle_hour"].fillna(0).astype(int)
    df["cycle_len"] = df.groupby("cycle")["ts"].transform("size").fillna(0).astype(int)
    df["phase"] = np.select(
        [stop, df.cycle_hour <= 3, df.cycle_hour == df.cycle_len],
        ["стоп", "начало", "последний час"], "стационар",
    )

    # режимы: ОСТАНОВ / ПРОСТОЙ / РОЗЖИГ / ПЛАВКИ
    p_norm = P[P > 0].median()
    mode = np.where(stop & (run_len >= th.repair_stop_h), "ОСТАНОВ",
                    np.where(stop, "ПРОСТОЙ", ""))
    mode = pd.Series(mode, index=df.index, dtype=object)
    for i in range(len(df)):
        if mode.iat[i]:
            continue
        prev = mode.iat[i - 1] if i else ""
        h = df["cycle_hour"].iat[i]
        if prev in ("ОСТАНОВ", "РОЗЖИГ") and h <= th.ramp_max_h and (
            P.iat[i] < th.ramp_power_share * p_norm or df["scrap_t"].iat[i] < th.ramp_min_scrap_t
        ):
            mode.iat[i] = "РОЗЖИГ"
        else:
            mode.iat[i] = "ПЛАВКИ"
    df["mode"] = mode
    return df


def add_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Почасовая погода — по метке конца часа; суточная — по дате."""
    if (weather.index.to_series().diff().dropna() < pd.Timedelta("1D")).any():
        return df.join(weather, on="ts")
    day = (df["ts"] - pd.Timedelta("1h")).dt.floor("D")
    return df.join(weather, on=day)


def load_dataset(data_path: str | Path, weather_path: str | Path,
                 th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Полный конвейер: сырой файл -> очищенная размеченная таблица с погодой."""
    df = clean(load_raw(data_path), th)
    df = label_periods(df, th)
    df = add_weather(df, load_weather(weather_path))
    return df
