"""Онлайн-погода в среднем по Челябинской области (Южный Урал): текущая и прогноз по часам.

Источник тот же, что у архива, на котором обучена модель (Open-Meteo), те же единицы
(°C, %, гПа, км/ч, градусы) и те же 7 городов: значения усредняются, направление ветра —
как средний вектор. Ответ кэшируется на 10 минут. Если сети нет, функции возвращают None,
а интерфейс переходит на ручной ввод погоды.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# Города Челябинской области, по которым усредняется погода (и в обучении, и онлайн)
POINTS = {"chelyabinsk": (55.15, 61.38), "zlatoust": (55.17, 59.67), "miass": (55.05, 60.11), "satka": (55.05, 59.03),
          "asha": (55.00, 57.27), "troitsk": (54.08, 61.56), "magnitogorsk": (53.42, 59.05)}
TZ = "Asia/Yekaterinburg"
VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,apparent_temperature"
HOURLY = VARS + ",boundary_layer_height"   # высота слоя перемешивания есть только в почасовом прогнозе
URL = "https://api.open-meteo.com/v1/forecast"
TTL = 600  # с

_cache: dict = {"t": 0.0, "data": None, "error": None}


def _mean_block(blocks: list[dict]) -> dict:
    """Среднее по городам для блока «current» или «hourly» ответа API."""
    out = {"time": blocks[0]["time"]}
    for k in blocks[0]:
        if k in ("time", "interval", "wind_direction_10m"):
            continue
        out[k] = np.nanmean(np.array([b[k] for b in blocks], dtype=float), axis=0).tolist()
    a = np.deg2rad(np.array([b["wind_direction_10m"] for b in blocks], dtype=float))
    out["wind_direction_10m"] = (np.degrees(np.arctan2(np.nanmean(np.sin(a), axis=0), np.nanmean(np.cos(a), axis=0))) % 360).tolist()
    return out


def _fetch(days: int = 7) -> dict:
    q = urllib.parse.urlencode({"latitude": ",".join(str(p[0]) for p in POINTS.values()),
                                "longitude": ",".join(str(p[1]) for p in POINTS.values()), "timezone": TZ,
                                "current": VARS, "hourly": HOURLY, "forecast_days": days, "past_days": 1})
    with urllib.request.urlopen(f"{URL}?{q}", timeout=10) as r:
        res = json.loads(r.read().decode("utf-8"))
    res = res if isinstance(res, list) else [res]
    return {"current": _mean_block([x["current"] for x in res]), "hourly": _mean_block([x["hourly"] for x in res])}


def get(force: bool = False) -> tuple[dict | None, str | None]:
    """(ответ API, текст ошибки). Кэш на TTL секунд."""
    if force or time.time() - _cache["t"] > TTL or _cache["data"] is None:
        try:
            _cache.update(data=_fetch(), error=None, t=time.time())
        except Exception as e:  # нет сети, таймаут, ошибка API
            _cache.update(error=f"нет связи с сервисом погоды ({type(e).__name__})", t=time.time())
    return _cache["data"], _cache["error"]


def current() -> tuple[dict | None, str | None]:
    """Текущая погода: T_air, RH_air, P_air, wind, wind_dir, time."""
    data, err = get()
    if not data:
        return None, err
    c = data["current"]
    h = data["hourly"]
    now = pd.Timestamp(c["time"]).floor("h")
    times = pd.to_datetime(h["time"])
    blh = pd.Series(h["boundary_layer_height"], index=times, dtype=float).reindex([now]).iloc[0]
    return {"time": pd.Timestamp(c["time"]), "T_air": c["temperature_2m"], "RH_air": c["relative_humidity_2m"],
            "P_air": c["pressure_msl"], "wind": c["wind_speed_10m"], "wind_dir": c["wind_direction_10m"],
            "T_app": c["apparent_temperature"], "blh": blh}, err


def hourly(hours: int = 72) -> tuple[pd.DataFrame | None, str | None]:
    """Прогноз по часам начиная со следующего часа.

    Как и в обучении: значение часа, который кончается в ts, — среднее
    мгновенных значений в ts−1ч и ts; направление ветра — как вектор.
    """
    data, err = get()
    if not data:
        return None, err
    h = data["hourly"]
    w = pd.DataFrame({"T_air": h["temperature_2m"], "RH_air": h["relative_humidity_2m"],
                      "P_air": h["pressure_msl"], "wind": h["wind_speed_10m"],
                      "T_app": h["apparent_temperature"], "blh": h["boundary_layer_height"],
                      "wd_x": np.cos(np.deg2rad(h["wind_direction_10m"])),
                      "wd_y": np.sin(np.deg2rad(h["wind_direction_10m"]))},
                     index=pd.to_datetime(h["time"]))
    w = (w + w.shift(1)) / 2
    w["wind_dir"] = np.degrees(np.arctan2(w.pop("wd_y"), w.pop("wd_x"))) % 360
    now = pd.Timestamp(data["current"]["time"]).floor("h")
    w = w[w.index > now].dropna().head(hours)
    return w.rename_axis("ts").reset_index(), err
