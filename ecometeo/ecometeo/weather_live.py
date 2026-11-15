"""Онлайн-погода в среднем по Челябинской области (Южный Урал): текущая и прогноз по часам.

Источник тот же, что у архива, на котором обучена модель (Open-Meteo), те же единицы
(°C, %, гПа, км/ч, градусы) и те же 7 городов: значения усредняются, направление ветра —
как средний вектор.

Как устроено, чтобы сайт выдерживал много посетителей:
  * к Open-Meteo ходит только ОДИН поток за раз (блокировка), а не каждый посетитель;
  * фоновый поток обновляет погоду каждые 10 минут сам — страницы не ждут сеть;
  * если данные устарели, посетитель сразу получает последние известные, а обновление
    идёт в фоне;
  * последний удачный ответ хранится на диске: его видят все процессы сервера
    (gunicorn/waitress с несколькими рабочими) и он переживает перезапуск;
  * сбой сети не превращается в «нет связи», пока последним данным меньше 6 часов;
    после сбоя повтор через минуту.
Если погоды нет совсем (первый запуск без сети), функции возвращают None,
а интерфейс переходит на ручной ввод погоды.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Города Челябинской области, по которым усредняется погода (и в обучении, и онлайн)
POINTS = {"chelyabinsk": (55.15, 61.38), "zlatoust": (55.17, 59.67), "miass": (55.05, 60.11), "satka": (55.05, 59.03),
          "asha": (55.00, 57.27), "troitsk": (54.08, 61.56), "magnitogorsk": (53.42, 59.05)}
TZ = "Asia/Yekaterinburg"
VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,apparent_temperature"
HOURLY = VARS + ",boundary_layer_height"   # высота слоя перемешивания есть только в почасовом прогнозе
URL = "https://api.open-meteo.com/v1/forecast"

TTL = 600            # с: погоду считаем свежей 10 минут
RETRY = 60           # с: после сбоя пробуем снова через минуту
MAX_STALE = 6 * 3600 # с: столько показываем последние удачные данные без плашки «нет связи»
WAIT_FIRST = 15      # с: сколько посетитель ждёт самый первый ответ, когда данных ещё нет совсем
CACHE_FILE = Path(os.environ.get("ECOMETEO_WEATHER_CACHE",
                                 Path(__file__).resolve().parent.parent / "artifacts" / "weather_live_cache.json"))

_lock = threading.Lock()          # один поток в сети
_state = {"data": None, "t": 0.0, "error": None, "next_try": 0.0, "refreshing": False}
_parsed: dict = {}                # разобранные таблицы для текущей версии данных
_bg_started = False


# ---------- запрос к Open-Meteo ----------
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
    """Один запрос сразу за 7 городов; два коротких повтора при сбое."""
    q = urllib.parse.urlencode({"latitude": ",".join(str(p[0]) for p in POINTS.values()),
                                "longitude": ",".join(str(p[1]) for p in POINTS.values()), "timezone": TZ,
                                "current": VARS, "hourly": HOURLY, "forecast_days": days, "past_days": 1})
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{URL}?{q}", headers={"User-Agent": "EcoMeteo/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                res = json.loads(r.read().decode("utf-8"))
            res = res if isinstance(res, list) else [res]
            return {"current": _mean_block([x["current"] for x in res]), "hourly": _mean_block([x["hourly"] for x in res])}
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:            # слишком часто — подождать подольше
                time.sleep(2 + 3 * attempt)
            elif 400 <= e.code < 500:    # ошибка в запросе — повтор не поможет
                break
        except Exception as e:           # таймаут, нет сети, битый ответ
            last = e
        time.sleep(0.5 + attempt)
    raise last if last else RuntimeError("Open-Meteo не ответил")


# ---------- общий кэш на диске (для нескольких процессов и перезапусков) ----------
def _read_disk():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            obj = json.load(f)
        return obj["data"], float(obj["t"])
    except Exception:
        return None, 0.0


def _write_disk(data, t):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"t": t, "data": data}, f)
        os.replace(tmp, CACHE_FILE)          # атомарно: другие процессы не прочтут полфайла
    except Exception:
        pass                                 # диск недоступен — работаем только с памятью


def _adopt_disk_if_newer():
    data, t = _read_disk()
    if data is not None and t > _state["t"]:
        _state.update(data=data, t=t, error=None)


def _refresh():
    """Обновить погоду. Вызывается под _lock — в сети всегда только один поток процесса."""
    _adopt_disk_if_newer()                         # возможно, другой процесс уже обновил
    if _state["data"] is not None and time.time() - _state["t"] < TTL:
        return
    try:
        data = _fetch()
        now = time.time()
        _state.update(data=data, t=now, error=None, next_try=0.0)
        _write_disk(data, now)
    except Exception as e:
        _state.update(error=f"нет связи с сервисом погоды ({type(e).__name__})", next_try=time.time() + RETRY)


def _refresh_in_background():
    if _state["refreshing"]:
        return
    _state["refreshing"] = True

    def run():
        try:
            with _lock:
                _refresh()
        finally:
            _state["refreshing"] = False
    threading.Thread(target=run, name="weather-refresh", daemon=True).start()


def start_background_refresh():
    """Фоновый поток: держит погоду свежей сам, чтобы страницы не ждали сеть."""
    global _bg_started
    if _bg_started:
        return
    _bg_started = True

    def loop():
        while True:
            try:
                with _lock:
                    _refresh()
            except Exception:
                pass
            wait = RETRY if _state["error"] and (_state["data"] is None or time.time() - _state["t"] > TTL) else TTL
            time.sleep(wait)
    threading.Thread(target=loop, name="weather-loop", daemon=True).start()


# ---------- то, что вызывает сайт ----------
def get(force: bool = False) -> tuple[dict | None, str | None]:
    """(данные, текст ошибки или None). Никогда не держит посетителя дольше WAIT_FIRST секунд."""
    now = time.time()
    if _state["data"] is None:
        _adopt_disk_if_newer()
    fresh = _state["data"] is not None and now - _state["t"] < TTL
    if fresh and not force:
        return _state["data"], None

    if _state["data"] is not None:
        # данные есть, но устарели: отдаём их сразу, обновляем в фоне
        if now >= _state["next_try"]:
            _refresh_in_background()
        age = now - _state["t"]
        return _state["data"], (None if age < MAX_STALE else _state["error"])

    # данных нет совсем: ждём ответ одного потока (остальные посетители ждут его же, а не идут в сеть сами)
    if now >= _state["next_try"] and _lock.acquire(timeout=WAIT_FIRST):
        try:
            if _state["data"] is None:
                _refresh()
        finally:
            _lock.release()
    return _state["data"], (None if _state["data"] is not None else _state["error"] or "нет связи с сервисом погоды")


def updated_at() -> pd.Timestamp | None:
    """Когда погода последний раз успешно получена (для подписи на странице)."""
    return pd.Timestamp(_state["t"], unit="s", tz="UTC").tz_convert(TZ).tz_localize(None) if _state["t"] else None


def version() -> float:
    """Метка версии данных: меняется при каждом обновлении погоды (ключ для кэша страниц)."""
    return _state["t"]


def _parse(data: dict) -> dict:
    """Разбор ответа в таблицы — один раз на версию данных, а не на каждый запрос страницы."""
    global _parsed
    cached = _parsed
    if cached.get("data") is data:
        return cached
    h = data["hourly"]
    times = pd.to_datetime(h["time"])
    w = pd.DataFrame({"T_air": h["temperature_2m"], "RH_air": h["relative_humidity_2m"],
                      "P_air": h["pressure_msl"], "wind": h["wind_speed_10m"],
                      "T_app": h["apparent_temperature"], "blh": h["boundary_layer_height"],
                      "wd_x": np.cos(np.deg2rad(h["wind_direction_10m"])),
                      "wd_y": np.sin(np.deg2rad(h["wind_direction_10m"]))},
                     index=times)
    w = (w + w.shift(1)) / 2
    w["wind_dir"] = np.degrees(np.arctan2(w.pop("wd_y"), w.pop("wd_x"))) % 360
    blh = pd.Series(h["boundary_layer_height"], index=times, dtype=float)
    _parsed = {"data": data, "hourly": w, "blh": blh}   # подмена целиком: параллельные запросы не видят полупустой словарь
    return _parsed


def current() -> tuple[dict | None, str | None]:
    """Текущая погода: T_air, RH_air, P_air, wind, wind_dir, T_app, blh, time."""
    data, err = get()
    if not data:
        return None, err
    c = data["current"]
    now = pd.Timestamp(c["time"]).floor("h")
    blh = _parse(data)["blh"].reindex([now]).iloc[0]
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
    w = _parse(data)["hourly"]
    now = pd.Timestamp(data["current"]["time"]).floor("h")
    w = w[w.index > now].dropna().head(hours)
    return w.rename_axis("ts").reset_index(), err


def _reset_for_tests():
    """Сбросить память процесса (кэш на диске остаётся). Только для проверок."""
    global _parsed
    _state.update(data=None, t=0.0, error=None, next_try=0.0, refreshing=False)
    _parsed = {}
