"""Прогноз на завтра и сценарии для периодов НМУ (по линейной модели)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .features import build_features
from .models import DustModel


def make_plan(hours: int = 24, power: float = 66.0, td_min: float = 59.0, scrap_t: float = 180.0,
              T_air: float = 0.0, RH_air: float = 75.0, P_air: float = 1015.0, wind: float = 12.0,
              wind_dir: float = 270.0, stop_hours: tuple[int, ...] = ()) -> pd.DataFrame:
    """План на сутки: работа печи + прогноз погоды. stop_hours — часы останова (1..24).
    wind_dir — откуда дует ветер, градусы (0 — север, 90 — восток, 180 — юг, 270 — запад)."""
    h = np.arange(1, hours + 1)
    plan = pd.DataFrame({"hour": h, "power": power, "td_min": td_min, "scrap_t": scrap_t,
                         "T_air": T_air, "RH_air": RH_air, "P_air": P_air, "wind": wind, "wind_dir": wind_dir})
    stop = plan.hour.isin(stop_hours)
    plan.loc[stop, ["power", "td_min", "scrap_t"]] = 0
    # циклы внутри плана
    work = ~stop
    run = (work != work.shift()).cumsum()
    plan["cycle_hour"] = np.where(work, plan.groupby(run).cumcount() + 1, 0)
    plan["cycle_len"] = np.where(work, plan.groupby(run)["hour"].transform("size"), 0)
    plan["ts"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(plan.hour, unit="h")
    return plan


def stop_window_range(ts: pd.Series) -> tuple[str, str]:
    """Первый и последний час прогноза в том виде, как их вводят в остановы: «15.11 13», «18.11 12»."""
    def label(t):
        t = pd.Timestamp(t)
        h = t.hour if t.hour != 0 else 24                     # 00:00 = 24-й час предыдущих суток
        return f"{(t - pd.Timedelta('1h')):%d.%m} {h}"
    return label(ts.iloc[0]), label(ts.iloc[-1])


def parse_stops(text: str, ts: pd.Series) -> tuple[pd.Series, str | None]:
    """Окна останова печи вида «15.11 10-14; 16.11 8-12» (часы конца, включительно).

    Каждое окно проверяется: формат, существующая дата, часы 1–24, начало не позже конца
    и — главное — что окно попадает в часы прогноза. Прошедшие даты и даты за горизонтом
    прогноза не применяются, а возвращаются в тексте ошибки, чтобы опечатка (например,
    10-й месяц вместо 11-го) не прошла незаметно. Правильные окна применяются в любом случае.

    Возвращает маску остановленных часов для меток ts и текст ошибки (или None).
    """
    stop = pd.Series(False, index=ts.index)
    if not text or not text.strip() or ts.empty:
        return stop, None
    hour = ts.dt.hour.where(ts.dt.hour != 0, 24)              # 00:00 = 24-й час предыдущих суток
    date = (ts - pd.Timedelta("1h")).dt
    first, last = stop_window_range(ts)
    years = sorted(set(date.year))
    errors = []
    text = re.sub(r"\b(\d{1,2}),(\d{1,2})(?=\s)", r"\1.\2", text)   # «16,11 10-14» — опечатка, это дата 16.11
    for part in text.replace(",", ";").split(";"):
        part = " ".join(part.split())
        if not part:
            continue
        try:
            day, hrs = part.split(" ")
            dd, mm = (int(x) for x in day.split("."))
            h1, h2 = (int(x) for x in hrs.split("-"))
        except ValueError:
            errors.append(f"«{part}» — не понял запись, нужно так: {first.split()[0]} 10-14")
            continue
        if not any(_is_date(y, mm, dd) for y in years):
            errors.append(f"«{part}» — такой даты нет")
            continue
        if not (1 <= h1 <= 24 and 1 <= h2 <= 24):
            errors.append(f"«{part}» — часы должны быть от 1 до 24")
            continue
        if h1 > h2:
            errors.append(f"«{part}» — начало позже конца; через полночь разбейте на два окна: "
                          f"{dd:02d}.{mm:02d} {h1}-24; следующий день 1-{h2}")
            continue
        hit = (date.day == dd) & (date.month == mm) & hour.between(h1, h2)
        if not hit.any():
            errors.append(f"«{part}» — не входит в прогноз, остановы можно задать с {first} ч по {last} ч")
            continue
        stop |= hit
    return stop, ("; ".join(errors) if errors else None)


def _is_date(y, m, d) -> bool:
    try:
        pd.Timestamp(year=int(y), month=int(m), day=int(d))
        return True
    except (ValueError, TypeError):
        return False


def plan_from_weather(weather: pd.DataFrame, power: float = 66.0, td_min: float = 59.0,
                      scrap_t: float = 180.0, stop: pd.Series | None = None) -> pd.DataFrame:
    """План печи на реальные часы прогноза погоды (столбец ts = конец часа)."""
    plan = weather.copy()
    plan["power"], plan["td_min"], plan["scrap_t"] = power, td_min, scrap_t
    stop = pd.Series(False, index=plan.index) if stop is None else stop.reindex(plan.index, fill_value=False)
    plan.loc[stop, ["power", "td_min", "scrap_t"]] = 0
    work = ~stop
    run = (work != work.shift()).cumsum()
    plan["cycle_hour"] = np.where(work, plan.groupby(run).cumcount() + 1, 0)
    plan["cycle_len"] = np.where(work, plan.groupby(run)["ts"].transform("size"), 0)
    if work.iloc[0]:   # печь уже работает до начала прогноза: первые часы — не розжиг
        first = run == run.iloc[0]
        plan.loc[first, ["cycle_hour", "cycle_len"]] += 10
    plan["stop"] = stop
    plan.attrs["running_before"] = bool(work.iloc[0])
    return plan


def predict_plan(plan: pd.DataFrame, model: DustModel) -> pd.DataFrame:
    X = build_features(plan, lag_fill="first" if plan.attrs.get("running_before") else "zero")
    return plan.join(model.predict(X)).assign(вне_опыта=model.out_of_range(X))


def reduction_scenarios(model: DustModel, base: pd.DataFrame,
                        cuts=(0, 10, 20, 30, 40, 50)) -> pd.DataFrame:
    """Насколько снизится пыль, если урезать мощность и загрузку на X%."""
    rows = []
    ref = predict_plan(base, model)["dust_linear"].mean()
    for c in cuts:
        p = base.copy()
        p[["power", "scrap_t"]] *= (1 - c / 100)
        dust = predict_plan(p, model)["dust_linear"].mean()
        rows.append({"снижение_мощности_%": c, "пыль_ср": dust, "изменение_пыли_%": (dust / ref - 1) * 100})
    return pd.DataFrame(rows)


def temperature_effect(model: DustModel, base: pd.DataFrame, temps=(-20, -10, 0, 10, 20, 30)) -> pd.DataFrame:
    lo, hi = model.ranges.get("T_air", (-np.inf, np.inf))
    rows = []
    for t in temps:
        p = base.copy(); p["T_air"] = t
        rows.append({"температура_°C": t, "пыль_ср": predict_plan(p, model)["dust_linear"].mean(),
                     "в_опыте_модели": "да" if lo <= t <= hi else "нет (экстраполяция)"})
    return pd.DataFrame(rows)


def wind_effect(model: DustModel, base: pd.DataFrame, dirs=(0, 45, 90, 135, 180, 225, 270, 315)) -> pd.DataFrame:
    names = {0: "С", 45: "СВ", 90: "В", 135: "ЮВ", 180: "Ю", 225: "ЮЗ", 270: "З", 315: "СЗ"}
    rows = []
    for d in dirs:
        p = base.copy(); p["wind_dir"] = d
        rows.append({"ветер_из": f"{names.get(d, d)} ({d}°)", "пыль_ср": predict_plan(p, model)["dust_linear"].mean()})
    return pd.DataFrame(rows)
