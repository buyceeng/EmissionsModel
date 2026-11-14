"""Мониторинг: сравнение замера пыли с ожидаемой по модели + контроль прибора."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import DustModel


def run_monitor(X: pd.DataFrame, model: DustModel, k_sigma: float = 2.0) -> pd.DataFrame:
    """Для каждого часа: ожидаемая пыль (линейная модель), замер, отклонение и сигнал.

    Сигналы:
      ПРИБОР_МОЛЧИТ   — печь работает, анализатор выключен (все нули);
      СБОЙ_ПРИБОРА    — невозможные значения;
      НЕТ_ИЗМЕРЕНИЙ   — печь стоит и прибор выключен;
      ПРЕВЫШЕНИЕ      — замер выше ожидаемого больше чем на k·σ ошибки модели;
      ВНЕ_ОПЫТА       — погода вне диапазона обучения: ожидаемое — экстраполяция;
      норма           — всё в порядке.
    """
    out = X[["ts", "power", "scrap_t", "T_air", "wind", "wind_dir", "mode", "phase", "status", "dust", "dust_ok"]].copy()
    out = out.join(model.predict(X))
    s = model.resid_std
    out["deviation"] = out["dust_ok"] - out["dust_linear"]
    out["z"] = out["deviation"] / s
    out["dust_corr"] = corrected_forecast(out, model)
    out["signal"] = np.select(
        [
            (out.status == "OFF") & (out.power > 0),
            out.status == "FAULT",
            out.status == "OFF",
            out["z"] > k_sigma,
            model.out_of_range(X) & X[["T_air", "RH_air", "P_air", "wind"]].notna().all(axis=1),
        ],
        ["ПРИБОР_МОЛЧИТ", "СБОЙ_ПРИБОРА", "НЕТ_ИЗМЕРЕНИЙ", "ПРЕВЫШЕНИЕ", "ВНЕ_ОПЫТА"],
        "норма",
    )
    return out


def corrected_forecast(out: pd.DataFrame, model: DustModel, max_age: int = 24) -> pd.Series:
    """Прогноз часа с поправкой по последнему замеру ДО этого часа.

    Берётся последняя известная ошибка модели (замер − ожидаемое) и затухает
    как ar_rho**h, где h — сколько часов назад был замер. Старше max_age часов
    поправка не применяется. Тревога «ПРЕВЫШЕНИЕ» считается по модели без
    поправки: иначе медленный рост пыли (износ газоочистки) «растворился» бы.
    """
    res = out["deviation"].shift(1)                       # только прошлые часы
    have = res.notna()
    last = res.ffill()
    age = have.groupby(have.cumsum()).cumcount() + 1      # часов с последнего замера
    age = age.where(have.cumsum() > 0)
    corr = out["dust_linear"] + model.correction(last, age)
    return corr.where(age <= max_age, out["dust_linear"])


def summary(mon: pd.DataFrame) -> pd.DataFrame:
    """Сколько часов с каждым сигналом по месяцам."""
    t = mon.assign(month=mon.ts.dt.month).pivot_table(
        index="month", columns="signal", values="ts", aggfunc="count", fill_value=0
    )
    return t
