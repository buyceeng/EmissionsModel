"""Командная строка ЭкоМетео.

Примеры:
  python -m ecometeo.cli train
  python -m ecometeo.cli monitor --date 2022-06-15
  python -m ecometeo.cli forecast --temp 25 --stops 10,11,12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_dataset
from .features import FEATURES, TARGET, build_features, training_rows
from .forecast import make_plan, predict_plan, reduction_scenarios, temperature_effect, wind_effect
from .models import DustModel, MLPEnsemble, evaluate_leave_one_month_out
from .monitor import run_monitor, summary

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def cmd_train(a):
    ART.mkdir(exist_ok=True)
    print("1/4 Загрузка и очистка данных…")
    df = load_dataset(a.data, a.weather)
    X = build_features(df)
    rows = training_rows(X)
    print(f"    часов всего {len(X)}, годных для обучения {int(rows.sum())}")
    print(df["status"].value_counts().to_string())

    print("2/4 Честная проверка: обучаемся без одного месяца и проверяем на нём…")
    ev, cv = evaluate_leave_one_month_out(X, rows, n_models=a.n_models)
    cv.to_csv(ART / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    print(ev.round(3).to_string())
    ev.to_csv(ART / "evaluation.csv", encoding="utf-8-sig")

    print("3/4 Обучение финальной модели на всех годных часах…")
    model = DustModel(nn=MLPEnsemble(n_models=a.n_models))
    model.fit(X.loc[rows], X.loc[rows, TARGET])
    model.metrics = ev.round(4).to_dict(orient="index")
    # поправка по последнему замеру: насколько ошибка часа переходит на следующий час
    res = (cv.set_index("ts")["y"] - cv.set_index("ts")["linear"]).asfreq("h")
    model.ar_rho = float(res.autocorr(1))
    print(f"    поправка по последнему замеру: ошибка переходит на следующий час с коэф. {model.ar_rho:.2f}")
    model.save(ART / "model.joblib")
    print("Коэффициенты линейной модели:")
    print(model.coefficients().to_string(index=False))

    print("4/4 Мониторинг всего года…")
    mon = run_monitor(X, model, k_sigma=a.k)
    mon.to_csv(ART / "monitor.csv", index=False, encoding="utf-8-sig")
    X.to_pickle(ART / "dataset.pkl")
    print(summary(mon).to_string())
    (ART / "metrics.json").write_text(json.dumps(model.metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {ART}")


def _load():
    model = DustModel.load(ART / "model.joblib")
    X = pd.read_pickle(ART / "dataset.pkl")
    return model, X


def cmd_monitor(a):
    model, X = _load()
    mon = run_monitor(X, model, k_sigma=a.k)
    day = mon[mon.ts.dt.strftime("%Y-%m-%d") == a.date]
    if day.empty:
        print("Нет данных за эту дату"); return
    cols = ["ts", "mode", "power", "T_air", "wind_dir", "status", "dust_ok", "dust_linear", "dust_nn", "z", "signal"]
    print(day[cols].round(3).to_string(index=False))


def cmd_forecast(a):
    model, _ = _load()
    stops = tuple(int(x) for x in a.stops.split(",")) if a.stops else ()
    plan = make_plan(power=a.power, scrap_t=a.scrap, T_air=a.temp, RH_air=a.rh, wind_dir=a.wind_dir, stop_hours=stops)
    res = predict_plan(plan, model)
    print(res[["hour", "power", "scrap_t", "T_air", "dust_linear", "dust_nn", "вне_опыта"]].round(4).to_string(index=False))
    print("\nСценарии снижения мощности (для НМУ):")
    print(reduction_scenarios(model, plan).round(4).to_string(index=False))
    print("\nВлияние температуры воздуха при том же плане:")
    print(temperature_effect(model, plan).round(4).to_string(index=False))
    print("\nВлияние направления ветра при том же плане:")
    print(wind_effect(model, plan).round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="ЭкоМетео: модель пыли ДСП")
    sub = p.add_subparsers(required=True)
    t = sub.add_parser("train", help="обучить и проверить модель")
    t.add_argument("--data", default=str(ROOT / "data" / "data.xlsx"))
    t.add_argument("--weather", default=str(ROOT / "data" / "weather_region_2022_hourly.json"))
    t.add_argument("--n-models", type=int, default=5)
    t.add_argument("--k", type=float, default=2.0)
    t.set_defaults(f=cmd_train)
    m = sub.add_parser("monitor", help="сигналы за сутки")
    m.add_argument("--date", required=True)
    m.add_argument("--k", type=float, default=2.0)
    m.set_defaults(f=cmd_monitor)
    f = sub.add_parser("forecast", help="прогноз на сутки и сценарии НМУ")
    f.add_argument("--temp", type=float, default=0.0)
    f.add_argument("--rh", type=float, default=75.0)
    f.add_argument("--power", type=float, default=66.0)
    f.add_argument("--scrap", type=float, default=180.0)
    f.add_argument("--wind-dir", type=float, default=270.0, help="откуда дует ветер, градусы")
    f.add_argument("--stops", default="")
    f.set_defaults(f=cmd_forecast)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
