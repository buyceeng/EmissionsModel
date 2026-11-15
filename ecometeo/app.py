"""Веб-интерфейс ЭкоМетео (Flask)

Запуск:  python app.py   → http://127.0.0.1:5000
Перед первым запуском:  python -m ecometeo.cli train

Страницы:
  /          мнемосхема онлайн: погода сейчас (Open-Meteo) + план печи → ожидаемая пыль
  /forecast  прогноз выбросов по часам на 1–7 суток по прогнозу погоды, сценарии НМУ
  /archive   архив 2022: замер против ожидаемого, сигналы по часам
  /quality   качество модели: на сколько процентов ошибается и как часто
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from ecometeo import weather_live
from ecometeo.forecast import parse_stops, plan_from_weather, predict_plan, reduction_scenarios, temperature_effect, wind_effect, make_plan
from ecometeo.models import DustModel
from ecometeo.monitor import run_monitor
from ecometeo.quality import by_month, correction_table, grade, quality

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
app = Flask(__name__)
app.jinja_env.globals.update(sin_=lambda d: math.sin(math.radians(d)), cos_=lambda d: math.cos(math.radians(d)))

# ---------- стиль графиков (SCADA) ----------
C = {"fact": "#6d28d9", "exp": "#1d4f9c", "nn": "#e0701b", "temp": "#c8312d", "grid": "#c9ced3", "ink": "#16202a",
     "ok": "#21a336", "warn": "#e6a100", "bad": "#d6262b", "stop": "#8d949b"}
plt.rcParams.update({"font.family": ["Tahoma", "Verdana", "DejaVu Sans"], "font.size": 9,
                     "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff", "axes.edgecolor": "#7b838b",
                     "axes.grid": True, "grid.color": C["grid"], "grid.linestyle": "-", "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.color": C["ink"], "ytick.color": C["ink"], "axes.labelcolor": C["ink"],
                     "legend.frameon": False, "legend.fontsize": 8.5})

SIGNAL = {  # сигнал -> (подпись, класс лампы)
    "норма": ("НОРМА", "ok"), "ПРЕВЫШЕНИЕ": ("ПРЕВЫШЕНИЕ", "bad"), "ПРИБОР_МОЛЧИТ": ("ПРИБОР МОЛЧИТ", "warn"),
    "СБОЙ_ПРИБОРА": ("СБОЙ ПРИБОРА", "bad"), "НЕТ_ИЗМЕРЕНИЙ": ("НЕТ ИЗМЕРЕНИЙ", "off"), "ВНЕ_ОПЫТА": ("ВНЕ ОПЫТА МОДЕЛИ", "warn"),
}
LEVEL = {"ok": "обычный", "warn": "повышенный", "bad": "высокий"}
DIRS = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]

_state: dict = {}


def state():
    if not _state:
        model = DustModel.load(ART / "model.joblib")
        X = pd.read_pickle(ART / "dataset.pkl")
        cv = pd.read_csv(ART / "cv_predictions.csv", parse_dates=["ts"])
        d = X["dust_ok"].dropna()
        lv = {"p75": float(d.quantile(.75)), "p95": float(d.quantile(.95)),
              "min": float(d.quantile(.005)), "max": float(d.quantile(.995))}
        q = quality(cv)
        _state.update(model=model, X=X, mon=run_monitor(X, model), cv=cv, lv=lv, q=q, grade=grade(q["mape"]))
    return _state


def level(v, lv) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "off"
    return "ok" if v <= lv["p75"] else "warn" if v <= lv["p95"] else "bad"


def compass_name(deg) -> str:
    return DIRS[int(((deg % 360) + 22.5) // 45) % 8]


def png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout(); fig.savefig(buf, format="png", dpi=120); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fnum(v, fmt="{:.1f}"):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


app.jinja_env.filters.update(level_name=lambda c: LEVEL.get(c, "—"), compass=compass_name)


# ---------- параметры плана печи из формы ----------
def plan_params():
    f = request.values
    g = lambda k, d: float(str(f.get(k, d)).replace(",", ".") or d)
    meas = str(f.get("meas", "")).replace(",", ".").strip()
    try:
        meas = float(meas) if meas else None
    except ValueError:
        meas = None
    return dict(power=g("power", 66), scrap=g("scrap", 180), cut=g("cut", 0), stops=f.get("stops", ""),
                hours=int(g("hours", 72)), meas=meas,   # последний замер прибора, мг/Нм³ (необязательно)
                # ручная погода — если нет связи с сервисом погоды
                temp=g("temp", 0), rh=g("rh", 75), pres=g("pres", 1015), wind=g("wind", 12), wind_dir=g("wind_dir", 270))


def forecast_weather(p):
    """Почасовой прогноз погоды онлайн; без сети — ручная погода на весь горизонт."""
    w, err = weather_live.hourly(p["hours"])
    if w is not None and len(w):
        return w, "Open-Meteo, прогноз по часам", None
    start = pd.Timestamp.now().floor("h") + pd.Timedelta("1h")
    w = pd.DataFrame({"ts": pd.date_range(start, periods=p["hours"], freq="h"), "T_air": p["temp"], "RH_air": p["rh"],
                      "P_air": p["pres"], "wind": p["wind"], "wind_dir": p["wind_dir"]})
    return w, "ручной ввод (нет связи)", err


def expected_now(p, model):
    """Ожидаемая пыль в текущий час: погода сейчас (онлайн или ручная) + план печи."""
    cur, err = weather_live.current()
    if cur:
        wnow = pd.DataFrame([{**{k: cur[k] for k in ("T_air", "RH_air", "P_air", "wind", "wind_dir", "T_app", "blh")}, "ts": cur["time"]}])
        note = f"Open-Meteo, обновлено {cur['time']:%d.%m %H:%M}"
    else:
        wnow = pd.DataFrame([{"ts": pd.Timestamp.now(), "T_air": p["temp"], "RH_air": p["rh"], "P_air": p["pres"],
                              "wind": p["wind"], "wind_dir": p["wind_dir"]}])
        note = "нет связи — погода из ручного ввода"
    k = 1 - p["cut"] / 100
    now = predict_plan(plan_from_weather(wnow, power=p["power"] * k, scrap_t=p["scrap"] * k), model).iloc[0]
    return now, cur, err, note


def run_forecast(p, model):
    w, source, err = forecast_weather(p)
    stop, stop_err = parse_stops(p["stops"], w["ts"])
    k = 1 - p["cut"] / 100
    plan = plan_from_weather(w, power=p["power"] * k, scrap_t=p["scrap"] * k, stop=stop)
    res = predict_plan(plan, model)
    if p.get("meas") is not None:
        # поправка по последнему замеру: ошибка модели сейчас затухает как ρ^h
        now, *_ = expected_now(p, model)
        hours = np.arange(1, len(res) + 1)
        res["dust_base"] = res["dust_linear"]
        res["dust_linear"] = res["dust_base"] + model.correction(p["meas"] - float(now.dust_linear), hours)
    return res, source, err, stop_err


def daily_summary(res, lv):
    r = res.assign(day=(res.ts - pd.Timedelta("1h")).dt.normalize())
    out = []
    for day, g in r.groupby("day"):
        if len(g) < 6:          # неполные сутки по краям горизонта не показываем
            continue
        i = g.dust_linear.idxmax()
        wd = np.degrees(np.arctan2(np.sin(np.deg2rad(g.wind_dir)).mean(), np.cos(np.deg2rad(g.wind_dir)).mean())) % 360
        out.append({"day": day.strftime("%d.%m"), "wd": ["пн", "вт", "ср", "чт", "пт", "сб", "вс"][day.dayofweek],
                    "hours": len(g), "mean": g.dust_linear.mean(), "max": g.dust_linear.max(),
                    "peak": g.ts[i].strftime("%H:%M"), "tmin": g.T_air.min(), "tmax": g.T_air.max(),
                    "wind": compass_name(wd), "cls": level(g.dust_linear.max(), lv),
                    "stops": int(g.stop.sum()), "extra": bool(g["вне_опыта"].any())})
    return out


def forecast_chart(res, lv, sigma, height=3.6):
    fig, ax = plt.subplots(figsize=(12, height))
    lo = min(res.dust_linear.min() - 2 * sigma, lv["p75"] - .02)
    hi = max(res.dust_linear.max() + 2 * sigma, lv["p95"] + .01)
    ax.axhspan(lo, lv["p75"], color=C["ok"], alpha=.07); ax.axhspan(lv["p75"], lv["p95"], color=C["warn"], alpha=.10)
    ax.axhspan(lv["p95"], hi, color=C["bad"], alpha=.08)
    ax.fill_between(res.ts, res.dust_linear - 2 * sigma, res.dust_linear + 2 * sigma, color=C["exp"], alpha=.12, lw=0, label="диапазон нормы ±2σ")
    if "dust_base" in res:
        ax.plot(res.ts, res.dust_base, color=C["exp"], lw=1.2, ls=":", label="без поправки")
        ax.plot(res.ts, res.dust_linear, color=C["exp"], lw=2, label="ожидаемая пыль (с поправкой по замеру)")
    else:
        ax.plot(res.ts, res.dust_linear, color=C["exp"], lw=2, label="ожидаемая пыль")
    for t in res.ts[res.stop]:
        ax.axvspan(t - pd.Timedelta("1h"), t, color=C["stop"], alpha=.35, lw=0)
    ax.set_ylim(lo, hi); ax.set_ylabel("пыль, мг/Нм³")
    ax.text(.005, lv["p95"], "высокий уровень", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8, color=C["bad"])
    ax.text(.005, lv["p75"], "повышенный уровень", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8, color="#9a6a00")
    t2 = ax.twinx(); t2.plot(res.ts, res.T_air, color=C["temp"], lw=1.2, ls="--", label="температура воздуха")
    t2.set_ylabel("°C", color=C["temp"]); t2.tick_params(axis="y", colors=C["temp"]); t2.grid(False)
    t2.spines["right"].set_visible(True); t2.spines["top"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = t2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=4)
    return png(fig)


# ---------- страницы ----------
@app.route("/")
def live():
    s = state(); model, lv = s["model"], s["lv"]
    p = plan_params()
    now, cur, err, note = expected_now(p, model)
    k = 1 - p["cut"] / 100
    exp, sg = float(now.dust_linear), model.resid_std
    out_of_range = bool(now["вне_опыта"])
    meas = p["meas"]
    if meas is not None:
        z = (meas - exp) / sg
        an, an_cls = (f"ЗАМЕР ВВЕДЁН · {z:+.1f}σ", "bad" if z > 2 else "ok")
        if z > 2:
            an = f"ПРЕВЫШЕНИЕ · {z:+.1f}σ"
    else:
        an, an_cls = "РАСЧЁТ ПО МОДЕЛИ (нет канала прибора)", "warn" if out_of_range else "ok"
    m = dict(live=True, state="РАБОТА" if p["power"] * k > 0 else "ОСТАНОВ",
             mode_name="ПЛАВКИ" if p["power"] * k > 0 else "ОСТАНОВ",
             power=p["power"] * k, scrap=p["scrap"] * k, T=now.T_air, RH=now.RH_air, P=now.P_air, wind=now.wind,
             wind_dir=now.wind_dir, blh=float(now["blh"]) if "blh" in now and pd.notna(now["blh"]) else None,
             expected=exp, lo=exp - 2 * sg, hi=exp + 2 * sg, measured=meas,
             analyzer=an, an_cls=an_cls, weather_note=note)
    p72 = dict(p, hours=72)
    res, source, ferr, stop_err = run_forecast(p72, model)
    # пики пыли в ближайшие 72 ч: не больше одного часа на сутки, чтобы не дублировать один и тот же пик
    rd = res.assign(day=(res.ts - pd.Timedelta("1h")).dt.normalize())
    peaks = rd.loc[rd.groupby("day")["dust_linear"].idxmax()].nlargest(4, "dust_linear").sort_values("ts")
    peaks = [{"ts": r.ts, "v": r.dust_linear, "T": r.T_air, "cls": level(r.dust_linear, lv)} for r in peaks.itertuples()]
    q, (gname, gcls) = s["q"], s["grade"]
    return render_template("live.html", active="live", m=m, p=p, lv=lv, cls=level(exp, lv), online=cur is not None,
                           err=err, out_of_range=out_of_range, q=q, gname=gname, gcls=gcls, sigma=sg,
                           chart=forecast_chart(res, lv, sg, 3.0), days=daily_summary(res, lv)[:4], source=source, peaks=peaks,
                           wind_dusty=wind_effect(model, make_plan()).sort_values("пыль_ср").iloc[-1]["ветер_из"],
                           coefs=model.coefficients())


@app.route("/forecast")
def forecast():
    s = state(); model, lv = s["model"], s["lv"]
    p = plan_params()
    res, source, err, stop_err = run_forecast(p, model)
    sg = model.resid_std
    peaks = res.nlargest(8, "dust_linear").sort_values("ts")
    table = res.assign(cls=[level(v, lv) for v in res.dust_linear])
    # сценарии НМУ на тот же горизонт
    base = dict(p, cut=0, meas=None)   # сценарии сравниваем без поправки: важна разница между ними
    scen = []
    for c in (0, 10, 20, 30, 50):
        r, *_ = run_forecast(dict(base, cut=c), model)
        scen.append({"cut": c, "mean": r.dust_linear.mean(), "max": r.dust_linear.max()})
    for sc in scen:
        sc["d"] = (sc["mean"] / scen[0]["mean"] - 1) * 100
    lo_t, hi_t = model.ranges["T_air"]
    return render_template("forecast.html", active="fc", p=p, lv=lv, res=table, peaks=peaks, source=source, err=err, rho=model.ar_rho,
                           stop_err=stop_err, chart=forecast_chart(res, lv, sg), days=daily_summary(res, lv), sigma=sg,
                           scen=scen, extra=bool(res["вне_опыта"].any()), t_lo=lo_t, t_hi=hi_t,
                           online=not err, level=lambda v: level(v, lv))


@app.route("/archive")
def archive():
    s = state(); mon, model, lv, X = s["mon"], s["model"], s["lv"], s["X"]
    date = request.args.get("date", "2022-02-15")
    hour = int(request.args.get("hour", 12))
    day = mon[(mon.ts - pd.Timedelta("1h")).dt.strftime("%Y-%m-%d") == date]
    if day.empty:
        return render_template("archive.html", active="arch", date=date, hour=hour, empty=True)
    hrow = day[day.ts.dt.hour == hour % 24]
    r = (hrow if len(hrow) else day).iloc[0]
    x = X.loc[r.name]
    sg = model.resid_std
    lab, cls = SIGNAL.get(r.signal, (r.signal, "off"))
    exp = None if pd.isna(r.dust_linear) else float(r.dust_linear)
    corr = None if pd.isna(r.dust_corr) else float(r.dust_corr)
    m = dict(live=False, corr=corr, state="РАБОТА" if r.power > 0 else "ОСТАНОВ", mode_name=r["mode"], power=r.power, scrap=r.scrap_t,
             T=x.T_air, RH=x.RH_air, P=x.P_air, wind=x.wind, wind_dir=x.wind_dir, blh=float(x.blh), expected=exp,
             lo=None if exp is None else exp - 2 * sg, hi=None if exp is None else exp + 2 * sg,
             measured=None if pd.isna(r.dust_ok) else float(r.dust_ok), analyzer=lab + (f" · {r.z:+.1f}σ" if pd.notna(r.z) else ""),
             an_cls=cls, weather_note=f"архив Open-Meteo · час, кончающийся в {r.ts:%H:%M}")
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.fill_between(day.ts, day.dust_linear - 2 * sg, day.dust_linear + 2 * sg, color=C["exp"], alpha=.12, lw=0, label="норма ±2σ")
    ax.plot(day.ts, day.dust_linear, color=C["exp"], lw=2, label="ожидаемая (модель)")
    ax.plot(day.ts, day.dust_corr, color=C["ok"], lw=1.4, ls="--", label="с поправкой по прошлому замеру")
    ax.plot(day.ts, day.dust_ok, "o-", color=C["fact"], ms=4, lw=1.4, label="замер")
    bad = day[day.signal == "ПРЕВЫШЕНИЕ"]
    ax.scatter(bad.ts, bad.dust_ok, s=70, facecolor="none", edgecolor=C["bad"], lw=2, label="превышение", zorder=5)
    ax.axvline(r.ts, color="#16202a", lw=1, ls=":")
    ax.set_ylabel("пыль, мг/Нм³"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(ncol=5, loc="lower right", bbox_to_anchor=(1, 1.0))
    rows = [dict(ts=t.ts, mode=t["mode"], power=t.power, T=t.T_air, wd=t.wind_dir, meas=t.dust_ok, exp=t.dust_linear, corr=t.dust_corr, z=t.z,
                 sig=SIGNAL.get(t.signal, (t.signal, "off"))) for _, t in day.iterrows()]
    counts = day.signal.value_counts().to_dict()
    return render_template("archive.html", active="arch", date=date, hour=hour, m=m, lv=lv, chart=png(fig), rows=rows,
                           counts={SIGNAL.get(k, (k, 'off'))[0]: (v, SIGNAL.get(k, (k, 'off'))[1]) for k, v in counts.items()},
                           cls=level(exp, lv), empty=False, fnum=fnum)


@app.route("/quality")
def quality_page():
    s = state(); cv, q, model = s["cv"], s["q"], s["model"]
    qn = quality(cv, "nn")
    ape = (cv.linear - cv.y).abs() / cv.y * 100
    # 1) замер против прогноза
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(cv.y, cv.linear, s=5, alpha=.35, color=C["exp"], lw=0)
    lim = [cv[["y", "linear"]].min().min(), cv[["y", "linear"]].max().max()]
    ax.plot(lim, lim, color="#16202a", lw=1, label="идеально")
    ax.fill_between(lim, [v * .9 for v in lim], [v * 1.1 for v in lim], color=C["ok"], alpha=.15, label="±10 %")
    ax.set_xlabel("замер, мг/Нм³"); ax.set_ylabel("прогноз модели, мг/Нм³"); ax.legend(loc="upper left")
    scatter = png(fig)
    # 2) распределение ошибки в %
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bins = np.arange(0, 41, 2.5)
    ax.hist(ape.clip(upper=40), bins=bins, color=C["exp"], edgecolor="#fff")
    for v, c, t in [(10, C["ok"], "10 %"), (20, C["warn"], "20 %")]:
        ax.axvline(v, color=c, lw=2); ax.text(v + .5, ax.get_ylim()[1] * .92, t, color=c, fontsize=9, fontweight="bold")
    ax.set_xlabel("ошибка прогноза часа, % (40 = 40 и больше)"); ax.set_ylabel("часов")
    hist = png(fig)
    # 3) по месяцам
    bm = by_month(cv)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    names = ["янв", "фев", "мар", "апр", "май", "июн", "июл"]
    bars = ax.bar([names[m - 1] for m in bm["месяц"]], bm["ср. ошибка, %"], color=[C["ok"] if v <= 10 else C["warn"] for v in bm["ср. ошибка, %"]])
    ax.bar_label(bars, fmt="%.1f %%", fontsize=8); ax.set_ylabel("средняя ошибка, %"); ax.axhline(10, color="#16202a", lw=.8, ls=":")
    months = png(fig)
    # 4) пример: неделя замер против прогноза (+ прогноз на 3 ч вперёд с поправкой)
    sh = cv.set_index("ts")[["y", "linear"]].asfreq("h")
    sh["corr3"] = sh.linear + model.ar_rho ** 3 * (sh.y - sh.linear).shift(3)
    wk = sh.loc["2022-03-14":"2022-03-20 23:00"]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(wk.index, wk.y, color=C["fact"], lw=1.3, label="замер")
    ax.plot(wk.index, wk.linear, color=C["exp"], lw=1.6, label="модель без поправки (месяц не видела)")
    ax.plot(wk.index, wk.corr3, color=C["ok"], lw=1.4, ls="--", label="с поправкой по замеру 3 ч назад")
    ax.set_ylabel("пыль, мг/Нм³"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m")); ax.legend(loc="upper left", ncol=3)
    week = png(fig)
    gname, gcls = s["grade"]
    return render_template("quality.html", active="q", q=q, qn=qn, gname=gname, gcls=gcls, scatter=scatter, hist=hist,
                           months=months, week=week, bm=bm, metrics=model.metrics, coefs=model.coefficients(),
                           corr=correction_table(cv, model.ar_rho), rho=model.ar_rho)


if __name__ == "__main__":
    app.run(debug=False)
