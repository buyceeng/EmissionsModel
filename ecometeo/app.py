"""ЭкоМетео — черновая версия (v1): одна страница с обзором года.

Запуск:
    python -m ecometeo.cli train    # один раз: очистка данных, проверка и обучение модели
    python app.py                   # http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template

from ecometeo.models import DustModel
from ecometeo.monitor import run_monitor

ART = Path(__file__).resolve().parent / "artifacts"
app = Flask(__name__)


def png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    if not (ART / "model.joblib").exists():
        return "Сначала обучите модель: <code>python -m ecometeo.cli train</code>", 503
    model = DustModel.load(ART / "model.joblib")
    X = pd.read_pickle(ART / "dataset.pkl")
    mon = run_monitor(X, model)

    # пыль по дням: замер и ожидаемая по модели (модель — только там, где есть замер)
    m = mon.assign(dust_linear=mon.dust_linear.where(mon.dust_ok.notna())).set_index("ts")
    daily = m.resample("D")[["dust_ok", "dust_linear"]].mean()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(daily.index, daily.dust_ok, color="#6250d6", lw=1.6, label="замер")
    ax.plot(daily.index, daily.dust_linear, color="#2a78d6", lw=1.2, ls="--", label="ожидаемая по модели")
    ax.set_ylabel("пыль, мг/Нм³"); ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.grid(alpha=.3)

    metrics = pd.DataFrame(model.metrics).T
    cols = [c for c in ("r", "R2", "MAE") if c in metrics]
    alarms = (mon[mon.signal == "ПРЕВЫШЕНИЕ"].sort_values("z", ascending=False).head(15)
              [["ts", "mode", "power", "T_air", "dust_ok", "dust_linear", "z"]]
              .rename(columns={"ts": "время", "mode": "режим", "power": "мощность, МВт·ч", "T_air": "t воздуха, °C",
                               "dust_ok": "замер", "dust_linear": "ожидается", "z": "отклонение, σ"}))
    return render_template(
        "index.html", chart=png(fig), counts=mon.signal.value_counts().to_dict(), sigma=round(model.resid_std, 4),
        metrics=metrics[cols].round(3).to_html(classes="tbl"), alarms=alarms.round(3).to_html(classes="tbl", index=False))


if __name__ == "__main__":
    app.run(debug=False)
