"""Быстрые проверки: python -m pytest tests  (или python tests/test_pipeline.py)"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ecometeo.data import load_dataset
from ecometeo.features import build_features, training_rows, TARGET
from ecometeo.forecast import make_plan

DF = load_dataset(ROOT / "data" / "data.xlsx", ROOT / "data" / "weather_region_2022_hourly.json")


def test_time_axis_is_hourly_and_ordered():
    assert len(DF) == 8760
    assert DF["ts"].is_monotonic_increasing
    assert (DF["ts"].diff().dropna().dt.total_seconds() == 3600).all()


def test_status_counts():
    c = DF["status"].value_counts()
    assert c["OK"] == 4203 and c["OFF"] == 4145 and c["FAULT"] == 412


def test_bad_hours_never_used_for_training():
    X = build_features(DF)
    rows = training_rows(X)
    assert X.loc[rows, "status"].eq("OK").all()
    assert X.loc[rows, TARGET].notna().all()


def test_cleaning_rules():
    assert DF["td_min"].max() <= 60
    assert DF["power"].min() >= 0
    assert DF.loc[DF.status != "OK", "dust_ok"].isna().all()


def test_cycles_and_modes():
    assert DF["cycle"].max() == 74
    starts = (DF["mode"].eq("ОСТАНОВ") & DF["mode"].shift().ne("ОСТАНОВ")).sum()
    assert starts == 29


def test_hourly_weather_joined():
    ok = DF[DF.status == "OK"]
    assert ok[["T_air", "RH_air", "P_air", "wind", "wind_dir"]].notna().all().all()
    # погода почасовая: внутри суток температура меняется
    assert DF.groupby(DF.ts.dt.date)["T_air"].nunique().median() > 12
    assert DF["wind_dir"].dropna().between(0, 360).all()


def test_plan_shapes():
    p = make_plan(stop_hours=(10, 11))
    assert len(p) == 24 and p.loc[p.hour.isin([10, 11]), "power"].eq(0).all()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("OK", name)
