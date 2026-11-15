"""Наблюдения метеостанций Южного Урала за 2022 г.
Источник: NOAA NCEI, Integrated Surface Database (ISD-Lite) — https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
Файлы (официальное зеркало NOAA Open Data): https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2022/<USAF>-<WBAN>-2022.gz ; список станций: isd-history.csv
Запуск: python experiments/fetch_isd.py  -> data/isd/
"""
import csv, io, time, urllib.request
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "data" / "isd"; out.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ncei.noaa.gov/pub/data/noaa/"
BOX = (52.5, 56.3, 57.0, 63.5)      # Южный Урал: Челябинская обл., восток Башкирии, север Оренбургской обл.


def get(url):
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=90).read()
        except Exception as e:
            print("retry", url.rsplit("/", 1)[-1], e, flush=True); time.sleep(10)
    return None


h = out / "isd-history.csv"
if not h.exists():
    h.write_bytes(get(BASE + "isd-history.csv"))
rows = []
for r in csv.DictReader(io.StringIO(h.read_text(encoding="latin-1"))):
    try:
        lat, lon = float(r["LAT"]), float(r["LON"])
    except ValueError:
        continue
    if BOX[0] <= lat <= BOX[1] and BOX[2] <= lon <= BOX[3] and r["BEGIN"] <= "20220101" and r["END"] >= "20221231":
        rows.append(r)
print(len(rows), "станций")
with open(out / "stations.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["usaf", "wban", "name", "lat", "lon", "elev"])
    for r in rows:
        w.writerow([r["USAF"], r["WBAN"], r["STATION NAME"], r["LAT"], r["LON"], r["ELEV(M)"]])
for r in rows:
    name = f"{r['USAF']}-{r['WBAN']}-2022.gz"
    if (out / name).exists():
        continue
    b = get("https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2022/" + name)   # официальное зеркало NOAA Open Data
    if b:
        (out / name).write_bytes(b); print("ok", r["STATION NAME"], r["LAT"], r["LON"], len(b), flush=True)
    time.sleep(0.5)
