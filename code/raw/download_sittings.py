"""Where each plenary sitting took place (Strasbourg, Brussels), from the Parliament's Open Data API.

Every plenary sitting is an activity MTG-PL-YYYY-MM-DD of the meetings dataset, with the property
vcard:hasLocality (1..1), a place of the EU authority table: FRA_SXB (Strasbourg), BEL_BRU (Brussels).
Dataset description: https://europarl.github.io/eli-ep/3.0.0/api/meetings

One request per year: https://data.europarl.europa.eu/api/v2/meetings?year=YYYY (all sittings of the year, with
their place), for every year of data/translation/unified.parquet. The API has the sittings only from 2014.
Sitting days of unified that the API does not list get status "not in API". Run it from the Mac (the API times
out from the JRC server); it takes about a minute.

    uv run python code/raw/download_sittings.py

Out: data/raw/api/sittings.csv   date, event_id, status (ok / not in API), locality (place codes, ';'-separated),
     in_unified (the day has speeches in unified)
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
import config as cfg  # noqa: E402

BASE = "https://data.europarl.europa.eu/api/v2/meetings"
OUT = cfg.DATA / "raw" / "api" / "sittings.csv"
PLACE = re.compile(r"authority/place/([A-Z]{3}_[A-Z0-9]{3})")
EVENT = re.compile(r"MTG-PL-(\d{4}-\d{2}-\d{2})")
LIMIT = 500
PAUSE = 1.2


def get(params: dict, tries: int = 6) -> dict:
    url = BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"Accept": "application/ld+json",
                                               "User-Agent": "europarl-emi-research"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode("utf-8")
                return json.loads(body) if body.strip() else {"data": []}
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return {"data": []}
            if e.code == 429:
                print("HTTP 429: waiting 5 minutes", flush=True)
                time.sleep(300)
                continue
            time.sleep(5 * (i + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def sittings_of(year: int) -> list[dict]:
    """All plenary sittings of a year with their place codes (pages of LIMIT items)."""
    rows, offset = [], 0
    while True:
        items = get({"year": year, "format": "application%2Fld%2Bjson", "offset": offset, "limit": LIMIT})
        items = items.get("data", [])
        for it in items:
            text = json.dumps(it)
            ident = str(it.get("activity_id") or it.get("id") or it.get("@id") or "") if isinstance(it, dict) else ""
            m = EVENT.search(ident) if ident else EVENT.search(text)  # the sitting's own id, not a referenced one
            if not m:  # not a plenary sitting
                continue
            rows.append({"date": m.group(1), "event_id": f"MTG-PL-{m.group(1)}",
                         "locality": ";".join(sorted(set(PLACE.findall(text))))})
        time.sleep(PAUSE)
        if len(items) < LIMIT:
            return rows
        offset += LIMIT


def main() -> None:
    days = pd.read_parquet(cfg.UNIFIED, columns=["date"]).date.dropna().dt.strftime("%Y-%m-%d").unique()
    years = sorted({int(d[:4]) for d in days})
    print(f"sitting days in {cfg.UNIFIED.name}: {len(days):,}; years {years[0]}-{years[-1]}")
    api = []
    for y in years:
        rows = sittings_of(y)
        api += rows
        places = pd.Series([r["locality"] for r in rows]).value_counts().to_dict() if rows else {}
        print(f"{y}: {len(rows):3d} sittings in the API {places}", flush=True)
    a = pd.DataFrame(api, columns=["date", "event_id", "locality"]).drop_duplicates("date")
    a["status"] = "ok"
    miss = pd.DataFrame({"date": sorted(set(days) - set(a.date))})
    miss["event_id"], miss["locality"], miss["status"] = "MTG-PL-" + miss.date, "", "not in API"
    s = pd.concat([a, miss], ignore_index=True).sort_values("date").reset_index(drop=True)
    s["in_unified"] = s.date.isin(set(days))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s[["date", "event_id", "status", "locality", "in_unified"]].to_csv(OUT, index=False)
    print(f"\n{OUT.relative_to(cfg.ROOT)}: {len(s):,} days")
    s["year"] = s.date.str[:4]
    print("\nsitting days of unified, by year and place:")
    print(pd.crosstab(s[s.in_unified].year, s[s.in_unified].locality.replace("", "(not in API)")).to_string())
    extra = s[~s.in_unified]
    if len(extra):
        print(f"\nsittings in the API without speeches in unified: {len(extra)} "
              f"(from {extra.date.min()} to {extra.date.max()})")


if __name__ == "__main__":
    main()
