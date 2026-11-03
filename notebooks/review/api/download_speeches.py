"""REVIEW ONLY. Download plenary speeches from the European Parliament Open Data API (v2, /speeches).

The API has complete data from July 2021. For each weekday, all speeches of that sitting day,
100 per request. Kept for each speech: the original-language text and, for speeches not given in
English, the English machine translation provided by the API (xml:lang "xx-t-en-mtec").
Resumable: one file per day; days without a sitting give an empty file.

    uv run python notebooks/review/api/download_speeches.py                                    # 2021-07-01 to today
    uv run python notebooks/review/api/download_speeches.py --start 2025-01-20 --end 2025-01-20   # test

Output: data/raw/api/speeches/YYYY-MM-DD.jsonl (one speech per line)
"""

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "api" / "speeches"
BASE = "https://data.europarl.europa.eu/api/v2/speeches"
PAGE = 100
PAUSE = 1.2  # seconds between requests: about 250 per 5 minutes, well under the API limit
LANG = {"BUL": "bg", "CES": "cs", "DAN": "da", "DEU": "de", "ELL": "el", "ENG": "en", "EST": "et", "FIN": "fi",
        "FRA": "fr", "GLE": "ga", "HRV": "hr", "HUN": "hu", "ITA": "it", "LAV": "lv", "LIT": "lt", "MLT": "mt",
        "NLD": "nl", "POL": "pl", "POR": "pt", "RON": "ro", "SLK": "sk", "SLV": "sl", "SPA": "es", "SWE": "sv"}


def get(params, tries=10):
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/ld+json", "User-Agent": "europarl-emi-research"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
                return json.loads(body) if body.strip() else {"data": []}
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return {"data": []}
            wait = 300 if e.code == 429 else 15 * (k + 1)
            tqdm.write(f"  HTTP {e.code}, retry in {wait}s")
        except Exception as e:
            wait = 15 * (k + 1)
            tqdm.write(f"  {type(e).__name__}: {e}, retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def plain(xml, sep=" "):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", sep, xml or ""))).strip()


def last(uri):
    return str(uri).rsplit("/", 1)[-1]


def is_mt(xml):
    return any("-t-" in v for v in re.findall(r'xml:lang="([^"]+)"', xml or ""))


def body_text(xml):
    body = re.search(r"<blockContainer[^>]*>(.*)</blockContainer>", xml or "", re.S)
    return plain(body.group(1)) if body else (plain(xml) if xml else None)


def original_fragment(rec):
    """The original-language XML: 'mul' if the speaker switched language, else the original language."""
    frags = rec.get("api:xmlFragment") or {}
    codes = [LANG.get(last(x), last(x).lower()) for x in rec.get("originalLanguage", [])]
    for key in (["mul"] if "mul" in frags else []) + codes + list(frags):
        x = frags.get(key)
        if x and not is_mt(x):
            return key, x
    return None, None


def english_mt(rec):
    """English machine translation of a non-English speech, as provided by the API."""
    x = (rec.get("api:xmlFragment") or {}).get("en")
    return body_text(x) if x and is_mt(x) else None


def slim(r):
    rec = (r.get("recorded_in_a_realization_of") or [{}])[0]
    key, x = original_fragment(rec)
    head = re.search(r"<from[^>]*>(.*?)</from>", x or "", re.S)
    part = r.get("had_participation") or {}
    if isinstance(part, list):
        part = part[0] if part else {}
    return {
        "speech_id": rec.get("identifier") or r.get("activity_id"),
        "date": r.get("activity_date"),
        "start": r.get("activity_start_date"),
        "activity_type": last(r.get("had_activity_type", "")),
        "role": last(part.get("participation_role", "")),
        "person_id": ",".join(last(p) for p in part.get("had_participant_person", [])),
        "agenda_item": rec.get("is_part_of"),
        "agenda_title_en": (r.get("activity_label") or {}).get("en"),
        "original_language": ",".join(LANG.get(last(x), last(x).lower()) for x in rec.get("originalLanguage", [])),
        "fragment_lang": key,
        "header": plain(head.group(1), sep="") if head else None,
        "text": body_text(x),
        "text_en_mt": english_mt(rec),
        "xml": x,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01", help="YYYY-MM-DD (default: 2021-07-01, start of complete API data)")
    ap.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    print(f"Downloading {start} to {end} ({len(days):,} weekdays) into {OUT}", flush=True)
    n_speeches = n_new_days = 0
    bar = tqdm(days, desc="Days", unit="day", dynamic_ncols=True)
    for day in bar:
        bar.set_postfix_str(f"{day} | downloaded this run: {n_speeches:,} speeches")
        path = OUT / f"{day}.jsonl"
        if path.exists():
            continue
        rows, offset, other_dates = [], 0, 0
        while True:
            d = get({"sitting-date": day.isoformat(), "format": "application/ld+json", "language-filter": "mul",
                     "include-output": "xml_fragment", "offset": offset, "limit": PAGE})
            time.sleep(PAUSE)
            recs = d.get("data", [])
            for r in recs:
                if r.get("activity_date") == day.isoformat():
                    rows.append(slim(r))
                else:
                    other_dates += 1
            if len(recs) < PAGE:
                break
            offset += PAGE
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        tmp.rename(path)
        n_new_days += 1
        n_speeches += len(rows)
        if rows or other_dates:
            en = sum(x["original_language"] == "en" for x in rows)
            mt = sum(bool(x["text_en_mt"]) for x in rows)
            tqdm.write(f"{day}  {len(rows):4d} speeches, {en:4d} in English, {mt:4d} with English MT"
                       + (f"  (WARNING: {other_dates} records from other dates)" if other_dates else ""))
    bar.close()
    print(f"done: {n_new_days:,} new days, {n_speeches:,} speeches", flush=True)


if __name__ == "__main__":
    main()
