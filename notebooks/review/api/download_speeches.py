"""REVIEW ONLY. Download plenary speeches from the European Parliament Open Data API (v2, /speeches).

For each weekday, all speeches of that sitting day, 100 per request. Only the original-language
text is kept: the other language versions are machine translations (xml:lang "xx-t-yy-mtec").
Resumable: one file per day; days without a sitting give an empty file.

    uv run python notebooks/review/api/download_speeches.py --start 2025-01-20 --end 2025-01-21   # test
    uv run python notebooks/review/api/download_speeches.py --start 2019-07-01 --end 2026-11-03

Output: data/raw/api/speeches/YYYY-MM-DD.jsonl (one speech per line)
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "api" / "speeches"
BASE = "https://data.europarl.europa.eu/api/v2/speeches"
PAGE = 100
PAUSE = 0.7  # seconds between requests: about 430 per 5 minutes, under the API limit of 500
LANG = {"BUL": "bg", "CES": "cs", "DAN": "da", "DEU": "de", "ELL": "el", "ENG": "en", "EST": "et", "FIN": "fi",
        "FRA": "fr", "GLE": "ga", "HRV": "hr", "HUN": "hu", "ITA": "it", "LAV": "lv", "LIT": "lt", "MLT": "mt",
        "NLD": "nl", "POL": "pl", "POR": "pt", "RON": "ro", "SLK": "sk", "SLV": "sl", "SPA": "es", "SWE": "sv"}


def get(params, tries=6):
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
            wait = 120 if e.code == 429 else 15 * (k + 1)
            print(f"  HTTP {e.code}, retry in {wait}s", flush=True)
        except Exception as e:
            wait = 15 * (k + 1)
            print(f"  {type(e).__name__}: {e}, retry in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def plain(xml, sep=" "):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", sep, xml or "")).strip()


def last(uri):
    return str(uri).rsplit("/", 1)[-1]


def original_fragment(rec):
    """The original-language XML: 'mul' if the speaker switched language, else the original language."""
    frags = rec.get("api:xmlFragment") or {}
    codes = [LANG.get(last(x), last(x).lower()) for x in rec.get("originalLanguage", [])]
    for key in (["mul"] if "mul" in frags else []) + codes + list(frags):
        x = frags.get(key)
        if x and not any("-t-" in v for v in re.findall(r'xml:lang="([^"]+)"', x)):
            return key, x
    return None, None


def slim(r):
    rec = (r.get("recorded_in_a_realization_of") or [{}])[0]
    key, x = original_fragment(rec)
    head = re.search(r"<from[^>]*>(.*?)</from>", x or "", re.S)
    body = re.search(r"<blockContainer[^>]*>(.*)</blockContainer>", x or "", re.S)
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
        "text": plain(body.group(1)) if body else (plain(x) if x else None),
        "xml": x,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    day, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    while day <= end:
        path = OUT / f"{day}.jsonl"
        if day.weekday() < 5 and not path.exists():
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
            if rows or other_dates:
                en = sum(x["original_language"] == "en" for x in rows)
                print(f"{day}  {len(rows):4d} speeches, {en:4d} in English"
                      + (f"  (WARNING: {other_dates} records from other dates)" if other_dates else ""), flush=True)
        day += timedelta(days=1)
    print("done")


if __name__ == "__main__":
    main()
