"""Translate the non-English speeches of the corpus into English with eTranslation
(European Commission machine translation, https://language-tools.ec.europa.eu).

eTranslation is asynchronous: each request returns an id, and the translation is later POSTed to a
callback URL that must be reachable from the internet. This script runs that callback server on this
computer (127.0.0.1:8765) and needs a public HTTPS address that forwards to it, for example:
    cloudflared tunnel --url http://localhost:8765     (prints https://<random>.trycloudflare.com)

Credentials are read from environment variables and never written to files:
    export ETRANSLATION_APP=...          # application name
    export ETRANSLATION_PASSWORD=...
    export ETRANSLATION_USER=...         # optional, defaults to the application name

    uv run python -m europarl_emi.etranslation.translate prepare --from-year 2013 --to-year 2023
    uv run python -m europarl_emi.etranslation.translate run --public-url https://<...>.trycloudflare.com --limit 5
    uv run python -m europarl_emi.etranslation.translate status
    uv run python -m europarl_emi.etranslation.translate export

Data (outside git): data/interim/etranslation/
    segments.sqlite        one row per text segment: source, request id, status, translation
    translations.parquet   one row per speech: guid, language, year, text_en  (written by `export`)
    callbacks_unparsed.log callbacks that could not be matched or read (for debugging)
"""

import argparse
import json
import os
import re
import secrets
import sqlite3
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from europarl_emi import config as cfg

ROOT = cfg.ROOT
CORPUS = cfg.RAW_CORPUS / "all_years_combined.csv.gzip"
DIR = cfg.INTERIM / "etranslation"
DB = DIR / "segments.sqlite"
OUT = DIR / "translations.parquet"
UNPARSED = DIR / "callbacks_unparsed.log"
TOKEN_FILE = DIR / "callback_token.txt"

ENDPOINT = "https://webgate.ec.europa.eu/etranslation/si/translate"
DOMAIN = "GEN"          # general domain
PORT = 8765
MAX_CHARS = 4500        # longest segment sent in one request (check the limit in the eTranslation docs)
MAX_IN_FLIGHT = 20      # requests waiting for their translation at the same time
PAUSE = 0.5             # seconds between two submissions
RESEND_AFTER = 3600     # seconds after which an unanswered request is sent again
EU = {"bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl",
      "pl", "pt", "ro", "sk", "sl", "sv"}

APP = os.environ.get("ETRANSLATION_APP")
PASSWORD = os.environ.get("ETRANSLATION_PASSWORD")
USER = os.environ.get("ETRANSLATION_USER") or APP
LOCK = threading.Lock()


# ---------------------------------------------------------------- storage

def db():
    DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, check_same_thread=False, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS segments (
        seg_id TEXT PRIMARY KEY, guid TEXT, seg_no INTEGER, n_segs INTEGER, lang TEXT, year INTEGER,
        source TEXT, status TEXT DEFAULT 'new', request_id TEXT, sent_at REAL,
        translated TEXT, error TEXT, received_at REAL)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_status ON segments(status)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_request ON segments(request_id)")
    return con


def token():
    DIR.mkdir(parents=True, exist_ok=True)
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_hex(16))
    return TOKEN_FILE.read_text().strip()


def split_text(text, max_chars=MAX_CHARS):
    """Segments of at most max_chars characters, cut at sentence ends (or at spaces for very long sentences)."""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return [text] if text else []
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?;])\s+", text):
        while len(sent) > max_chars:
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut])
            sent = sent[cut:].lstrip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


# ---------------------------------------------------------------- prepare

def prepare(args):
    con = db()
    n_texts = n_segs = n_chars = 0
    by_lang = {}
    reader = pd.read_csv(CORPUS, compression="gzip", usecols=["guid", "language", "pubdate", "description"],
                         dtype=str, chunksize=50_000)
    for chunk in reader:
        chunk["lang"] = chunk["language"].str.strip().str.lower()
        chunk["year"] = pd.to_datetime(chunk["pubdate"], errors="coerce", utc=True, format="mixed").dt.year
        keep = chunk["lang"].isin(EU) & chunk["description"].notna() & chunk["year"].between(args.from_year, args.to_year)
        rows = []
        for guid, lang, year, text in chunk.loc[keep, ["guid", "lang", "year", "description"]].itertuples(index=False):
            segs = split_text(text)
            rows += [(f"{guid}#{i}", guid, i, len(segs), lang, int(year), s) for i, s in enumerate(segs)]
            n_texts += 1
            n_segs += len(segs)
            n_chars += sum(map(len, segs))
            by_lang[lang] = by_lang.get(lang, 0) + 1
        with LOCK:
            con.executemany("INSERT OR IGNORE INTO segments (seg_id, guid, seg_no, n_segs, lang, year, source) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            con.commit()
    print(f"Texts to translate ({args.from_year}-{args.to_year}): {n_texts:,}  segments: {n_segs:,}  "
          f"characters: {n_chars:,}")
    print("By language:", dict(sorted(by_lang.items(), key=lambda kv: -kv[1])))


# ---------------------------------------------------------------- callback server

def parse_payload(raw, ctype):
    text = raw.decode("utf-8", errors="replace")
    if "json" in ctype or text.lstrip().startswith("{"):
        try:
            return {k: (v[0] if isinstance(v, list) else v) for k, v in json.loads(text).items()}
        except ValueError:
            pass
    return {k: v[0] for k, v in urllib.parse.parse_qs(text, keep_blank_values=True).items()}


def make_handler(con, tok):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            split = urllib.parse.urlsplit(self.path)
            path = split.path.rstrip("/")
            kind = "ok" if path == f"/callback/{tok}" else "error" if path == f"/error/{tok}" else None
            if kind is None:
                self.send_response(403)
                self.end_headers()
                return
            p = parse_payload(raw, self.headers.get("Content-Type", ""))
            p.update({k: v[0] for k, v in urllib.parse.parse_qs(split.query).items()})
            ref, rid = p.get("external-reference"), p.get("request-id")
            with LOCK:
                row = con.execute("SELECT seg_id FROM segments WHERE seg_id = ?", (ref,)).fetchone() if ref else None
                if row is None and rid:
                    row = con.execute("SELECT seg_id FROM segments WHERE request_id = ?", (str(rid),)).fetchone()
                if row is None or (kind == "ok" and "translated-text" not in p):
                    with open(UNPARSED, "a", encoding="utf-8") as f:
                        f.write(f"{time.ctime()} {kind} {self.headers.get('Content-Type')} {raw[:2000]!r}\n")
                elif kind == "ok":
                    con.execute("UPDATE segments SET status = 'done', translated = ?, received_at = ? WHERE seg_id = ?",
                                (p["translated-text"], time.time(), row[0]))
                else:
                    con.execute("UPDATE segments SET status = 'error', error = ?, received_at = ? WHERE seg_id = ?",
                                (f"{p.get('error-code')}: {p.get('error-message')}", time.time(), row[0]))
                con.commit()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *args):
            pass

    return Handler


# ---------------------------------------------------------------- run

def ssl_context():
    """Certificates of the operating system (macOS keychain): uv's Python cannot verify the eTranslation server alone."""
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def make_opener():
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, ENDPOINT, APP, PASSWORD)
    return urllib.request.build_opener(urllib.request.HTTPDigestAuthHandler(mgr),
                                       urllib.request.HTTPSHandler(context=ssl_context()))


def submit(opener, seg_id, lang, text, callback, error_cb):
    body = {
        "priority": 0,
        "externalReference": seg_id,
        "callerInformation": {"application": APP, "username": USER},
        "textToTranslate": text,
        "sourceLanguage": lang.upper(),
        "targetLanguages": ["EN"],
        "domain": DOMAIN,
        "requesterCallback": callback,
        "errorCallback": error_cb,
    }
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with opener.open(req, timeout=60) as r:
        return r.read().decode("utf-8").strip().strip('"')


def counts(con):
    with LOCK:
        return dict(con.execute("SELECT status, COUNT(*) FROM segments GROUP BY status").fetchall())


def run(args):
    if not APP or not PASSWORD:
        sys.exit("Set ETRANSLATION_APP and ETRANSLATION_PASSWORD first (export ...).")
    con, tok = db(), token()
    if args.retry_errors:
        with LOCK:
            con.execute("UPDATE segments SET status = 'new', error = NULL WHERE status = 'error'")
            con.commit()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(con, tok))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = args.public_url.rstrip("/")
    callback, error_cb = f"{base}/callback/{tok}", f"{base}/error/{tok}"
    print(f"Callback server on 127.0.0.1:{PORT}; eTranslation will answer at {base}/callback/...", flush=True)
    opener = make_opener()
    sent, last_print, waiting_since = 0, 0.0, None
    try:
        while True:
            now = time.time()
            with LOCK:
                in_flight = con.execute("SELECT COUNT(*) FROM segments WHERE status = 'sent' AND sent_at > ?",
                                        (now - RESEND_AFTER,)).fetchone()[0]
            room = MAX_IN_FLIGHT - in_flight
            if args.limit is not None:
                room = min(room, args.limit - sent)
            todo = []
            if room > 0:
                with LOCK:
                    todo = con.execute(
                        "SELECT seg_id, lang, source FROM segments WHERE status = 'new' "
                        "OR (status = 'sent' AND sent_at <= ?) ORDER BY year, seg_id LIMIT ?",
                        (now - RESEND_AFTER, room)).fetchall()
            for seg_id, lang, source in todo:
                try:
                    rid = submit(opener, seg_id, lang, source, callback, error_cb)
                except Exception as e:
                    print(f"  submission failed ({type(e).__name__}: {e}); pausing 60 s", flush=True)
                    time.sleep(60)
                    break
                with LOCK:
                    if re.fullmatch(r"-\d+", rid):
                        con.execute("UPDATE segments SET status = 'error', error = ? WHERE seg_id = ?",
                                    (f"submission refused, code {rid}", seg_id))
                    else:
                        con.execute("UPDATE segments SET status = 'sent', request_id = ?, sent_at = ? WHERE seg_id = ?",
                                    (rid, time.time(), seg_id))
                    con.commit()
                sent += 1
                time.sleep(PAUSE)
            c = counts(con)
            if time.time() - last_print > 30:
                print(f"{time.strftime('%H:%M:%S')}  sent this run: {sent:,}  status: {c}", flush=True)
                last_print = time.time()
            done_sending = c.get("new", 0) == 0 or (args.limit is not None and sent >= args.limit)
            if done_sending and c.get("sent", 0) == 0:
                print("Finished: nothing left to send or to wait for.", flush=True)
                break
            if done_sending:
                waiting_since = waiting_since or time.time()
                if time.time() - waiting_since > args.max_wait:
                    print(f"Stopped waiting after {args.max_wait} s; rerun `run` later to resend unanswered requests.")
                    break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nInterrupted: progress is saved; run the same command to continue.")
    finally:
        server.shutdown()
    print("Status:", counts(con))


# ---------------------------------------------------------------- status and export

def status(args):
    con = db()
    print("Segments by status:", counts(con))
    with LOCK:
        rows = con.execute("SELECT lang, SUM(status = 'done'), COUNT(*) FROM segments GROUP BY lang "
                           "ORDER BY COUNT(*) DESC").fetchall()
    print("Done / total by language:", {lang: f"{d}/{n}" for lang, d, n in rows})
    if UNPARSED.exists():
        print(f"Unparsed callbacks logged in {UNPARSED.relative_to(ROOT)}")


def export(args):
    con = db()
    df = pd.read_sql_query("SELECT guid, seg_no, n_segs, lang, year, status, translated FROM segments", con)
    per = df.groupby("guid").agg(n_segs=("n_segs", "first"), n_done=("status", lambda s: int((s == "done").sum())))
    complete = set(per.index[per["n_done"] == per["n_segs"]])
    d = df[df["guid"].isin(complete)].sort_values(["guid", "seg_no"])
    out = d.groupby("guid").agg(language=("lang", "first"), year=("year", "first"),
                                text_en=("translated", lambda s: " ".join(s.fillna(""))),
                                n_segments=("seg_no", "size")).reset_index()
    out.to_parquet(OUT, index=False)
    print(f"Speeches fully translated: {len(out):,} of {per.shape[0]:,}; saved {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="split the non-English texts into segments to translate")
    p.add_argument("--from-year", type=int, default=1996)
    p.add_argument("--to-year", type=int, default=2023)
    r = sub.add_parser("run", help="send segments to eTranslation and receive the translations")
    r.add_argument("--public-url", required=True, help="public HTTPS address forwarding to localhost:8765")
    r.add_argument("--limit", type=int, default=None, help="send at most this many segments (for tests)")
    r.add_argument("--max-wait", type=int, default=1800, help="seconds to wait for answers after the last submission")
    r.add_argument("--retry-errors", action="store_true", help="send again the segments that failed")
    sub.add_parser("status", help="show progress")
    sub.add_parser("export", help="write one English text per fully translated speech")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "status": status, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
