"""Traduzione in inglese dei testi non inglesi del dataset unified con eTranslation REST v2.

Flusso:
  1. prepare  : legge il dataset, riconosce la lingua di ogni testo (lingua-py, in locale,
                limitato alle 24 lingue UE; il maltese con una regola sui caratteri), e
                spezza i testi non inglesi in segmenti <= 4500 caratteri ai confini di frase
                -> SQLite (data/interim/etranslation/unified.sqlite)
  2. run      : avvia il ricevitore dei callback su 127.0.0.1:8765, invia i segmenti a
                eTranslation v2 (askTranslate, HTTP Basic) e salva le traduzioni che arrivano
  3. status   : conteggi per stato e per lingua, e disaccordi con la colonna lingua del dataset
  4. export   : un testo inglese per ogni testo completamente tradotto -> parquet

eTranslation v2 vuole sempre sourceLanguage esplicito, per questo la lingua si riconosce prima.

Blocchi: con HTTP 429/408/5xx, errori di rete, risposte non valide, callback di quota o
sovraccarico, o tunnel non raggiungibile, lo script non scarta nulla: aspetta (pause che
raddoppiano da 30 s a 15 min, o il Retry-After del server) e riprova lo stesso segmento.
Un callback viene accettato solo se il suo requestId è tra quelli inviati (guida, punto 4).

Credenziali (mai nel codice): variabili d'ambiente ETR_USER / ETR_PASS
(accettate anche ETRANSLATION_APP / ETRANSLATION_PASSWORD).

Esempio (Mac, zsh):
  uv run --with lingua-language-detector --with tqdm \
      python -m europarl_emi.etranslation.translate_unified prepare \
      --input data/processed/unified.parquet --id-col speech_id --text-col text \
      --lang-col original_language
  cloudflared tunnel --url http://127.0.0.1:8765      # in un'altra finestra
  uv run --with truststore --with tqdm python -m europarl_emi.etranslation.translate_unified run \
      --public-url https://xxxx.trycloudflare.com --limit 5
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ENDPOINT = "https://language-tools.ec.europa.eu/etranslation/api/askTranslate"
DEFAULT_DB = Path("data/interim/etranslation/unified.sqlite")
DEFAULT_OUT = Path("data/interim/etranslation/unified_translations.parquet")
LOG_DIR = Path("data/interim/etranslation")

# Le 24 lingue ufficiali UE (codici eTranslation, ISO 639-1 maiuscolo)
EU_LANGS = {
    "BG", "CS", "DA", "DE", "EL", "EN", "ES", "ET", "FI", "FR", "GA", "HR",
    "HU", "IT", "LT", "LV", "MT", "NL", "PL", "PT", "RO", "SK", "SL", "SV",
}
# lingua-py non ha il maltese: lo riconosciamo con una regola (ħ, ġ, ċ esistono solo in maltese)
LINGUA_NAMES = [
    "BULGARIAN", "CZECH", "DANISH", "GERMAN", "GREEK", "ENGLISH", "SPANISH",
    "ESTONIAN", "FINNISH", "FRENCH", "IRISH", "CROATIAN", "HUNGARIAN", "ITALIAN",
    "LITHUANIAN", "LATVIAN", "DUTCH", "POLISH", "PORTUGUESE", "ROMANIAN",
    "SLOVAK", "SLOVENE", "SWEDISH",
]
MT_CHARS = re.compile(r"[ħĦġĠċĊ]")
SENT_SPLIT = re.compile(r"(?<=[.!?;:…])\s+")


# ----------------------------------------------------------------------------- utilità

try:
    from tqdm import tqdm
except ImportError:  # senza tqdm lo script funziona, solo senza barra
    tqdm = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    if tqdm is not None:
        tqdm.write(line)  # scrive sopra la barra senza romperla
        sys.stdout.flush()  # visibile subito anche se l'output va in un file di log
    else:
        print(line, flush=True)


# messaggi di errore che indicano un blocco temporaneo (quota, sovraccarico): si aspetta e si riprova
TRANSIENT = ("quota", "limit", "too many", "concurren", "busy", "overload", "unavailable",
             "timeout", "timed out", "try again", "temporar", "maintenance", "capacity")


def is_transient(msg) -> bool:
    m = str(msg or "").lower()
    return any(k in m for k in TRANSIENT)


def fmt_secs(s: float) -> str:
    s = int(max(s, 0))
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}:{s % 60:02d}"


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db, timeout=60, isolation_level=None)  # autocommit
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


SCHEMA = """
CREATE TABLE IF NOT EXISTS texts (
    text_id     TEXT PRIMARY KEY,
    hint_lang   TEXT,      -- lingua dichiarata nel dataset (se c'è la colonna)
    det_lang    TEXT,      -- lingua riconosciuta
    det_conf    REAL,      -- confidenza della lingua riconosciuta (0-1)
    lang        TEXT,      -- lingua usata per la traduzione
    lang_source TEXT,      -- detected / hint / maltese_rule / short
    n_chars     INTEGER,
    n_segments  INTEGER
);
CREATE TABLE IF NOT EXISTS segments (
    seg_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    text_id     TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    lang        TEXT NOT NULL,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new',   -- new / sent / done / error
    request_id  INTEGER,
    sent_at     REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    translation TEXT,
    error       TEXT,
    done_at     REAL,
    UNIQUE (text_id, idx)
);
CREATE INDEX IF NOT EXISTS seg_status ON segments(status);
CREATE INDEX IF NOT EXISTS seg_req ON segments(request_id);
CREATE TABLE IF NOT EXISTS requests (          -- ogni requestId accettato (anche dei reinvii)
    request_id  INTEGER PRIMARY KEY,
    seg_id      INTEGER NOT NULL,
    sent_at     REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def read_dataset(path: Path):
    import pandas as pd

    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
        return pd.read_json(path, lines=True)
    if ".csv" in name:
        comp = "gzip" if name.endswith((".gz", ".gzip")) else "infer"
        return pd.read_csv(path, compression=comp, low_memory=False)
    raise SystemExit(f"Formato non riconosciuto: {path}")


def norm_lang(x) -> str | None:
    """Normalizza codici tipo 'it', 'IT', 'it-IT', 'ita' -> 'IT' (solo lingue UE)."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE", "MUL"}:
        return None
    s = re.split(r"[-_]", s)[0]
    three = {"ENG": "EN", "ITA": "IT", "FRA": "FR", "DEU": "DE", "SPA": "ES",
             "POR": "PT", "NLD": "NL", "ELL": "EL", "POL": "PL", "MLT": "MT"}
    s = three.get(s, s)
    if s == "GR":
        s = "EL"
    return s if s in EU_LANGS else None


def split_segments(text: str, max_chars: int) -> list[str]:
    """Spezza ai confini di frase; frasi troppo lunghe si tagliano agli spazi."""
    out, cur = [], ""
    for sent in SENT_SPLIT.split(text):
        while len(sent) > max_chars:  # frase enorme: taglio all'ultimo spazio utile
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            piece, sent = sent[:cut].strip(), sent[cut:].strip()
            if cur:
                out.append(cur)
                cur = ""
            out.append(piece)
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur = f"{cur} {sent}"
        else:
            out.append(cur)
            cur = sent
    if cur:
        out.append(cur)
    return [s for s in out if s.strip()]


# ----------------------------------------------------------------------------- prepare

def build_detector(low_accuracy: bool):
    from lingua import Language, LanguageDetectorBuilder

    b = LanguageDetectorBuilder.from_languages(*[getattr(Language, n) for n in LINGUA_NAMES])
    if low_accuracy:
        b = b.with_low_accuracy_mode()
    else:
        b = b.with_preloaded_language_models()
    return b.build()


def detect_batch(detector, texts: list[str], sample_chars: int):
    samples = [t[:sample_chars] for t in texts]
    res = detector.compute_language_confidence_values_in_parallel(samples)
    out = []
    for t, vals in zip(samples, res):
        # maltese: lingua-py non lo conosce, regola sui caratteri esclusivi
        n_mt = len(MT_CHARS.findall(t))
        # ħ/ġ/ċ non esistono in nessun'altra lingua UE; un solo carattere isolato in un testo
        # lungo è probabilmente un nome maltese citato in un'altra lingua
        if n_mt >= 1 and n_mt / max(len(t), 1) > 0.002:
            out.append(("MT", 1.0, "maltese_rule"))
        elif vals:
            top = vals[0]
            out.append((top.language.iso_code_639_1.name, float(top.value), "detected"))
        else:
            out.append((None, 0.0, "detected"))
    return out


def cmd_prepare(a) -> None:
    import pandas as pd

    con = connect(a.db)
    con.executescript(SCHEMA)

    df = read_dataset(a.input)
    log(f"dataset: {len(df):,} righe, colonne: {list(df.columns)}")
    for c in [a.id_col, a.text_col] + ([a.lang_col] if a.lang_col else []):
        if c not in df.columns:
            raise SystemExit(f"Colonna '{c}' assente nel dataset")
    if a.query:
        df = df.query(a.query)
        log(f"dopo --query: {len(df):,} righe")
    if a.date_col and (a.from_year or a.to_year):
        years = pd.to_datetime(df[a.date_col], errors="coerce").dt.year
        mask = pd.Series(True, index=df.index)
        if a.from_year:
            mask &= years >= a.from_year
        if a.to_year:
            mask &= years <= a.to_year
        df = df[mask]
        log(f"dopo filtro anni: {len(df):,} righe")

    if df[a.id_col].duplicated().any():
        raise SystemExit(f"La colonna id '{a.id_col}' ha valori duplicati")

    done_ids = {r[0] for r in con.execute("SELECT text_id FROM texts")}
    df = df[~df[a.id_col].astype(str).isin(done_ids)]
    log(f"da processare (non ancora in {a.db}): {len(df):,}")
    if df.empty:
        return

    detector = build_detector(a.low_accuracy)
    ids = df[a.id_col].astype(str).tolist()
    texts = [html.unescape(str(t)) if pd.notna(t) else "" for t in df[a.text_col]]
    hints = [norm_lang(x) for x in df[a.lang_col]] if a.lang_col else [None] * len(ids)

    n_seg_tot = n_foreign = 0
    bar = tqdm(total=len(ids), unit="testi", desc="lingua", dynamic_ncols=True) if tqdm else None
    for start in range(0, len(ids), a.batch):
        sl = slice(start, start + a.batch)
        b_ids, b_txt, b_hint = ids[sl], texts[sl], hints[sl]
        long_idx = [i for i, t in enumerate(b_txt) if len(t.strip()) >= a.min_chars]
        dets = dict(zip(long_idx, detect_batch(detector, [b_txt[i] for i in long_idx], a.sample_chars)))

        text_rows, seg_rows = [], []
        for i, (tid, txt, hint) in enumerate(zip(b_ids, b_txt, b_hint)):
            det_lang, det_conf, src = dets.get(i, (None, 0.0, "short"))
            lang = det_lang
            if src == "short" or det_lang is None:
                lang, src = (hint, "hint") if hint else (None, "short")
            elif det_conf < a.min_conf and hint:
                lang, src = hint, "hint"
            segs = []
            if lang and lang != "EN":
                segs = split_segments(txt.strip(), a.max_chars)
                n_foreign += 1
            text_rows.append((tid, hint, det_lang, det_conf, lang, src, len(txt), len(segs)))
            seg_rows += [(tid, k, lang, s) for k, s in enumerate(segs)]
        con.execute("BEGIN")
        con.executemany("INSERT OR REPLACE INTO texts VALUES (?,?,?,?,?,?,?,?)", text_rows)
        con.executemany(
            "INSERT OR IGNORE INTO segments (text_id, idx, lang, text) VALUES (?,?,?,?)", seg_rows)
        con.execute("COMMIT")
        n_seg_tot += len(seg_rows)
        if bar:
            bar.update(len(b_ids))
            bar.set_postfix(non_inglesi=n_foreign, segmenti=n_seg_tot)
        else:
            log(f"{min(start + a.batch, len(ids)):,}/{len(ids):,} testi | non inglesi {n_foreign:,}")
    if bar:
        bar.close()
    log(f"fatto: non inglesi {n_foreign:,}, segmenti {n_seg_tot:,}. Controlla con: status")


# ----------------------------------------------------------------------------- callback

def _parse_body(raw: bytes) -> dict:
    s = raw.decode("utf-8", errors="replace")
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_value": obj}
    except json.JSONDecodeError:
        return {k: v[0] for k, v in parse_qs(s).items()}


def _seg_from_payload(p: dict, con: sqlite3.Connection):
    """Accetta il callback solo se il requestId è uno di quelli inviati da noi
    e, se c'è externalReference, punta allo stesso segmento (guida, punto 4)."""
    try:
        rid = int(p.get("requestId"))
    except (TypeError, ValueError):
        return None
    known = con.execute("SELECT seg_id FROM requests WHERE request_id=?", (rid,)).fetchone()
    if known is None:
        return None
    ref = p.get("externalReference")
    if ref is not None and str(ref) != str(known[0]):
        return None
    return con.execute("SELECT seg_id, status FROM segments WHERE seg_id=?", (known[0],)).fetchone()


def _delivery_text(p: dict) -> str | None:
    res = p.get("result")
    if res is None:
        return None
    fmt = str(p.get("outputFormat") or "").lower()
    if fmt and not any(k in fmt for k in ("text", "txt", "plain")):
        try:
            return base64.b64decode(res, validate=True).decode("utf-8")
        except Exception:
            return res
    return res


def make_handler(db: Path, token: str, cooldown: float = 300, max_attempts: int = 3):
    raw_log = LOG_DIR / "callbacks_raw.jsonl"
    bad_log = LOG_DIR / "callbacks_unparsed.log"
    lock = threading.Lock()
    counter = {"raw": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # niente log per ogni richiesta
            pass

        def _reply(self, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):
            self._reply(200 if self.path.startswith(f"/{token}/") else 404)

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            if len(parts) != 2 or parts[0] != token or parts[1] not in {"success", "failure", "delivery"}:
                self._reply(404)
                return
            kind = parts[1]
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._reply(200)  # risposta subito, poi elaborazione
            p = _parse_body(raw)
            with lock:
                if counter["raw"] < 50:  # i primi callback per controllare il formato
                    counter["raw"] += 1
                    with raw_log.open("a") as f:
                        f.write(json.dumps({"kind": kind, "payload": p}, ensure_ascii=False)[:20000] + "\n")
            con = connect(db)
            try:
                row = _seg_from_payload(p, con)
                if row is None:
                    with lock, bad_log.open("a") as f:
                        f.write(f"{time.ctime()} {kind} {raw[:2000]!r}\n")
                    return
                seg_id, status = row
                con.execute("INSERT OR REPLACE INTO meta VALUES ('last_callback', ?)", (str(time.time()),))
                if status == "done":
                    return
                if kind == "failure":
                    err = f"{p.get('errorCode')}: {p.get('errorMessage')}"
                    if any(k in err.upper() for k in ("BACKEND", "DELIVER")) or \
                            str(p.get("errorCode")) in ("-40005", "-70001"):
                        # guasto momentaneo del motore (es. NO_TRANSLATION_RECEIVED_FROM_BACKEND): si ritenta
                        att = con.execute("SELECT attempts FROM segments WHERE seg_id=?", (seg_id,)).fetchone()[0]
                        new_status = "new" if att < max_attempts else "error"
                        con.execute("UPDATE segments SET status=?, error=? WHERE seg_id=? AND status!='done'",
                                    (new_status, err, seg_id))
                    elif is_transient(err):
                        # blocco temporaneo: il segmento torna in coda e gli invii si fermano per un po'
                        con.execute("UPDATE segments SET status='new', error=? WHERE seg_id=? AND status!='done'",
                                    (err, seg_id))
                        until = time.time() + cooldown
                        con.execute("INSERT INTO meta VALUES ('cooldown_until', ?) ON CONFLICT(key) DO UPDATE "
                                    "SET value=MAX(CAST(value AS REAL), excluded.value)", (str(until),))
                        con.execute("INSERT OR REPLACE INTO meta VALUES ('cooldown_reason', ?)", (err[:200],))
                    else:
                        con.execute("UPDATE segments SET status='error', error=? WHERE seg_id=? AND status!='done'",
                                    (err, seg_id))
                    return
                txt = p.get("translatedText") if kind == "success" else _delivery_text(p)
                if txt is None:
                    with lock, bad_log.open("a") as f:
                        f.write(f"{time.ctime()} {kind} senza testo {raw[:2000]!r}\n")
                    return
                con.execute(
                    "UPDATE segments SET status='done', translation=?, done_at=?, error=NULL "
                    "WHERE seg_id=? AND status!='done'", (txt, time.time(), seg_id))
            finally:
                con.close()

    return Handler


# ----------------------------------------------------------------------------- run

def credentials() -> tuple[str, str]:
    user = os.environ.get("ETR_USER") or os.environ.get("ETRANSLATION_APP")
    pwd = os.environ.get("ETR_PASS") or os.environ.get("ETRANSLATION_PASSWORD")
    if not user or not pwd:
        raise SystemExit("Mancano le credenziali: imposta ETR_USER e ETR_PASS")
    return user, pwd


def send_one(session, endpoint, base, seg, domain, institution):
    seg_id, lang, text = seg
    payload = {
        "callerInformation": {"externalReference": str(seg_id)},
        "textToTranslate": text,
        "sourceLanguage": lang,
        "targetLanguages": ["EN"],
        "domain": domain,
        "notifications": {
            "success": {"http": f"{base}/success"},
            "failure": {"http": f"{base}/failure"},
        },
        "deliveries": {"http": f"{base}/delivery"},
    }
    if institution:
        payload["callerInformation"]["institution"] = institution
    r = session.post(endpoint, json=payload, timeout=60)
    return r


def cmd_run(a) -> None:
    import requests

    try:
        import truststore  # certificati di sistema su macOS
        truststore.inject_into_ssl()
    except ImportError:
        pass

    con = connect(a.db)
    con.executescript(SCHEMA)
    row = con.execute("SELECT value FROM meta WHERE key='token'").fetchone()
    token = row[0] if row else secrets.token_urlsafe(16)
    con.execute("INSERT OR REPLACE INTO meta VALUES ('token', ?)", (token,))
    if a.retry_errors:
        n = con.execute("UPDATE segments SET status='new', error=NULL WHERE status='error'").rowcount
        log(f"rimessi in coda {n} segmenti in errore")
    if a.requeue_sent:
        # inviati in una sessione precedente: i loro callback andavano a un tunnel che non c'è più
        n = con.execute("UPDATE segments SET status='new' WHERE status='sent'").rowcount
        if n:
            log(f"rimessi in coda {n} segmenti inviati in una sessione precedente e rimasti senza risposta")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(a.db, token, a.cooldown, a.max_attempts))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"{a.public_url.rstrip('/')}/{token}"
    log(f"ricevitore su {a.host}:{a.port}; callback pubblici sotto {a.public_url.rstrip('/')}/<token>/")

    session = requests.Session()
    session.auth = credentials()
    session.headers["Content-Type"] = "application/json"

    langs = [x.strip().upper() for x in a.langs.split(",")] if a.langs else None
    lang_sql = f" AND lang IN ({','.join('?' * len(langs))})" if langs else ""
    lp = tuple(langs or ())

    def counts() -> dict:
        return dict(con.execute("SELECT status, COUNT(*) FROM segments WHERE 1=1" + lang_sql +
                                " GROUP BY status", lp).fetchall())

    # barra: segmenti tradotti (anche quelli arrivati in sessioni precedenti) sul totale da fare
    c0 = counts()
    done0 = c0.get("done", 0)
    todo = sum(v for k, v in c0.items() if k != "done")
    total = done0 + (min(a.limit, todo) if a.limit is not None else todo)
    bar = (tqdm(total=total, initial=done0, unit="seg", desc="traduzione", dynamic_ncols=True,
                smoothing=0.05, mininterval=1 if sys.stderr.isatty() else 60) if tqdm else None)
    state = {"sent": 0, "status": "invio", "last_print": 0.0}

    def refresh(status: str | None = None) -> dict:
        if status:
            state["status"] = status
        c = counts()
        if bar is not None:
            bar.total = max(bar.total, c.get("done", 0))
            bar.n = c.get("done", 0)
            bar.set_postfix_str(f"in attesa {c.get('sent', 0)} | errori {c.get('error', 0)} | "
                                f"{state['status']}", refresh=False)
            bar.refresh()
        elif time.time() - state["last_print"] > 30:
            state["last_print"] = time.time()
            log(f"{c} | {state['status']}")
        return c

    def wait(seconds: float, reason: str) -> None:
        """Pausa con conto alla rovescia; intanto i callback continuano ad arrivare."""
        end = time.time() + seconds
        while (left := end - time.time()) > 0:
            refresh(f"pausa ({reason}), riprendo tra {fmt_secs(left)}")
            time.sleep(min(1.0, left))
        refresh("invio")

    def tunnel_ok() -> bool:
        # richiesta senza credenziali: va al nostro ricevitore, non a eTranslation
        try:
            return requests.get(f"{base}/ping", timeout=20).status_code == 200
        except requests.RequestException:
            return False

    class Backoff:
        def __init__(self, start: float, cap: float):
            self.start, self.cap, self.cur = start, cap, start

        def next(self) -> float:
            v, self.cur = self.cur, min(self.cur * 2, self.cap)
            return v

        def reset(self) -> None:
            self.cur = self.start

    backoff = Backoff(a.backoff_start, a.backoff_max)
    run_start = time.time()
    consecutive_errors = 0
    next_tunnel_check = 0.0
    tunnel_down_since = None
    n401 = 0
    exit_code = 0
    try:
        while True:
            now = time.time()
            # segmenti senza risposta da troppo tempo: di nuovo in coda (o errore dopo N tentativi)
            con.execute("UPDATE segments SET status='error', error='nessuna risposta' "
                        "WHERE status='sent' AND sent_at<? AND attempts>=?",
                        (now - a.resend_after, a.max_attempts))
            con.execute("UPDATE segments SET status='new' WHERE status='sent' AND sent_at<?",
                        (now - a.resend_after,))

            # 1) blocco segnalato da eTranslation in un callback di errore (quota, sovraccarico)
            row = con.execute("SELECT value FROM meta WHERE key='cooldown_until'").fetchone()
            if row and float(row[0]) > now:
                why = con.execute("SELECT value FROM meta WHERE key='cooldown_reason'").fetchone()
                log(f"eTranslation segnala un blocco temporaneo ({why[0] if why else '?'}): aspetto")
                wait(float(row[0]) - now, "blocco eTranslation")
                continue

            # 2) tunnel: se non è raggiungibile le traduzioni non possono arrivare, inutile inviare
            if not a.no_tunnel_check and now >= next_tunnel_check:
                if tunnel_ok():
                    next_tunnel_check = now + a.tunnel_check
                    tunnel_down_since = None
                else:
                    tunnel_down_since = tunnel_down_since or now
                    if now - tunnel_down_since > a.max_tunnel_wait:
                        log(f"tunnel irraggiungibile da {fmt_secs(now - tunnel_down_since)}: esco "
                            "(il lanciatore riapre il tunnel e riparte)")
                        exit_code = 3
                        break
                    s = backoff.next()
                    log(f"tunnel non raggiungibile ({a.public_url}); riprovo tra {fmt_secs(s)}")
                    wait(s, "tunnel non raggiungibile")
                    continue

            if a.stall_timeout:
                oldest = con.execute("SELECT MIN(sent_at) FROM segments WHERE status='sent'").fetchone()[0]
                lc = con.execute("SELECT value FROM meta WHERE key='last_callback'").fetchone()
                last_cb = max(float(lc[0]) if lc else 0.0, run_start)
                if oldest and now - oldest > a.stall_timeout and now - last_cb > a.stall_timeout:
                    log(f"nessuna risposta da eTranslation da {fmt_secs(now - last_cb)} con richieste in attesa: "
                        "probabilmente non raggiunge il tunnel. Esco (il lanciatore riapre il tunnel e riparte)")
                    exit_code = 3
                    break

            refresh()
            pending = con.execute("SELECT COUNT(*) FROM segments WHERE status='sent'").fetchone()[0]
            limit_hit = a.limit is not None and state["sent"] >= a.limit
            seg = None
            if not limit_hit and pending < a.max_inflight:
                seg = con.execute("SELECT seg_id, lang, text FROM segments WHERE status='new'" + lang_sql +
                                  " ORDER BY seg_id LIMIT 1", lp).fetchone()
            if seg is None:
                if pending == 0 and (limit_hit or not con.execute(
                        "SELECT 1 FROM segments WHERE status='new'" + lang_sql + " LIMIT 1", lp).fetchone()):
                    break
                time.sleep(1)
                continue

            # 3) invio; ogni problema temporaneo -> pausa crescente e nuovo tentativo dello stesso segmento
            try:
                r = send_one(session, a.endpoint, base, seg, a.domain, a.institution)
            except requests.RequestException as e:
                s = backoff.next()
                log(f"errore di rete ({type(e).__name__}); riprovo tra {fmt_secs(s)}")
                wait(s, "rete")
                continue
            if r.status_code == 401:
                n401 += 1
                if n401 >= a.max_401:
                    log(f"HTTP 401 per {n401} volte di fila: credenziali rifiutate. "
                        "Controlla application name e password (o se la password API è scaduta)")
                    exit_code = 2
                    break
                s = backoff.next()
                log(f"HTTP 401 (credenziali rifiutate), tentativo {n401}/{a.max_401}; riprovo tra {fmt_secs(s)}")
                wait(s, "HTTP 401")
                continue
            n401 = 0
            if r.status_code in (408, 425, 429) or r.status_code >= 500:
                ra = r.headers.get("Retry-After", "")
                s = float(ra) if ra.isdigit() else backoff.next()
                log(f"HTTP {r.status_code} (servizio occupato o limite di richieste); riprovo tra {fmt_secs(s)}")
                wait(s, f"HTTP {r.status_code}")
                continue
            try:
                body = r.json()
                rid = int(body["requestId"] if isinstance(body, dict) else body)
            except (ValueError, KeyError, TypeError):
                rid = None
            if r.status_code == 200 and rid is not None and rid > 0:
                # requestId registrato prima che possa arrivare il callback
                con.execute("INSERT OR REPLACE INTO requests VALUES (?,?,?)", (rid, seg[0], time.time()))
                con.execute("UPDATE segments SET status='sent', request_id=?, sent_at=?, "
                            "attempts=attempts+1 WHERE seg_id=? AND status='new'",
                            (rid, time.time(), seg[0]))
                state["sent"] += 1
                backoff.reset()
                consecutive_errors = 0
                time.sleep(a.pause)
                continue

            # 4) richiesta rifiutata (requestId negativo, risposta non valida, altro 4xx)
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if rid is None and r.status_code == 200 or is_transient(r.text):
                s = backoff.next()  # risposta strana o messaggio di blocco: non conta come tentativo
                log(f"risposta non utilizzabile ({err[:120]}); riprovo tra {fmt_secs(s)}")
                wait(s, "risposta non valida")
                continue
            con.execute("UPDATE segments SET attempts=attempts+1, error=? WHERE seg_id=?", (err, seg[0]))
            att = con.execute("SELECT attempts FROM segments WHERE seg_id=?", (seg[0],)).fetchone()[0]
            if att < a.max_attempts:
                s = backoff.next()
                log(f"segmento {seg[0]} rifiutato ({err[:120]}); tentativo {att}/{a.max_attempts}, "
                    f"riprovo tra {fmt_secs(s)}")
                wait(s, "rifiuto")
                continue
            con.execute("UPDATE segments SET status='error' WHERE seg_id=?", (seg[0],))
            consecutive_errors += 1
            log(f"segmento {seg[0]} in errore dopo {att} tentativi: {err[:200]}")
            if consecutive_errors >= a.max_rejected:
                log(f"{consecutive_errors} segmenti di fila rifiutati: probabile problema nei parametri, mi fermo. "
                    "Controlla con 'status', poi rilancia con --retry-errors")
                break
    except KeyboardInterrupt:
        log("interrotto: i segmenti 'sent' riceveranno ancora risposta solo se il ricevitore resta attivo")
        exit_code = 130
    finally:
        if bar is not None:
            refresh("fine")
            bar.close()
        server.shutdown()
    c = dict(con.execute("SELECT status, COUNT(*) FROM segments GROUP BY status").fetchall())
    log(f"fine. inviati in questa sessione: {state['sent']} | {c}")
    if exit_code == 0 and c.get("new", 0) + c.get("sent", 0) > 0 and a.limit is None:
        exit_code = 4  # uscito prima della fine (es. troppi rifiuti di fila)
    return exit_code


# ----------------------------------------------------------------------------- status / export

def cmd_status(a) -> None:
    con = connect(a.db)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    print("\nTesti per lingua usata e origine della lingua:")
    for r in q("SELECT COALESCE(lang,'?'), lang_source, COUNT(*) FROM texts GROUP BY 1,2 ORDER BY 3 DESC"):
        print(f"  {r[0]:>3} {r[1]:<13} {r[2]:>9,}")
    print("\nSegmenti per stato:")
    for r in q("SELECT status, COUNT(*), SUM(LENGTH(text)) FROM segments GROUP BY 1"):
        print(f"  {r[0]:<6} {r[1]:>9,} segmenti {r[2] or 0:>14,} caratteri")
    print("\nSegmenti per lingua (fatti / totale):")
    for r in q("SELECT lang, SUM(status='done'), COUNT(*) FROM segments GROUP BY 1 ORDER BY 3 DESC"):
        print(f"  {r[0]} {r[1]:>8,} / {r[2]:>8,}")
    dis = q("SELECT hint_lang, det_lang, COUNT(*) FROM texts WHERE hint_lang IS NOT NULL "
            "AND det_lang IS NOT NULL AND hint_lang!=det_lang GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15")
    tot = q("SELECT COUNT(*) FROM texts WHERE hint_lang IS NOT NULL AND det_lang IS NOT NULL")[0][0]
    if tot:
        n = sum(r[2] for r in q("SELECT hint_lang, det_lang, COUNT(*) FROM texts WHERE hint_lang IS NOT NULL "
                                "AND det_lang IS NOT NULL AND hint_lang!=det_lang GROUP BY 1,2"))
        print(f"\nDisaccordo lingua dataset vs riconosciuta: {n:,} su {tot:,} ({n / tot:.2%})")
        for r in dis:
            print(f"  dataset {r[0]} -> riconosciuta {r[1]}: {r[2]:,}")
    first, last, n_hour = q("SELECT MIN(done_at), MAX(done_at), SUM(done_at > strftime('%s','now') - 3600) "
                            "FROM segments WHERE status='done'")[0]
    if last:
        tot_done = q("SELECT COUNT(*) FROM segments WHERE status='done'")[0][0]
        todo = q("SELECT COUNT(*) FROM segments WHERE status!='done'")[0][0]
        print(f"\nPrima traduzione arrivata : {time.strftime('%d/%m %H:%M', time.localtime(first))}")
        print(f"Ultima traduzione arrivata: {time.strftime('%d/%m %H:%M', time.localtime(last))} "
              f"({fmt_secs(time.time() - last)} fa)")
        print(f"Arrivate nell'ultima ora  : {n_hour or 0:,}")
        if n_hour:
            print(f"A questo ritmo mancano circa {todo / n_hour:,.0f} ore")
        elif tot_done and last > first:
            rate = tot_done / (last - first) * 3600
            print(f"Ritmo medio finora: {rate:,.0f} segmenti/ora -> mancano circa {todo / rate:,.0f} ore di lavoro")
    errs = q("SELECT error, COUNT(*) FROM segments WHERE status='error' GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
    if errs:
        print("\nErrori più frequenti:")
        for r in errs:
            print(f"  {r[1]:>6,}  {r[0][:150]}")


def cmd_export(a) -> None:
    import pandas as pd

    con = connect(a.db)
    texts = pd.read_sql("SELECT * FROM texts", con)
    segs = pd.read_sql("SELECT text_id, idx, status, translation FROM segments ORDER BY text_id, idx", con)
    complete = segs.groupby("text_id")["status"].agg(lambda s: (s == "done").all())
    ok = segs[segs["text_id"].isin(complete[complete].index)]
    joined = ok.groupby("text_id")["translation"].agg(" ".join).rename("text_en")
    out = texts.merge(joined, left_on="text_id", right_index=True, how="left")
    out["translated"] = out["text_en"].notna()
    out = out.drop(columns=["n_segments"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    n_foreign = (out["lang"].notna() & (out["lang"] != "EN")).sum()
    log(f"scritto {a.out}: {len(out):,} testi, non inglesi {n_foreign:,}, tradotti {out['translated'].sum():,}")


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="riconosce la lingua e prepara i segmenti")
    pp.add_argument("--input", type=Path, required=True, help="dataset unified (.parquet/.csv[.gz]/.jsonl)")
    pp.add_argument("--id-col", required=True)
    pp.add_argument("--text-col", required=True)
    pp.add_argument("--lang-col", help="colonna con la lingua dichiarata (facoltativa, solo per confronto/ripiego)")
    pp.add_argument("--date-col")
    pp.add_argument("--from-year", type=int)
    pp.add_argument("--to-year", type=int)
    pp.add_argument("--query", help="filtro pandas, es. \"activity_type == 'PLENARY_DEBATE_SPEECH'\"")
    pp.add_argument("--max-chars", type=int, default=4500, help="caratteri per segmento (limite API 5000)")
    pp.add_argument("--min-chars", type=int, default=20, help="sotto questa lunghezza non si riconosce la lingua")
    pp.add_argument("--min-conf", type=float, default=0.5,
                    help="sotto questa confidenza si usa la lingua del dataset, se c'è")
    pp.add_argument("--sample-chars", type=int, default=2000, help="caratteri usati per riconoscere la lingua")
    pp.add_argument("--batch", type=int, default=5000)
    pp.add_argument("--low-accuracy", action="store_true", help="modalità veloce di lingua-py")
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="invia a eTranslation e riceve le traduzioni")
    pr.add_argument("--public-url", required=True, help="indirizzo pubblico del tunnel verso il ricevitore")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=8765)
    pr.add_argument("--endpoint", default=ENDPOINT)
    pr.add_argument("--domain", default="GEN")
    pr.add_argument("--institution", help="facoltativo, va in callerInformation")
    pr.add_argument("--limit", type=int, help="numero massimo di segmenti da inviare in questa sessione")
    pr.add_argument("--langs", help="solo queste lingue, es. IT,FR")
    pr.add_argument("--max-inflight", type=int, default=20)
    pr.add_argument("--pause", type=float, default=0.5)
    pr.add_argument("--resend-after", type=int, default=3600, help="secondi senza risposta prima del reinvio")
    pr.add_argument("--max-attempts", type=int, default=3, help="tentativi per segmento prima dell'errore")
    pr.add_argument("--backoff-start", type=float, default=30, help="prima pausa (s) quando il servizio blocca")
    pr.add_argument("--backoff-max", type=float, default=900, help="pausa massima (s); le pause raddoppiano")
    pr.add_argument("--tunnel-check", type=int, default=300, help="ogni quanti secondi verificare il tunnel")
    pr.add_argument("--no-tunnel-check", action="store_true")
    pr.add_argument("--cooldown", type=float, default=300,
                    help="pausa (s) quando un callback segnala quota o sovraccarico")
    pr.add_argument("--stall-timeout", type=int, default=900,
                    help="secondi senza alcun callback, con richieste in attesa, prima di uscire (codice 3)")
    pr.add_argument("--max-401", type=int, default=4, help="401 di fila prima di fermarsi")
    pr.add_argument("--max-tunnel-wait", type=int, default=900,
                    help="secondi di tunnel irraggiungibile prima di uscire (codice 3)")
    pr.add_argument("--requeue-sent", action="store_true",
                    help="all'avvio rimette in coda i segmenti inviati e mai tornati")
    pr.add_argument("--max-rejected", type=int, default=10,
                    help="segmenti di fila in errore definitivo prima di fermarsi")
    pr.add_argument("--retry-errors", action="store_true", help="rimette in coda i segmenti in errore")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status")
    ps.set_defaults(func=cmd_status)

    pe = sub.add_parser("export")
    pe.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pe.set_defaults(func=cmd_export)

    a = p.parse_args(argv)
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
