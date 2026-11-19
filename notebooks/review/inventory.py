"""REVIEW ONLY. Inventario dei dati: cosa c'e su questa macchina, cosa manca, cosa resta da scaricare o tradurre.

Non modifica nulla: legge i file e scrive solo il rapporto in data/review/inventory/ (fuori da git).
Gira uguale sul Mac e sul server; confrontando i due rapporti si vede cosa va copiato tra le macchine.

    uv run python notebooks/review/inventory.py            # completo (legge anche il corpus grezzo, ~1-2 min)
    uv run python notebooks/review/inventory.py --fast     # salta corpus, testi API e copertura delle traduzioni

Sezioni: 0 macchina e git | 1 file della pipeline | 2 varianti di sensibilita | 3 download API
         4 confronto corpus-API e dataset unificato | 5 eTranslation | 6 riepilogo delle cose da fare
"""

import argparse
import json
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from europarl_emi import config as cfg  # noqa: E402

DATA = cfg.DATA
LINES, TODO = [], []
EU = {"bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl",
      "pl", "pt", "ro", "sk", "sl", "sv"}
API_START = date(2021, 7, 1)


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def todo(s):
    TODO.append(s)


def title(s):
    out()
    out("=" * 78)
    out(s)
    out("=" * 78)


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024


def mtime(p):
    return datetime.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def parquet_rows(p):
    import pyarrow.parquet as pq
    p = Path(p)
    if p.is_dir():
        parts = sorted(p.glob("*.parquet"))
        return sum(pq.ParquetFile(x).metadata.num_rows for x in parts), len(parts)
    return pq.ParquetFile(p).metadata.num_rows, None


def show_file(label, p, expect_rows=None, needed_for=None):
    """One line per expected file: present or missing, size, date, rows for parquet."""
    p = Path(p)
    if not p.exists() or (p.is_dir() and not any(p.iterdir())):
        state = "VUOTA" if p.is_dir() and p.exists() else "MANCA"
        out(f"  [{state:5}] {label:36} {rel(p)}")
        if needed_for:
            todo(f"{rel(p)}: {state.lower()} ({needed_for})")
        return False
    info = ""
    try:
        if p.suffix == ".parquet" or p.is_dir() and any(p.glob("*.parquet")):
            n, parts = parquet_rows(p)
            info = f"{n:>10,} righe" + (f" in {parts} parti" if parts is not None else "")
            if expect_rows and n != expect_rows:
                info += f"  (atteso {expect_rows:,}!)"
                todo(f"{rel(p)}: {n:,} righe invece di {expect_rows:,}")
    except Exception as e:  # file corrotto o scritto a meta
        info = f"ILLEGGIBILE ({type(e).__name__})"
        todo(f"{rel(p)}: illeggibile")
    b = sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) if p.is_dir() else p.stat().st_size
    out(f"  [ ok  ] {label:36} {size(b):>10}  {mtime(p)}  {info}")
    return True


def ranges(days):
    """[d1, d2, d3 ...] -> '2024-01-01..2024-01-05, 2024-02-10' (consecutive weekdays collapsed)."""
    days = sorted(days)
    if not days:
        return ""
    groups, start, prev = [], days[0], days[0]
    for d in days[1:]:
        gap = (d - prev).days
        if gap == 1 or (gap <= 3 and prev.weekday() == 4):
            prev = d
            continue
        groups.append((start, prev))
        start = prev = d
    groups.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}..{b}" for a, b in groups)


# ---------------------------------------------------------------- 0. machine and git

def section_machine():
    title("0. MACCHINA E GIT")
    out(f"  macchina: {socket.gethostname()}   cartella: {ROOT}")
    out(f"  data: {datetime.now():%Y-%m-%d %H:%M}   python: {sys.version.split()[0]}")

    def git(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()

    out(f"  ultimo commit: {git('log', '-1', '--format=%h %ad %s', '--date=format:%Y-%m-%d %H:%M')}")
    out(f"  ramo: {git('status', '-sb').splitlines()[0] if git('status', '-sb') else '?'}  (ahead/behind rispetto all'ultimo fetch)")
    changed = [l for l in git("status", "--porcelain").splitlines() if l]
    code = [l for l in changed if not l[3:].startswith(("data/", "results/", "paper/", "models/"))]
    if code:
        out(f"  file di codice non salvati in git ({len(code)}):")
        for l in code:
            out(f"      {l}")
        todo(f"git: {len(code)} file di codice modificati o non tracciati su questa macchina (commit con il controllo 'solo codice')")
    else:
        out("  nessun file di codice in sospeso")


# ---------------------------------------------------------------- 1. pipeline files

def section_pipeline():
    title("1. FILE DELLA PIPELINE")
    I, P = cfg.INTERIM, cfg.PROCESSED
    out(" grezzi")
    show_file("corpus grezzo", cfg.RAW_CORPUS / "all_years_combined.csv.gzip", needed_for="tutto parte da qui")
    lex = sorted(cfg.LEXICONS.glob("*.csv"))
    out(f"  [{'ok' if lex else 'MANCA':^5}] {'lessici':34} {', '.join(x.name for x in lex)}")
    meps = sorted(cfg.RAW_MEPS.glob("meps_term*.csv"))
    have = {x.stem.replace("meps_term", "") for x in meps}
    miss = [str(t) for t in range(4, 11) if str(t) not in have]
    out(f"  [{'ok' if not miss else 'MANCA':^5}] {'membri per legislatura':34} {len(meps)} file"
        + (f", mancano le legislature {', '.join(miss)}" if miss else ""))
    if miss:
        todo(f"data/raw/meps: mancano meps_term{', '.join(miss)}.csv")
    show_file("database MEP 1994-oggi", P / "mep_database_1994_present.csv", needed_for="passo 7 e script API")
    out(" passi 1-8")
    show_file("01 inglese", I / "01_speeches_en.parquet", 340_771)
    show_file("02 puliti", I / "02_speeches_clean.parquet", 340_606)
    show_file("03 sostanziali", I / "03_speeches_substantive.parquet", 330_075, "input di passi 4-5 e varianti")
    show_file("04 punteggi dei pezzi", I / "04_chunk_scores")
    show_file("05 EMI per pezzo", P / "05_chunk_emi.parquet")
    show_file("05 EMI per discorso", P / "05_speech_emi.parquet", 329_102, "baseline di tutti i confronti")
    show_file("06 annotati", P / "06_speech_annotated.parquet", 329_102)
    show_file("07 campione principale", P / "07_speech_main_sample.parquet", 48_847, "modelli e figure")
    show_file("modello passo 3", cfg.MODELS / "procedural_nblr.joblib")
    out(" legacy (Colab)")
    show_file("outputv2", I / "outputv2.csv.gzip")
    show_file("outputv2_filled", P / "outputv2_filled.csv.gzip")
    show_file("political_groups_simplified", P / "political_groups_simplified.csv", needed_for="legacy_global.py")
    out(" risultati")
    figs = sorted((ROOT / "results" / "figures" / "paper").glob("*.pdf"))
    out(f"  [{'ok' if figs else 'MANCA':^5}] {'figure del paper (pdf)':34} {len(figs)}: "
        + " ".join(x.stem for x in figs))
    for name in ("trend_models.csv", "year_effects_speaker_fe.csv", "S3_examples.csv"):
        show_file(f"tabella {name}", ROOT / "results" / "tables" / name)
    show_file("paper_numbers.txt", ROOT / "results" / "paper_numbers.txt")


# ---------------------------------------------------------------- 2. sensitivity variants

def section_sensitivity():
    title("2. VARIANTI DI SENSIBILITA (data/review/sensitivity/)")
    R = DATA / "review" / "sensitivity"
    names = ["v1_mpnet", "v2_word2vec", "v2b_word2vec_p80", "v3_mpnet_only", "v4_lexicon_count", "v5_minilm_noheader"]
    extra = sorted(p.name for p in R.glob("*") if p.is_dir() and p.name not in names + ["logs"]) if R.exists() else []
    for name in names + extra:
        d = R / name
        if not d.exists():
            out(f"  [MANCA] {name}")
            if name in ("v1_mpnet", "v2_word2vec"):
                todo(f"sensibilita {name}: risultati non qui (cercarli sull'altra macchina o rilanciare)")
            elif name in ("v4_lexicon_count", "v5_minilm_noheader", "v3_mpnet_only"):
                todo(f"sensibilita {name}: risultati non qui (calcolati sul server?)")
            continue
        parts = len(list((d / "chunk_scores").glob("part-*.parquet")))
        s = d / "speech_emi.parquet"
        settings = {}
        if (d / "settings.json").exists():
            try:
                settings = json.loads((d / "settings.json").read_text())
            except ValueError:
                pass
        sample = settings.get("fraction")
        if s.exists():
            n, _ = parquet_rows(s)
            state = f"finita: {n:,} discorsi, {mtime(s)}"
            if sample not in (None, 1, 1.0):
                state += f"  (CAMPIONE {sample})"
                if not name.startswith("_smoke"):
                        todo(f"sensibilita {name}: calcolata solo su un campione ({sample})")
        else:
            state = f"NON FINITA: {parts} parti di punteggi, manca speech_emi.parquet"
            todo(f"sensibilita {name}: non finita ({parts} parti): rilanciare lo stesso comando")
        out(f"  [{'ok' if s.exists() else 'parz':^5}] {name:22} {state}")
    logs = sorted((R / "logs").glob("*.log")) if (R / "logs").exists() else []
    if logs:
        out("  log: " + ", ".join(f"{x.stem} ({mtime(x)})" for x in logs))
    for f in ("sensitivity.csv", "scale_compare.csv", "compare_panels.pdf"):
        p = ROOT / "notebooks" / "review" / "sensitivity" / f
        out(f"  [{'ok' if p.exists() else 'MANCA':^5}] {f:22} " + (mtime(p) if p.exists() else ""))


# ---------------------------------------------------------------- 3. API download

def section_api(fast):
    title("3. DOWNLOAD DALL'API DEL PARLAMENTO (data/raw/api/speeches/)")
    d = DATA / "raw" / "api" / "speeches"
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    tmps = sorted(d.glob("*.tmp")) if d.exists() else []
    today = date.today()
    if not files:
        out("  nessun file: il download non e su questa macchina (sul Mac?)")
        todo("API: nessun file su questa macchina")
        return None
    got = {}
    for f in files:
        try:
            got[date.fromisoformat(f.stem)] = f
        except ValueError:
            pass
    weekdays = [API_START + timedelta(days=i) for i in range((today - API_START).days + 1)]
    weekdays = [x for x in weekdays if x.weekday() < 5]
    first, lastd = min(got), max(got)
    missing_inside = [x for x in weekdays if first <= x <= lastd and x not in got]
    missing_after = [x for x in weekdays if x > lastd]
    out(f"  file: {len(got):,} giorni, dal {first} al {lastd} (ultimo scritto {mtime(got[lastd])})")
    out(f"  giorni feriali dal {API_START} a oggi: {len(weekdays):,}")
    if missing_inside:
        out(f"  BUCHI dentro il periodo ({len(missing_inside)}): {ranges(missing_inside)}")
        todo(f"API: {len(missing_inside)} giorni mancanti dentro il periodo scaricato (rilanciare download_speeches.py)")
    if missing_after:
        out(f"  DA SCARICARE dopo l'ultimo file: {len(missing_after)} giorni feriali, {missing_after[0]}..{missing_after[-1]}")
        todo(f"API: da scaricare {missing_after[0]}..{missing_after[-1]} (download_speeches.py --start {missing_after[0]})")
    if tmps:
        out(f"  file .tmp rimasti (download interrotto): {', '.join(x.name for x in tmps)}")
        todo("API: file .tmp rimasti, cancellarli e rilanciare")

    sizes = {k: f.stat().st_size for k, f in got.items()}
    nonempty = {k for k, s in sizes.items() if s > 0}
    weeks = defaultdict(set)
    for k in nonempty:
        weeks[k.isocalendar()[:2]].add(k)
    suspicious = sorted(k for k, s in sizes.items() if s == 0 and k.weekday() < 4 and weeks.get(k.isocalendar()[:2])
                        and min(weeks[k.isocalendar()[:2]]) < k < max(weeks[k.isocalendar()[:2]]))
    early = sorted(k for k, f in got.items() if sizes[k] == 0 and k >= today - timedelta(days=60)
                   and datetime.fromtimestamp(f.stat().st_mtime).date() - k < timedelta(days=14))
    out(f"  giorni con interventi: {len(nonempty):,}   file vuoti (nessuna seduta): {len(got) - len(nonempty):,}")
    if suspicious:
        out(f"  file vuoti SOSPETTI (in mezzo a una settimana di seduta): {ranges(suspicious)}")
        todo(f"API: {len(suspicious)} giorni vuoti sospetti: cancellare i file e riscaricarli")
    if early:
        out(f"  file vuoti scaricati a meno di 2 settimane dalla seduta (l'API pubblica in ritardo?): {ranges(early)}")
        todo(f"API: {len(early)} giorni recenti scaricati vuoti forse troppo presto: cancellarli e riscaricarli")
    if fast:
        return None

    by_year = defaultdict(Counter)
    bad = 0
    for k in sorted(nonempty):
        with open(got[k], encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                c = by_year[k.year]
                c["interventi"] += 1
                orig = r.get("original_language") or ""
                oral = r.get("activity_type") == "PLENARY_DEBATE_SPEECH"
                speaker = r.get("role") == "SPEAKER"
                if orig == "en":
                    c["in inglese"] += 1
                    if oral and speaker:
                        c["orali SPEAKER en"] += 1
                else:
                    c["non inglesi"] += 1
                    if r.get("text_en_mt"):
                        c["con MT inglese API"] += 1
                    elif r.get("text"):
                        c["senza MT inglese"] += 1
                        if oral and speaker:
                            c["orali SPEAKER senza MT"] += 1
                if not r.get("text"):
                    c["senza testo"] += 1
    cols = ["interventi", "in inglese", "orali SPEAKER en", "non inglesi", "con MT inglese API",
            "senza MT inglese", "orali SPEAKER senza MT", "senza testo"]
    short = ["tutti", "en", "en_orali", "non_en", "MT_API", "senza_MT", "orali_s_MT", "no_testo"]
    out()
    out("  colonne: en = lingua originale inglese; en_orali = PLENARY_DEBATE_SPEECH, ruolo SPEAKER;")
    out("           MT_API / senza_MT = non inglesi con / senza traduzione inglese dell'API; orali_s_MT = orali senza MT")
    out("  " + f"{'anno':6}" + "".join(f"{c:>11}" for c in short))
    for y in sorted(by_year):
        out("  " + f"{y:<6}" + "".join(f"{by_year[y][c]:>11,}" for c in cols))
    tot = sum((by_year[y] for y in by_year), Counter())
    out("  " + f"{'tot':6}" + "".join(f"{tot[c]:>11,}" for c in cols))
    if bad:
        out(f"  righe illeggibili: {bad}")
        todo(f"API: {bad} righe JSON illeggibili")
    if tot["senza MT inglese"]:
        todo(f"API: {tot['senza MT inglese']:,} interventi non inglesi senza traduzione inglese dell'API "
             f"({tot['orali SPEAKER senza MT']:,} orali): vedi la sezione 5 per quanti sono gia tradotti con eTranslation")
    return tot


# ---------------------------------------------------------------- 4. overlap and unified dataset

def section_unified():
    title("4. CONFRONTO CORPUS-API E DATASET UNIFICATO")
    show_file("coppie corpus-API", DATA / "review" / "overlap" / "pairs.parquet", needed_for="build_unified.py")
    show_file("tabella di raccordo", DATA / "review" / "overlap" / "crosswalk_corpus_api.csv")
    uni = DATA / "raw" / "unified" / "unified.parquet"
    ok = show_file("dataset unificato", uni)
    api = sorted((DATA / "raw" / "api" / "speeches").glob("*.jsonl"))
    if ok and api:
        newer = [f for f in api if f.stat().st_mtime > uni.stat().st_mtime and f.stat().st_size > 0]
        if newer:
            out(f"  ATTENZIONE: {len(newer)} giorni API con interventi scaricati dopo il dataset unificato "
                f"({newer[0].stem}..{newer[-1].stem})")
            todo(f"dataset unificato: ricostruirlo (overlap_corpus_api.py, build_unified.py) per includere "
                 f"{len(newer)} giorni API nuovi")
    ms = DATA / "raw" / "unified" / "main_series_by_year.csv"
    if ok and ms.exists():
        out("  serie principale per anno e fonte (main_series_by_year.csv):")
        for l in ms.read_text().splitlines():
            out("      " + l)
    elif not ok:
        todo("dataset unificato non costruito su questa macchina (overlap_corpus_api.py poi build_unified.py)")


# ---------------------------------------------------------------- 5. eTranslation

def corpus_non_english():
    """Same selection as translate.py prepare: EU non-English languages, by year."""
    import pandas as pd
    p = cfg.RAW_CORPUS / "all_years_combined.csv.gzip"
    if not p.exists():
        return None, None
    by, other = Counter(), Counter()
    for ch in pd.read_csv(p, compression="gzip", usecols=["guid", "language", "pubdate", "description"],
                          dtype=str, chunksize=50_000):
        lang = ch["language"].str.strip().str.lower()
        year = pd.to_datetime(ch["pubdate"], errors="coerce", utc=True, format="mixed").dt.year
        has = ch["description"].notna()
        for (l, y), n in pd.DataFrame({"l": lang[has], "y": year[has]}).value_counts().items():
            if l in EU:
                by[int(y)] += n
            elif l != "en":
                other[l] += n
    return by, other


def section_etranslation(fast):
    title("5. ETRANSLATION (data/interim/etranslation/)")
    D = DATA / "interim" / "etranslation"
    old = D / "segments.sqlite"
    if old.exists():
        out(f"  segments.sqlite (vecchio translate.py, API v1, superato): {size(old.stat().st_size)}, {mtime(old)}")
    db = D / "unified.sqlite"
    if not db.exists():
        out("  unified.sqlite: MANCA (translate_unified.py prepare non fatto qui)")
        todo("eTranslation: unified.sqlite manca su questa macchina")
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    out(f"  unified.sqlite: {size(db.stat().st_size)}, {mtime(db)}")
    n_t, n_tr, chars = q("SELECT COUNT(*), SUM(n_segments > 0), SUM(CASE WHEN n_segments > 0 THEN n_chars END) FROM texts")[0]
    out(f"  testi registrati: {n_t:,}   da tradurre (con segmenti): {n_tr or 0:,}   caratteri: {chars or 0:,}")
    out("  origine della lingua: " + ", ".join(f"{a} {n:,}" for a, n in q(
        "SELECT lang_source, COUNT(*) FROM texts GROUP BY 1 ORDER BY 2 DESC")))
    out("  testi tradotti per lingua: " + ", ".join(f"{a} {n:,}" for a, n in q(
        "SELECT lang, COUNT(*) FROM texts WHERE n_segments > 0 GROUP BY 1 ORDER BY 2 DESC")))
    st = dict(q("SELECT status, COUNT(*) FROM segments GROUP BY status"))
    out(f"  segmenti per stato: {st}")
    if st.get("error"):
        errs = q("SELECT error, COUNT(*) FROM segments WHERE status='error' GROUP BY error ORDER BY 2 DESC LIMIT 5")
        out("  errori principali: " + "; ".join(f"{e} ({n})" for e, n in errs))
        todo(f"eTranslation: {st['error']:,} segmenti in errore")
    left = st.get("new", 0) + st.get("sent", 0)
    if left:
        todo(f"eTranslation: {left:,} segmenti ancora da tradurre in unified.sqlite")
    show_file("traduzioni esportate", D / "unified_translations.parquet")
    log = D / "callbacks_unparsed.log"
    if log.exists():
        n = sum(1 for _ in open(log, encoding="utf-8", errors="replace"))
        out(f"  callbacks_unparsed.log: {n} righe (ultima {mtime(log)})")

    uni = DATA / "raw" / "unified" / "unified.parquet"
    if fast or not uni.exists():
        con.close()
        return
    import pandas as pd
    out("  copertura: interventi del dataset unificato senza testo inglese (english_text = none)")
    u = pd.read_parquet(uni, columns=["uid", "source", "year", "english_text", "category"])
    seg = pd.DataFrame(q("SELECT text_id, COUNT(*), SUM(status = 'done') FROM segments GROUP BY text_id"),
                       columns=["uid", "n", "done"])
    con.close()
    seg["full"] = seg["n"] == seg["done"]
    need = u[u["english_text"] == "none"].merge(seg, on="uid", how="left")
    need["full"] = need["full"].fillna(False).astype(bool)
    need["oral"] = need["category"].astype(str).str.startswith("MEP, oral")
    if len(need) and need["n"].notna().mean() < 0.5:
        out("  ATTENZIONE: meno della meta degli uid si trova in unified.sqlite: gli identificativi forse non coincidono")
    t = need.groupby(["year", "source"]).agg(senza_en=("uid", "size"), tradotti=("full", "sum"),
                                             orali_MEP=("oral", "sum"),
                                             orali_tradotti=("full", lambda x: int((x & need.loc[x.index, "oral"]).sum())))
    t["mancano"] = t["senza_en"] - t["tradotti"]
    out("  " + t.to_string().replace("\n", "\n  "))
    miss = int(t["mancano"].sum())
    miss_oral = int((t["orali_MEP"] - t["orali_tradotti"]).sum())
    out(f"  totale senza traduzione: {miss:,} (di cui orali MEP: {miss_oral:,})")
    if miss_oral:
        todo(f"eTranslation: {miss_oral:,} interventi orali MEP senza testo inglese ne traduzione")
    sc = DATA / "review" / "etranslation"
    out(" punteggi EMI delle traduzioni (score_translations.py)")
    parts = len(list((sc / "chunk_scores").glob("part-*.parquet")))
    out(f"  parti di punteggi: {parts}")
    if not show_file("EMI per intervento tradotto", sc / "speech_emi_mt.parquet"):
        todo("score_translations.py: speech_emi_mt.parquet non c'e" + (f" ({parts} parti gia calcolate)" if parts else ""))
    for f in ("yearly.csv", "trends.csv", "selection.csv"):
        show_file(f, sc / f)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="non leggere il corpus grezzo ne i testi API")
    args = ap.parse_args()
    t0 = time.time()
    section_machine()
    section_pipeline()
    section_sensitivity()
    section_api(args.fast)
    section_unified()
    section_etranslation(args.fast)
    title("6. DA FARE SU QUESTA MACCHINA")
    for i, t in enumerate(TODO, 1):
        out(f"  {i:2}. {t}")
    if not TODO:
        out("  niente")
    rep = DATA / "review" / "inventory" / f"inventory_{socket.gethostname().split('.')[0]}_{datetime.now():%Y%m%d-%H%M}.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nRapporto salvato in {rel(rep)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
