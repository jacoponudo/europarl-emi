"""REVIEW ONLY. Corpus ereditato e API del Parlamento nello stesso schema della Fig. 1.

Domanda: per quali interventi abbiamo una serie continua dal 1996 a oggi, e con quale fonte.

Ogni intervento di ciascuna fonte riceve UNA categoria, con le etichette della Fig. 1
(figures.fig1) e poche etichette in più, esplicite, per ciò che la Fig. 1 non mostra.

Corpus ereditato (1996 - fine corpus), tutte le righe del file grezzo:
  testi inglesi con punteggio (06_speech_annotated) -> regola di figures.fig1:
    MEP, oral, original English (main sample)   ruolo mep, orale, senza "(XX)"
    MEP, oral, translated                       ruolo mep, orale, con "(XX)"
    MEP, written statement                      ruolo mep, "in writing"
    Chair, Commission or Council                ruolo chair / commission / council
    Unrecognised header                         intestazione non riconosciuta (o ruolo unknown)
  testi che non arrivano al punteggio:
    Not English (outside Fig. 1)                language != en (dal 2013: discorsi non inglesi)
    Removed at step 2 (too short)               non in 02_speeches_clean
    Removed at step 3 (procedural)              non in 03_speeches_substantive
    Removed at steps 4-5 (not scored)           non in 06_speech_annotated

API (da luglio 2021), dai metadati strutturati, nello stesso schema:
    MEP, oral, original English (main sample)   PLENARY_DEBATE_SPEECH, SPEAKER, MEP in carica, lingua solo "en"
    MEP, oral, other original language          come sopra, lingua originale non inglese
                                                (fino al 2012 il corpus li ha come "translated";
                                                dal 2013 il corpus li ha solo in lingua originale)
    MEP, oral, several languages                come sopra, più lingue originali (es. "en,fr")
    MEP, written statement                      PLENARY_DEBATE_WRITTEN_STATEMENT
    Chair, Commission or Council                ruolo CHAIR o MEETING_NAME_CHAIR; oppure orale con
                                                intestazione da Commissione/Consiglio (regola del passo 6);
                                                oppure orale di persona non MEP in carica
    Oral, speaker without person id             orale senza person_id (non decidibile)
    Other API activity (no Fig. 1 equivalent)   PROCEEDING_ACTIVITY e altri tipi
  "MEP in carica" = (person_id, legislatura della seduta) presente in mep_database_1994_present.csv.

Abbinamento corpus <-> API nella finestra di sovrapposizione [2021-07-01, min(fine corpus, ultimo file API)].
Testo normalizzato: html.unescape, senza tag, NFKC, minuscolo, solo parole. Tre passi, 1:1:
  1. ultime 20 parole uguali, chiave unica in entrambe le fonti, data qualsiasi;
     da qui la corrispondenza "data del corpus -> giorno di seduta" (moda per data).
  2. ultime 20 parole uguali e stesso giorno di seduta (testi ripetuti appaiati per ordine).
  3. giorno di seduta +-1: 12 parole dal centro di un testo compaiono nell'altro.

Sezioni dell'output:
  A. controlli: i conteggi della Fig. 1 e del campione principale tornano con quelli noti
  B. abbinamento
  C. tabella incrociata categoria corpus x categoria API (il cuore: copertura, incongruenze, scomparse)
  D. il campione principale nei due sensi, per mese e per oratore
  E. altre incongruenze tra abbinati: data, lingua, oratore, lunghezza
  F. continuità 1996 - oggi: conteggi per anno e per mese, fonte disponibile, buchi

Uso (Mac, dalla cartella del progetto; il download API può essere ancora in corso):
    uv run python notebooks/review/api/overlap_corpus_api.py 2>&1 | tee data/review/overlap/run.log
Output (fuori da git) in data/review/overlap/.
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from europarl_emi import config as cfg
from europarl_emi.step06_parse_headers import parse_header
from europarl_emi.step08_models import TERM_STARTS

# ---------------------------------------------------------------- percorsi e parametri

RAW = cfg.RAW_CORPUS / "all_years_combined.csv.gzip"
CLEAN = cfg.INTERIM / "02_speeches_clean.parquet"
SUBSTANTIVE = cfg.INTERIM / "03_speeches_substantive.parquet"
ANNOTATED = cfg.PROCESSED / "06_speech_annotated.parquet"
MAIN_SAMPLE = cfg.PROCESSED / "07_speech_main_sample.parquet"
MEP_DB = cfg.PROCESSED / "mep_database_1994_present.csv"
API_DIR = cfg.DATA / "raw" / "api" / "speeches"
OUT = cfg.DATA / "review" / "overlap"

API_START = pd.Timestamp("2021-07-01")  # inizio dei dati completi dell'API
MARGIN = pd.Timedelta(days=3)  # margine sulle date del corpus attorno alla finestra
ONE_DAY = pd.Timedelta(days=1)
TAIL = 20  # parole della chiave di coda (passi 1-2)
WINDOW = 12  # parole della finestra centrale (passo 3)
MIN_TOKENS = 15  # sotto questa lunghezza la coda non è abbastanza specifica

# Valori noti dal documento di stato (controllo della sezione A)
EXPECTED = {
    "06_speech_annotated": 329_102,
    "07_speech_main_sample": 48_847,
    "MEP, oral, translated": 101_684,
    "MEP, written statement": 96_675,
    "Chair, Commission or Council": 40_974,
    "Unrecognised header": 40_922,
}

# Etichette: identiche a figures.fig1, più quelle fuori dalla Fig. 1
MAIN = "MEP, oral, original English (main sample)"
TRANSLATED = "MEP, oral, translated"
WRITTEN = "MEP, written statement"
CCC = "Chair, Commission or Council"
UNRECOGNISED = "Unrecognised header"
NOT_EN = "Not English (outside Fig. 1)"
DROP2 = "Removed at step 2 (too short)"
DROP3 = "Removed at step 3 (procedural)"
DROP45 = "Removed at steps 4-5 (not scored)"
OTHER_LANG = "MEP, oral, other original language"
SEVERAL = "MEP, oral, several languages"
NO_ID = "Oral, speaker without person id"
OTHER_API = "Other API activity (no Fig. 1 equivalent)"
NONE = "(no match in the other source)"

CORPUS_ORDER = [MAIN, TRANSLATED, WRITTEN, CCC, UNRECOGNISED, NOT_EN, DROP2, DROP3, DROP45]
API_ORDER = [MAIN, OTHER_LANG, SEVERAL, WRITTEN, CCC, NO_ID, OTHER_API]

ORAL, WRITTEN_ACT, CHAIR_ACT = "PLENARY_DEBATE_SPEECH", "PLENARY_DEBATE_WRITTEN_STATEMENT", "MEETING_NAME_CHAIR"

TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"\w+")
ISO_DAY = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 60)


# ---------------------------------------------------------------- utilità

def section(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def check(label, observed, expected):
    flag = "ok" if observed == expected else "DIVERSO"
    print(f"  {label:<45} {observed:>9,}   atteso {expected:>9,}   {flag}")


def norm_id(s):
    """'197490', '197490.0', 'person/197490' -> '197490'; il primo numero se ce ne sono più."""
    return s.astype("string").str.extract(r"(\d+)", expand=False)


def tokens(text):
    if not isinstance(text, str):
        return []
    text = TAG.sub(" ", html.unescape(html.unescape(text)))
    return WORD.findall(unicodedata.normalize("NFKC", text).lower())


def tail_key(t):
    return " ".join(t[-TAIL:]) if len(t) >= MIN_TOKENS else None


def centre(t):
    if len(t) < 2 * WINDOW:
        return None
    m = len(t) // 2
    return " " + " ".join(t[m - WINDOW // 2: m + WINDOW // 2]) + " "


def sitting_day(value):
    """Giorno di calendario come scritto in pubdate (non convertito in UTC: una seduta alle
    00:00+02:00 resta nel suo giorno). Le stringhe non ISO passano da pandas."""
    if not isinstance(value, str):
        return pd.NaT
    m = ISO_DAY.match(value)
    if m:
        return pd.Timestamp(m.group(1))
    d = pd.to_datetime(value, errors="coerce", format="mixed")
    return pd.Timestamp(d.date()) if pd.notna(d) else pd.NaT


TERM_STARTS_TS = sorted((pd.Timestamp(d), int(re.match(r"\d+", k).group())) for k, d in TERM_STARTS.items())


def term_of(day):
    term = None
    for start, t in TERM_STARTS_TS:
        if day >= start:
            term = t
    return term


def crosstab(df, rows, cols, row_order, col_order):
    t = pd.crosstab(df[rows], df[cols])
    t = t.reindex(index=[r for r in row_order if r in t.index] + [r for r in t.index if r not in row_order],
                  columns=[c for c in col_order if c in t.columns] + [c for c in t.columns if c not in col_order],
                  fill_value=0)
    t.loc["Total"] = t.sum()
    t["Total"] = t.sum(axis=1)
    return t


def show_examples(df, cols, text_col, n=4, chars=200):
    if df.empty:
        print("    (nessuno)")
        return
    for r in df.sample(min(n, len(df)), random_state=cfg.SEED).to_dict("records"):
        meta = " | ".join(f"{c}={r.get(c)}" for c in cols)
        text = re.sub(r"\s+", " ", str(r.get(text_col)))[:chars]
        print(f"    - {meta}\n      {text}")


def runs(months):
    """[Period] -> lista di (inizio, fine) dei tratti di mesi consecutivi."""
    out = []
    for m in sorted(months):
        if out and m == out[-1][1] + 1:
            out[-1][1] = m
        else:
            out.append([m, m])
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------- MEP in carica

def load_sitting_meps():
    db = pd.read_csv(MEP_DB, dtype=str)
    db = pd.DataFrame({"mep_id": norm_id(db["mep_id"]), "term": pd.to_numeric(db["term"], errors="coerce")})
    db = db.dropna().astype({"term": int}).drop_duplicates()
    print(f"database MEP: {db.mep_id.nunique():,} persone, {len(db):,} coppie (persona, legislatura); "
          f"legislature {sorted(db.term.unique())}")
    return set(zip(db.mep_id, db.term))


# ---------------------------------------------------------------- corpus ereditato

def load_corpus():
    section("A. Corpus ereditato: categorie della Fig. 1 e controlli")
    parts = []
    reader = pd.read_csv(RAW, compression="gzip", dtype=str, chunksize=50_000,
                         usecols=["guid", "language", "pubdate", "description", "detectedmepid"])
    for ch in reader:
        ch = ch.assign(day=ch["pubdate"].map(sitting_day), lang=ch["language"].str.lower().str.strip())
        # il testo serve solo per l'abbinamento: lo teniamo dalla vigilia della finestra in poi
        ch.loc[~(ch.day >= API_START - MARGIN), "description"] = pd.NA
        parts.append(ch.drop(columns=["pubdate", "language"]))
    c = pd.concat(parts, ignore_index=True)
    print(f"righe nel file grezzo: {len(c):,}; giorno non leggibile: {c.day.isna().sum():,}; "
          f"guid duplicati: {c.guid.duplicated().sum():,}; dal {c.day.min().date()} al {c.day.max().date()}")

    # passo a cui ogni testo si ferma
    in02 = set(pd.read_parquet(CLEAN, columns=["guid"]).guid)
    in03 = set(pd.read_parquet(SUBSTANTIVE, columns=["guid"]).guid)
    ann = pd.read_parquet(ANNOTATED, columns=["guid", "role", "written", "translated", "header_found"])
    main07 = pd.read_parquet(MAIN_SAMPLE, columns=["guid"])

    # regola di figures.fig1, riga per riga
    fig1 = pd.Series(UNRECOGNISED, index=ann.index)
    fig1[ann.role.isin(["chair", "commission", "council"])] = CCC
    mep = (ann.role == "mep") & ann.header_found
    fig1[mep & ann.written] = WRITTEN
    fig1[mep & ~ann.written & ann.translated] = TRANSLATED
    fig1[mep & ~ann.written & ~ann.translated] = MAIN
    ann["fig1"] = fig1

    print("\nControllo dei conteggi della pipeline:")
    check("06_speech_annotated", len(ann), EXPECTED["06_speech_annotated"])
    check("07_speech_main_sample", len(main07), EXPECTED["07_speech_main_sample"])
    for k in (TRANSLATED, WRITTEN, CCC, UNRECOGNISED):
        check(k, int((fig1 == k).sum()), EXPECTED[k])
    same = set(ann.guid[fig1 == MAIN]) == set(main07.guid)
    print(f"  categoria '{MAIN}' = 07_speech_main_sample: {'ok' if same else 'DIVERSO'}")

    c = c.merge(ann[["guid", "fig1"]], on="guid", how="left")
    stage = pd.Series(DROP45, index=c.index)
    stage[~c.guid.isin(in03)] = DROP3
    stage[~c.guid.isin(in02)] = DROP2
    stage[c.lang != "en"] = NOT_EN
    c["cat"] = c["fig1"].fillna(stage)
    c["mep_id"] = norm_id(c["detectedmepid"])
    c = c.drop(columns=["fig1", "detectedmepid"])
    return c


# ---------------------------------------------------------------- API

def load_api(sitting_meps):
    section("A. API: categorie nello schema della Fig. 1")
    files = []
    for f in sorted(API_DIR.glob("*.jsonl")):
        try:
            files.append((pd.Timestamp(f.stem), f))
        except ValueError:
            pass
    rows, bad, nonempty = [], 0, 0
    for day, f in files:
        n0 = len(rows)
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                r.pop("xml", None)
                r["day"] = day
                rows.append(r)
        nonempty += len(rows) > n0
    a = pd.DataFrame(rows)
    last_file = files[-1][0]
    print(f"file giornalieri: {len(files):,} (dal {files[0][0].date()} al {last_file.date()}), "
          f"con interventi: {nonempty}; righe JSON rotte: {bad}")
    if (pd.Timestamp.today().normalize() - last_file).days > 7:
        print("  NB: l'ultimo file è di più di una settimana fa; se il download gira ancora, "
              "i mesi recenti sono incompleti")

    a["sid"] = a["speech_id"].astype("string")
    dup = a.sid.duplicated()
    print(f"speech_id duplicati (tolti): {dup.sum():,}")
    a = a[~dup].reset_index(drop=True)
    if "date" in a:
        d2 = a["date"].map(lambda v: sitting_day(str(v)))
        print(f"campo date diverso dal giorno del file: {(d2 != a.day).sum():,}")

    a["langs"] = a["original_language"].fillna("").astype(str).str.split(",").map(lambda x: [y for y in x if y])
    a["person"] = norm_id(a["person_id"])
    a["term"] = a.day.map(term_of)
    a["sitting_mep"] = [(p, t) in sitting_meps if isinstance(p, str) else False for p, t in zip(a.person, a.term)]

    # regola del passo 6 sull'intestazione dell'API (stessa forma del corpus): ruolo e "in writing"
    parsed = [parse_header(f"{h.strip().rstrip('.')}. – {t}") if isinstance(h, str) and h.strip()
              else (None, None, False, "unknown", None) for h, t in zip(a["header"], a["text"])]
    a["header_role"] = [p[3] for p in parsed]

    act, role = a["activity_type"].fillna(""), a["role"].fillna("")
    oral = (act == ORAL) & (role == "SPEAKER")
    no_id = a.person.isna()
    official = a.header_role.isin(["commission", "council", "chair"])
    cat = pd.Series(OTHER_API, index=a.index)
    cat[(role == "CHAIR") | (act == CHAIR_ACT)] = CCC
    cat[oral & no_id] = NO_ID
    cat[oral & ~no_id & (official | ~a.sitting_mep)] = CCC
    mep_oral = oral & ~no_id & ~official & a.sitting_mep
    cat[mep_oral & a.langs.map(lambda x: x == ["en"])] = MAIN
    cat[mep_oral & a.langs.map(lambda x: len(x) == 1 and x != ["en"])] = OTHER_LANG
    cat[mep_oral & a.langs.map(lambda x: len(x) > 1)] = SEVERAL
    cat[mep_oral & a.langs.map(len).eq(0)] = NO_ID  # nessuna lingua: non decidibile
    cat[act == WRITTEN_ACT] = WRITTEN
    a["cat"] = cat

    print(f"\ninterventi API: {len(a):,}; senza testo originale: {a.text.isna().sum():,}")
    print("\nTipo di attività x ruolo:")
    print(a.groupby(["activity_type", "role"], dropna=False).size().rename("n").to_string())
    print("\nOrali SPEAKER: come è stata decisa la categoria")
    o = a[oral]
    why = pd.Series("MEP in carica", index=o.index)
    why[~o.sitting_mep] = "non MEP in carica -> Chair, Commission or Council"
    why[o.header_role.isin(["commission", "council", "chair"])] = "intestazione da Commissione/Consiglio/presidenza"
    why[o.person.isna()] = "senza person_id"
    print(why.value_counts().to_string())
    print("  di cui MEP in carica ma intestazione da Commissione/Consiglio:",
          int((o.sitting_mep & o.header_role.isin(["commission", "council"])).sum()))
    print("\nCategorie API x ruolo dall'intestazione (regola del passo 6 applicata all'API):")
    print(crosstab(a, "cat", "header_role", API_ORDER, ["mep", "chair", "commission", "council", "unknown"]).to_string())
    a["tok"] = a["text"].map(tokens)
    return a


# ---------------------------------------------------------------- abbinamento

def match(c, a, start, end):
    section(f"B. Abbinamento nella finestra di sovrapposizione {start.date()} - {end.date()}")
    cw = c[c.day.between(start - MARGIN, end + MARGIN)].copy()
    aw = a[a.day.between(start, end)].copy()
    cw["tok"] = cw["description"].map(tokens)
    cw["key"] = cw["tok"].map(tail_key)
    aw["key"] = aw["tok"].map(tail_key)
    print(f"corpus nella finestra (+-{MARGIN.days} giorni): {len(cw):,}; API nella finestra: {len(aw):,}")

    # passo 1
    L = cw[cw.key.notna() & ~cw.key.duplicated(keep=False)]
    A = aw[aw.key.notna() & ~aw.key.duplicated(keep=False)]
    p1 = L[["guid", "key", "day"]].merge(A[["sid", "key", "day"]], on="key", suffixes=("_c", "_a"))
    print(f"passo 1 (coda unica): {len(p1):,} coppie")
    print("  data del corpus - giorno di seduta (giorni):",
          (p1.day_c - p1.day_a).dt.days.value_counts().head(6).to_dict())
    daymap = p1.groupby("day_c")["day_a"].agg(lambda s: s.mode().iat[0])
    cw["sday"] = cw.day.map(daymap).fillna(cw.day)
    moved = daymap[daymap.index != daymap.values]
    print(f"  date del corpus ricondotte a un altro giorno di seduta: {len(moved)} su {len(daymap)}"
          + (f", es. {[(str(x.date()), str(y.date())) for x, y in moved.head(4).items()]}" if len(moved) else ""))

    # passo 2
    done_c, done_a = set(p1.guid), set(p1.sid)
    L = cw[~cw.guid.isin(done_c) & cw.key.notna()][["guid", "key", "sday"]].copy()
    A = aw[~aw.sid.isin(done_a) & aw.key.notna()][["sid", "key", "day"]].rename(columns={"day": "sday"})
    L["occ"] = L.groupby(["key", "sday"]).cumcount()
    A["occ"] = A.groupby(["key", "sday"]).cumcount()
    p2 = L.merge(A, on=["key", "sday", "occ"])[["guid", "sid"]]
    print(f"passo 2 (coda + giorno): {len(p2):,} coppie")

    # passo 3
    done_c |= set(p2.guid)
    done_a |= set(p2.sid)
    Lu, Au = cw[~cw.guid.isin(done_c)], aw[~aw.sid.isin(done_a)]
    l_full = {g: " " + " ".join(t) + " " for g, t in zip(Lu.guid, Lu.tok)}
    a_full = {s: " " + " ".join(t) + " " for s, t in zip(Au.sid, Au.tok)}
    l_mid = {g: centre(t) for g, t in zip(Lu.guid, Lu.tok)}
    a_mid = {s: centre(t) for s, t in zip(Au.sid, Au.tok)}
    by_day = Lu.groupby("sday")["guid"].apply(list).to_dict()
    p3, used = [], set()
    for d, sids in Au.groupby("day")["sid"].apply(list).items():
        cands = [g for off in (-1, 0, 1) for g in by_day.get(d + off * ONE_DAY, [])]
        for s in sids:
            for g in cands:
                if g in used:
                    continue
                if (a_mid[s] and a_mid[s] in l_full[g]) or (l_mid[g] and l_mid[g] in a_full[s]):
                    p3.append((g, s))
                    used.add(g)
                    break
    p3 = pd.DataFrame(p3, columns=["guid", "sid"])
    print(f"passo 3 (centro del testo, giorno +-1): {len(p3):,} coppie")

    pairs = pd.concat([p1[["guid", "sid"]].assign(step=1), p2.assign(step=2), p3.assign(step=3)], ignore_index=True)
    assert not pairs.guid.duplicated().any() and not pairs.sid.duplicated().any(), "abbinamento non 1:1"

    # finestra vera del corpus: giorno di seduta ricondotto, dentro [start, end]
    cw = cw[cw.sday.between(start, end) | cw.guid.isin(pairs.guid)].drop(columns=["key"])
    aw = aw.drop(columns=["key"])
    cw = cw.merge(pairs[["guid", "sid"]], on="guid", how="left")
    aw = aw.merge(pairs[["sid", "guid"]], on="sid", how="left")
    print(f"corpus nella finestra (giorno di seduta): {len(cw):,}, abbinati {cw.sid.notna().sum():,}; "
          f"API: {len(aw):,}, abbinati {aw.guid.notna().sum():,}")
    return cw, aw, pairs


def unmatched_reason(df, other_days, other_keys, own_key_counts):
    """Perché un testo non ha abbinato: seduta assente nell'altra fonte, duplicato, troppo corto, non ritrovato."""
    day = df["sday"] if "sday" in df else df["day"]
    has_day = day.isin(other_days) | (day - ONE_DAY).isin(other_days) | (day + ONE_DAY).isin(other_days)
    key = df["tok"].map(tail_key)
    r = pd.Series("in a sitting present in the other source, not found", index=df.index)
    r[key.map(lambda k: k in other_keys if k else False)] = "same ending as a matched text (duplicate)"
    r[key.map(lambda k: own_key_counts.get(k, 0) > 1 if k else False) & r.str.startswith("in a")] = "duplicate within its own source"
    r[df["tok"].map(len) < MIN_TOKENS] = f"too short to match (< {MIN_TOKENS} words)"
    r[df["tok"].map(len) == 0] = "no text"
    r[~has_day] = "sitting day absent from the other source"
    return r


# ---------------------------------------------------------------- report della sovrapposizione

def report_overlap(cw, aw, c, a):
    cats_c = cw.set_index("guid")["cat"]
    cats_a = aw.set_index("sid")["cat"]
    cw["cat_api"] = cw.sid.map(cats_a).fillna(NONE)
    aw["cat_corpus"] = aw.guid.map(cats_c).fillna(NONE)

    section("C. Categoria nel corpus x categoria nell'API (stessi interventi)")
    both = pd.concat([cw[["cat", "cat_api"]].rename(columns={"cat": "corpus", "cat_api": "api"}),
                      aw.loc[aw.guid.isna(), ["cat"]].rename(columns={"cat": "api"}).assign(corpus=NONE)],
                     ignore_index=True)
    t = crosstab(both, "corpus", "api", CORPUS_ORDER + [NONE], API_ORDER + [NONE])
    print("Righe: categoria nel corpus. Colonne: categoria nell'API. Diagonale = stessa categoria.")
    print("Riga/colonna '(no match...)': testi presenti in una sola fonte.\n")
    print(t.to_string())
    t.to_csv(OUT / "crosswalk_corpus_api.csv")

    # campione principale nei due sensi
    section("D. Campione principale (MEP, oral, original English) nei due sensi")
    api_main = aw[aw.cat == MAIN]
    cor_main = cw[cw.cat == MAIN]
    print(f"API, {MAIN}: {len(api_main):,}")
    print(api_main.cat_corpus.value_counts().rename("nel corpus è").to_string())
    print(f"\nCorpus, {MAIN}: {len(cor_main):,}")
    print(cor_main.cat_api.value_counts().rename("nell'API è").to_string())

    other_days_a, other_days_c = set(a.day), set(c.day)
    keys_a_matched = set(aw.loc[aw.guid.notna(), "tok"].map(tail_key).dropna())
    keys_c_matched = set(cw.loc[cw.sid.notna(), "tok"].map(tail_key).dropna())
    kc = cw.tok.map(tail_key).value_counts().to_dict()
    ka = aw.tok.map(tail_key).value_counts().to_dict()
    cw["reason"] = pd.NA
    aw["reason"] = pd.NA
    m = cw.sid.isna()
    cw.loc[m, "reason"] = unmatched_reason(cw[m], other_days_a, keys_c_matched, kc)
    m = aw.guid.isna()
    aw.loc[m, "reason"] = unmatched_reason(aw[m], other_days_c, keys_a_matched, ka)

    print("\nPerché i testi del campione principale non hanno abbinato:")
    print("  corpus senza API:", cor_main.loc[cor_main.sid.isna()].index.size, "->",
          cw.loc[(cw.cat == MAIN) & cw.sid.isna(), "reason"].value_counts().to_dict())
    print("  API senza corpus:", api_main.loc[api_main.guid.isna()].index.size, "->",
          aw.loc[(aw.cat == MAIN) & aw.guid.isna(), "reason"].value_counts().to_dict())
    print("\nPerché tutti i testi senza abbinato, per categoria:")
    for name, df in (("corpus", cw[cw.sid.isna()]), ("API", aw[aw.guid.isna()])):
        print(f"  {name}:")
        print(pd.crosstab(df.cat, df.reason).to_string())

    print("\nEsempi: API main sample, nel corpus con un'altra categoria:")
    x = api_main[api_main.cat_corpus.ne(MAIN) & api_main.cat_corpus.ne(NONE)].merge(
        cw[["guid", "description"]], on="guid")
    show_examples(x, ["day", "sid", "cat_corpus"], "description")
    print("\nEsempi: API main sample senza abbinato nel corpus, seduta presente:")
    show_examples(aw[(aw.cat == MAIN) & aw.reason.eq("in a sitting present in the other source, not found")],
                  ["day", "sid", "person"], "text")
    print("\nEsempi: corpus main sample senza abbinato nell'API:")
    show_examples(cw[(cw.cat == MAIN) & cw.sid.isna()], ["sday", "guid", "reason"], "description")
    print("\nEsempi: corpus 'Unrecognised header' abbinati a orali MEP in inglese dell'API:")
    show_examples(cw[(cw.cat == UNRECOGNISED) & (cw.cat_api == MAIN)], ["sday", "guid"], "description")

    # per mese
    print("\nCampione principale per mese (corpus = main sample del corpus; API = main sample dell'API):")
    cw["month"] = cw.sday.dt.to_period("M")
    aw["month"] = aw.day.dt.to_period("M")
    mm = pd.DataFrame({
        "corpus": cw[cw.cat == MAIN].groupby("month").size(),
        "corpus in API main": cw[(cw.cat == MAIN) & (cw.cat_api == MAIN)].groupby("month").size(),
        "API": aw[aw.cat == MAIN].groupby("month").size(),
        "API in corpus main": aw[(aw.cat == MAIN) & (aw.cat_corpus == MAIN)].groupby("month").size(),
        "API in corpus, other cat.": aw[(aw.cat == MAIN) & ~aw.cat_corpus.isin([MAIN, NONE])].groupby("month").size(),
    }).fillna(0).astype(int)
    mm["% API ritrovati"] = (100 * (mm["API in corpus main"] + mm["API in corpus, other cat."]) / mm["API"]).round(1)
    mm["% API = corpus main"] = (100 * mm["API in corpus main"] / mm["API"]).round(1)
    print(mm.to_string())
    mm.to_csv(OUT / "main_sample_by_month.csv")

    # per oratore
    print("\nCampione principale API per oratore (almeno 20 interventi): quota ritrovata come main sample nel corpus")
    sp = api_main.groupby("person").agg(n=("sid", "size"), main=("cat_corpus", lambda s: (s == MAIN).sum()),
                                        assente=("cat_corpus", lambda s: (s == NONE).sum()))
    sp["%"] = (100 * sp["main"] / sp["n"]).round(1)
    sp = sp[sp.n >= 20].sort_values("%")
    print(f"  oratori: {len(sp)}; quota mediana {sp['%'].median():.1f}%; sotto il 50%: {(sp['%'] < 50).sum()}")
    print(sp.head(10).to_string())
    sp.to_csv(OUT / "main_sample_by_speaker.csv")
    return cw, aw


def report_inconsistencies(cw, aw):
    section("E. Altre incongruenze tra gli abbinati")
    pe = cw[cw.sid.notna()].merge(
        aw[["sid", "day", "langs", "person", "cat", "activity_type", "role", "text", "tok"]],
        on="sid", suffixes=("_c", "_a"))
    print(f"coppie: {len(pe):,}")

    d = pe[pe.day_c != pe.day_a]
    print(f"\nData: {len(d):,} coppie con data del corpus diversa dal giorno di seduta; "
          f"{(d.sday == d.day_a).sum():,} spiegate da uno spostamento dell'intera data, {(d.sday != d.day_a).sum():,} isolate")

    lang_a = pe.langs.map(lambda x: x[0] if len(x) == 1 else ("mixed" if len(x) > 1 else None))
    lm = pe[lang_a.notna() & lang_a.ne("mixed") & (pe.lang != lang_a)]
    print(f"\nLingua: {len(lm):,} coppie con lingua diversa (corpus 'language' vs API 'original_language'); "
          f"{(lang_a == 'mixed').sum():,} interventi in più lingue")
    if len(lm):
        print(pd.crosstab(lm.lang, lang_a[lm.index]).to_string())
        show_examples(lm, ["sday", "lang", "sid"], "description", 3)

    pm = pe[pe.mep_id.notna() & pe.person.notna()]
    diff = pm[pm.mep_id != pm.person]
    print(f"\nOratore: {len(pe):,} coppie; con id in entrambe {len(pm):,}; id diverso {len(diff):,}; "
          f"solo API {(pe.mep_id.isna() & pe.person.notna()).sum():,}; solo corpus {(pe.mep_id.notna() & pe.person.isna()).sum():,}")
    print("  id diverso, per categoria API:", diff.cat_a.value_counts().to_dict())
    show_examples(diff, ["sday", "mep_id", "person", "sid"], "description", 3)

    pe["n_c"] = pe.tok_c.map(len)
    pe["n_a"] = pe.tok_a.map(len)
    big = pe[pe.n_a >= 50]
    ratio = big.n_c / big.n_a
    print(f"\nLunghezza (parole corpus / parole API, API >= 50 parole, n={len(big):,}; "
          f"il corpus include l'intestazione, quindi ~1,0-1,1 è normale):")
    print("  quantili 1/5/50/95/99%:", ratio.quantile([.01, .05, .5, .95, .99]).round(2).tolist())
    print(f"  corpus molto più corto (< 0,8): {(ratio < 0.8).sum():,}; molto più lungo (> 1,3): {(ratio > 1.3).sum():,}")
    show_examples(big[ratio < 0.8], ["sday", "n_c", "n_a", "sid"], "description", 3)
    pe[["guid", "sid", "day_c", "sday", "day_a", "cat_c", "cat_a"]].to_parquet(OUT / "pairs.parquet", index=False)


# ---------------------------------------------------------------- continuità

def report_continuity(c, a, start, end):
    section("F. Continuità 1996 - oggi nello schema della Fig. 1")
    c = c.assign(year=c.day.dt.year, month=c.day.dt.to_period("M"))
    a = a.assign(year=a.day.dt.year, month=a.day.dt.to_period("M"))

    yc = pd.crosstab(c.year, c.cat).reindex(columns=[k for k in CORPUS_ORDER if k in set(c.cat)], fill_value=0)
    ya = pd.crosstab(a.year, a.cat).reindex(columns=[k for k in API_ORDER if k in set(a.cat)], fill_value=0)
    print("Corpus ereditato per anno (le prime cinque colonne sono la Fig. 1, pannello a):")
    print(yc.to_string())
    print("\nAPI per anno:")
    print(ya.to_string())
    pd.concat({"corpus": yc, "api": ya}, axis=1).to_csv(OUT / "categories_by_year.csv")

    # per mese: giorni di seduta, tutti i testi, main sample
    def per_month(df, name):
        g = df.groupby("month")
        return pd.DataFrame({f"{name} days": g["day"].nunique(), f"{name} all": g.size(),
                             f"{name} main": g["cat"].apply(lambda s: (s == MAIN).sum())})

    months = pd.period_range(c.month.min(), max(c.month.max(), a.month.max()), freq="M")
    mo = per_month(c, "corpus").join(per_month(a, "API"), how="outer").reindex(months).fillna(0).astype(int)
    mo.index.name = "month"
    # completezza del corpus misurata sull'API, solo nella finestra di sovrapposizione
    ref = mo.index[(mo.index >= start.to_period("M")) & (mo.index <= end.to_period("M"))]
    mo["corpus/API main"] = (mo["corpus main"] / mo["API main"].where(mo["API main"] > 0)).round(2)
    mo.loc[~mo.index.isin(ref), "corpus/API main"] = float("nan")
    mo.to_csv(OUT / "continuity_by_month.csv")

    print("\nPer mese dal 2018 (days = giorni di seduta con almeno un testo; main = main sample):")
    print(mo[mo.index >= pd.Period("2018-01", "M")].to_string())

    with_c = set(mo.index[mo["corpus main"] > 0])
    with_a = set(mo.index[mo["API main"] > 0])
    print("\nMesi con almeno un intervento del main sample:")
    print("  corpus:", ", ".join(f"{s}..{e}" for s, e in runs(with_c)) or "-")
    print("  API:   ", ", ".join(f"{s}..{e}" for s, e in runs(with_a)) or "-")
    sitting = set(mo.index[(mo["corpus days"] > 0) | (mo["API days"] > 0)])
    gaps = sorted(m for m in sitting if m not in with_c and m not in with_a)
    print(f"  mesi con sedute ma senza main sample in nessuna fonte: {', '.join(map(str, gaps)) or 'nessuno'}")
    ov = mo.loc[ref, "corpus/API main"].dropna()
    if len(ov):
        print(f"\nCompletezza del main sample del corpus rispetto all'API ({ref.min()}..{ref.max()}): "
              f"mediana {ov.median():.2f}, min {ov.min():.2f} ({ov.idxmin()}), max {ov.max():.2f} ({ov.idxmax()})")
        print("  Fuori da questa finestra la completezza del corpus non si può misurare: l'API non ha dati "
              "nel 2014-2018 né da aprile 2019 a giugno 2021.")

    # grafico: main sample per mese, due fonti
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = mo.index.to_timestamp()
    ax.axvspan(pd.Timestamp("2019-04-01"), API_START, color="#f3f1b5", lw=0, label="no API data (Apr 2019 - Jun 2021)")
    # un mese senza nessuna seduta in una fonte non è uno zero: non si disegna
    ax.plot(x, mo["corpus main"].where(mo["corpus days"] > 0), color="#a40606", lw=1, label=f"corpus: {MAIN}")
    ax.plot(x, mo["API main"].where(mo["API days"] > 0), color="#230007", lw=1, ls="--", label=f"API: {MAIN}")
    ax.set_ylabel("speeches per month")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "continuity_main_sample.png", dpi=200)
    print(f"\ngrafico: {(OUT / 'continuity_main_sample.png').relative_to(cfg.ROOT)}")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meps = load_sitting_meps()
    c = load_corpus()
    a = load_api(meps)
    start, end = API_START, min(c.day.max(), a.day.max())
    print(f"\nFinestra di sovrapposizione: {start.date()} - {end.date()} "
          f"(fine corpus {c.day.max().date()}, ultimo giorno API con interventi {a.day.max().date()})")
    cw, aw, _ = match(c, a, start, end)
    cw, aw = report_overlap(cw, aw, c, a)
    report_inconsistencies(cw, aw)
    report_continuity(c.drop(columns="description"), a, start, end)
    print(f"\nFatto in {time.time() - t0:.0f} s. Tabelle in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
