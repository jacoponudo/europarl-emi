"""REVIEW ONLY. Dataset unificato: corpus ereditato fino al 30-06-2021, API del Parlamento dal 01-07-2021.

Una riga per intervento, stesse colonne per le due fonti, nessun intervento contato due volte.
La sovrapposizione luglio 2021 - novembre 2023 serve solo a controllare le regole (sezione C):
nel dataset quegli interventi vengono dall'API, che è completa (il corpus ha circa il 70%
degli interventi tra ottobre 2022 e marzo 2023 e dal 2020 perde l'intestazione).

Colonne
  uid               "corpus:<guid>" oppure "api:<speech_id>"
  source            corpus / api
  source_id         guid del corpus o speech_id dell'API
  date, year, term  giorno di seduta, anno, legislatura
  person_id         detectedmepid (corpus) o person_id (API), solo cifre
  sitting_mep       person_id è MEP nella legislatura della seduta (mep_database_1994_present.csv)
  speaker, speaker_title   colonne del corpus (vuote per l'API)
  agenda_title      title (corpus) o agenda_title_en (API)
  text_language     lingua del testo in `text`
  original_language lingua in cui l'intervento è stato pronunciato ("unknown" se non ricavabile)
  header            intestazione ("Nome (GRUPPO)"), ricavata con la regola del passo 6 o dall'API
  text              testo nella forma che legge la pipeline: "intestazione. – corpo"
                    (corpus: description invariata; API: header + ". – " + text)
  text_en_mt        traduzione automatica inglese dell'API, solo per gli interventi non inglesi
  english_text      dove si trova un testo inglese: "text (original)", "text (human translation)",
                    "text_en_mt (machine translation)", "none"
  role, written     dalla regola del passo 6 sull'intestazione
  activity_type, api_role   metadati dell'API (vuoti per il corpus)
  category_header   regola della Fig. 1 sull'intestazione, identica alla pipeline, per entrambe le fonti
  category          categoria omogenea (regole sotto)
  category_rule     header / metadata / api
  main_series       category == "MEP, oral, original English (main sample)"

Regola omogenea (category)
  API: dai metadati (stessa regola di overlap_corpus_api.py).
  Corpus, testo inglese: la regola della Fig. 1 (category_header). Solo se l'intestazione non è
    riconosciuta e la seduta è dal FALLBACK_FROM in poi (quando il corpus perde nome e gruppo), si
    decide dai metadati: "in writing" -> dichiarazione scritta; MEP in carica -> orale in inglese;
    persona non MEP in carica -> Chair, Commission or Council; senza persona -> Unrecognised header.
    Prima del 2013 il fallback non è possibile: il testo inglese può essere una traduzione.
  Corpus, testo non inglese: Not English prima del 2013; dal 2013 (lingua del testo = lingua
    originale) dai metadati, con le parole "in writing" e "President" nelle lingue UE.

Uso (Mac, dopo overlap_corpus_api.py, che scrive data/review/overlap/pairs.parquet):
    uv run python notebooks/review/api/build_unified.py 2>&1 | tee data/raw/unified/build.log
Esce: data/raw/unified/unified.parquet e tabelle di riepilogo nella stessa cartella (fuori da git).
"""

from __future__ import annotations

import json
import re
import time

import pandas as pd

from europarl_emi import config as cfg
from europarl_emi.step06_parse_headers import parse_header
from europarl_emi.step08_models import TERM_STARTS

RAW = cfg.RAW_CORPUS / "all_years_combined.csv.gzip"
ANNOTATED = cfg.PROCESSED / "06_speech_annotated.parquet"
MEP_DB = cfg.PROCESSED / "mep_database_1994_present.csv"
API_DIR = cfg.DATA / "raw" / "api" / "speeches"
PAIRS = cfg.DATA / "review" / "overlap" / "pairs.parquet"
OUT = cfg.DATA / "raw" / "unified"

API_START = pd.Timestamp("2021-07-01")  # da qui la fonte è l'API
FALLBACK_FROM = pd.Timestamp("2020-01-01")  # da qui, nel corpus, intestazioni non riconosciute decise dai metadati
ORIGINAL_FROM = pd.Timestamp("2013-01-01")  # da qui il corpus ha ogni testo nella lingua originale

MAIN = "MEP, oral, original English (main sample)"
TRANSLATED = "MEP, oral, translated"
WRITTEN = "MEP, written statement"
CCC = "Chair, Commission or Council"
UNRECOGNISED = "Unrecognised header"
NOT_EN = "Not English (outside Fig. 1)"
OTHER_LANG = "MEP, oral, other original language"
SEVERAL = "MEP, oral, several languages"
NO_ID = "Oral, speaker without person id"
OTHER_API = "Other API activity (no Fig. 1 equivalent)"
ORDER = [MAIN, TRANSLATED, OTHER_LANG, SEVERAL, WRITTEN, CCC, UNRECOGNISED, NOT_EN, NO_ID, OTHER_API]

ORAL, WRITTEN_ACT, CHAIR_ACT = "PLENARY_DEBATE_SPEECH", "PLENARY_DEBATE_WRITTEN_STATEMENT", "MEETING_NAME_CHAIR"

# "in writing" e "President" (presidenza di seduta) nelle lingue UE, per le intestazioni non inglesi
IN_WRITING = re.compile(
    r"in writing|par écrit|schriftlich|per iscritto|por escrito|schriftelijk|na piśmie|írásban|în scris|"
    r"písemně|písomne|pisno|pisanom obliku|писмено|γραπτώς|skriftlig|kirjallisesti|kirjalikult|rakstiski|"
    r"raštu|bil-miktub|i scríbhinn", re.IGNORECASE)
CHAIR_WORDS = {
    "president", "the president", "presidente", "präsident", "präsidentin", "président", "présidente",
    "przewodniczący", "przewodnicząca", "elnök", "elnök asszony", "předsedající", "predsedajúci", "voorzitter",
    "puhemies", "talman", "formanden", "πρόεδρος", "председател", "председателят", "predsjedavajući",
    "predsjedavajuća", "presidenta", "eesistuja", "priekšsēdētājs", "priekšsēdētāja", "pirmininkas",
    "pirmininkė", "președinte", "președintele", "predsedujoči", "predsedujoča", "il-president", "an tuachtarán",
}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def section(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def norm_id(s):
    return s.astype("string").str.extract(r"(\d+)", expand=False)


def sitting_day(value):
    if not isinstance(value, str):
        return pd.NaT
    m = re.match(r"\s*(\d{4}-\d{2}-\d{2})", value)
    if m:
        return pd.Timestamp(m.group(1))
    d = pd.to_datetime(value, errors="coerce", format="mixed")
    return pd.Timestamp(d.date()) if pd.notna(d) else pd.NaT


TERMS = sorted((pd.Timestamp(d), int(re.match(r"\d+", k).group())) for k, d in TERM_STARTS.items())


def term_of(day):
    term = pd.NA
    for start, t in TERMS:
        if pd.notna(day) and day >= start:
            term = t
    return term


def fig1(role, written, translated, found):
    """La regola di figures.fig1 su un solo intervento."""
    if role in ("chair", "commission", "council"):
        return CCC
    if role == "mep" and found:
        return WRITTEN if written else (TRANSLATED if translated else MAIN)
    return UNRECOGNISED


def sitting_meps():
    db = pd.read_csv(MEP_DB, dtype=str)
    db = pd.DataFrame({"mep_id": norm_id(db["mep_id"]), "term": pd.to_numeric(db["term"], errors="coerce")}).dropna()
    return set(zip(db.mep_id, db.term.astype(int)))


def is_sitting(person, term, meps):
    return [(p, int(t)) in meps if isinstance(p, str) and pd.notna(t) else False for p, t in zip(person, term)]


# ---------------------------------------------------------------- corpus

def classify_corpus(c):
    """category per righe del corpus (anche fuori dal periodo usato: serve per il controllo sull'API)."""
    en = c.text_language.eq("en")
    head_low = c.header.fillna("").str.lower().str.strip()
    written_any = c.written | c.header.fillna("").str.contains(IN_WRITING)
    chair_any = head_low.isin(CHAIR_WORDS)

    cat = c.category_header.copy()
    rule = pd.Series("header", index=c.index)

    # testi inglesi con intestazione non riconosciuta, dal FALLBACK_FROM
    fb = en & cat.eq(UNRECOGNISED) & (c.date >= FALLBACK_FROM)
    cat[fb & c.person_id.isna()] = UNRECOGNISED
    cat[fb & c.person_id.notna() & ~c.sitting_mep] = CCC
    cat[fb & c.sitting_mep] = MAIN
    cat[fb & c.sitting_mep & written_any] = WRITTEN
    cat[fb & chair_any] = CCC
    rule[fb] = "metadata"

    # testi non inglesi dal 2013: lingua del testo = lingua originale
    ne = ~en & (c.date >= ORIGINAL_FROM)
    cat[ne & c.person_id.isna()] = NO_ID
    cat[ne & c.person_id.notna() & ~c.sitting_mep] = CCC
    cat[ne & c.sitting_mep] = OTHER_LANG
    cat[ne & c.sitting_mep & written_any] = WRITTEN
    cat[ne & chair_any] = CCC
    cat[ne & c.role.isin(["commission", "council"])] = CCC
    rule[ne] = "metadata"
    return cat, rule


def load_corpus(meps):
    section("A. Corpus ereditato")
    cols = pd.read_csv(RAW, compression="gzip", nrows=0).columns
    use = [x for x in ["guid", "language", "pubdate", "description", "detectedmepid", "speaker", "speaker-title", "title"]
           if x in cols]
    c = pd.read_csv(RAW, compression="gzip", dtype=str, usecols=use)
    for x in ["speaker", "speaker-title", "title", "detectedmepid"]:
        if x not in c:
            c[x] = pd.NA
    c["date"] = c.pubdate.map(sitting_day)
    c["text_language"] = c.language.str.lower().str.strip()
    print(f"righe: {len(c):,}; dal {c.date.min().date()} al {c.date.max().date()}")

    # regola del passo 6 sul testo come lo vede la pipeline (spazi normalizzati, come al passo 2)
    norm = c.description.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    parsed = pd.DataFrame([parse_header(t) for t in norm],
                          columns=["header", "orig_lang", "written", "role", "group_raw"], index=c.index)
    c = pd.concat([c, parsed[["header", "orig_lang", "written", "role"]]], axis=1)
    found = c.header.notna()
    translated = c.orig_lang.notna()
    c["category_header"] = [fig1(r, w, t, f) for r, w, t, f in zip(c.role, c.written, translated, found)]
    c.loc[c.text_language.ne("en"), "category_header"] = NOT_EN

    # controllo: sui testi della Fig. 1 la regola deve dare esattamente le categorie della pipeline
    ann = pd.read_parquet(ANNOTATED, columns=["guid", "role", "written", "translated", "header_found"])
    ann["fig1"] = [fig1(r, w, t, f) for r, w, t, f in zip(ann.role, ann.written, ann.translated, ann.header_found)]
    chk = ann.merge(c[["guid", "category_header"]], on="guid", how="left")
    diff = chk.fig1 != chk.category_header
    print(f"controllo sui {len(ann):,} testi della Fig. 1: categoria diversa da quella della pipeline in "
          f"{diff.sum():,} casi {'(ok)' if diff.sum() == 0 else '(DIVERSO: vedi sotto)'}")
    if diff.any():
        print(pd.crosstab(chk.fig1[diff], chk.category_header[diff].fillna("assente")).to_string())

    c["person_id"] = norm_id(c.detectedmepid)
    c["term"] = c.date.map(term_of)
    c["sitting_mep"] = is_sitting(c.person_id, c.term, meps)
    c["original_language"] = c.text_language
    en = c.text_language.eq("en")
    c.loc[en & translated, "original_language"] = c.orig_lang.str.lower()
    c.loc[en & ~translated & (c.date < ORIGINAL_FROM) & ~found, "original_language"] = "unknown"
    c["english_text"] = "none"
    c.loc[en, "english_text"] = "text (original)"
    c.loc[en & translated, "english_text"] = "text (human translation)"
    c.loc[en & ~translated & (c.date < ORIGINAL_FROM) & ~found, "english_text"] = "text (original or translation)"
    c["category"], c["category_rule"] = classify_corpus(c)

    return pd.DataFrame({
        "uid": "corpus:" + c.guid, "source": "corpus", "source_id": c.guid,
        "date": c.date, "year": c.date.dt.year.astype("Int64"), "term": c.term.astype("Int64"),
        "person_id": c.person_id, "sitting_mep": c.sitting_mep,
        "speaker": c.speaker, "speaker_title": c["speaker-title"], "agenda_title": c.title,
        "text_language": c.text_language, "original_language": c.original_language,
        "header": c.header, "text": c.description, "text_en_mt": pd.NA, "english_text": c.english_text,
        "role": c.role, "written": c.written, "activity_type": pd.NA, "api_role": pd.NA,
        "category_header": c.category_header, "category": c.category, "category_rule": c.category_rule,
    })


# ---------------------------------------------------------------- API

def load_api(meps):
    section("B. API")
    rows = []
    for f in sorted(API_DIR.glob("*.jsonl")):
        try:
            day = pd.Timestamp(f.stem)
        except ValueError:
            continue
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r.pop("xml", None)
                r["day"] = day
                rows.append(r)
    a = pd.DataFrame(rows)
    for x in ["agenda_title_en", "fragment_lang", "text_en_mt", "header", "text"]:
        if x not in a:
            a[x] = pd.NA
    print(f"righe: {len(a):,}; dal {a.day.min().date()} al {a.day.max().date()}")

    # speech_id ripetuti: stessa riga oppure interventi diversi?
    dup = a[a.speech_id.duplicated(keep=False)]
    same = dup.duplicated(subset=["speech_id", "day", "text", "person_id", "activity_type"], keep=False)
    print(f"speech_id ripetuti: {dup.speech_id.nunique():,} identificativi su {len(dup):,} righe; "
          f"righe identiche (testo, giorno, persona, tipo): {same.sum():,}; diverse: {(~same).sum():,}")
    if (~same).any():
        d = dup[~same]
        print("  le righe diverse differiscono per:",
              {k: int(d.groupby("speech_id")[k].nunique(dropna=False).gt(1).sum())
               for k in ["day", "text", "person_id", "activity_type", "role"]})
    a = a.drop_duplicates(subset=["speech_id", "day", "text", "person_id", "activity_type"]).reset_index(drop=True)
    k = a.groupby("speech_id").cumcount()
    a["source_id"] = a.speech_id.where(k == 0, a.speech_id + "#" + (k + 1).astype(str))
    print(f"dopo aver tolto le righe identiche: {len(a):,}; identificativi resi unici con #n: {(k > 0).sum():,}")

    a["person"] = norm_id(a.person_id)
    a["term"] = a.day.map(term_of)
    a["sitting_mep"] = is_sitting(a.person, a.term, meps)
    langs = a.original_language.fillna("").str.split(",").map(lambda x: [y for y in x if y])
    a["original_language"] = langs.map(lambda x: ",".join(x) if x else "unknown")
    a["text_language"] = a.fragment_lang.fillna(a.original_language)

    head = a.header.fillna("").str.strip().str.rstrip(".").str.strip()
    a["full"] = [f"{h}. – {t}" if h else t for h, t in zip(head, a.text.fillna(""))]
    parsed = [parse_header(t) for t in a.full]
    a["role_h"] = [p[3] for p in parsed]
    a["written_h"] = [p[2] for p in parsed]
    found = [p[0] is not None for p in parsed]
    a["category_header"] = [fig1(r, w, False, f) for r, w, f in zip(a.role_h, a.written_h, found)]
    a.loc[a.text_language.ne("en"), "category_header"] = NOT_EN

    act, role = a.activity_type.fillna(""), a.role.fillna("")
    oral = (act == ORAL) & (role == "SPEAKER")
    official = a.role_h.isin(["commission", "council", "chair"])
    cat = pd.Series(OTHER_API, index=a.index)
    cat[(role == "CHAIR") | (act == CHAIR_ACT)] = CCC
    cat[oral & a.person.isna()] = NO_ID
    cat[oral & a.person.notna() & (official | ~a.sitting_mep)] = CCC
    mep_oral = oral & a.person.notna() & ~official & a.sitting_mep
    cat[mep_oral & langs.map(lambda x: x == ["en"])] = MAIN
    cat[mep_oral & langs.map(lambda x: len(x) == 1 and x != ["en"])] = OTHER_LANG
    cat[mep_oral & langs.map(lambda x: len(x) > 1)] = SEVERAL
    cat[mep_oral & langs.map(len).eq(0)] = NO_ID
    cat[act == WRITTEN_ACT] = WRITTEN

    en_orig = a.original_language.eq("en")
    eng = pd.Series("none", index=a.index)
    eng[en_orig] = "text (original)"
    eng[~en_orig & a.text_en_mt.notna()] = "text_en_mt (machine translation)"
    return pd.DataFrame({
        "uid": "api:" + a.source_id, "source": "api", "source_id": a.source_id,
        "date": a.day, "year": a.day.dt.year.astype("Int64"), "term": a.term.astype("Int64"),
        "person_id": a.person, "sitting_mep": a.sitting_mep,
        "speaker": pd.NA, "speaker_title": pd.NA, "agenda_title": a.get("agenda_title_en"),
        "text_language": a.text_language, "original_language": a.original_language,
        "header": a.header, "text": a.full, "text_en_mt": a.text_en_mt.where(~en_orig), "english_text": eng,
        "role": a.role_h, "written": a.written_h, "activity_type": a.activity_type, "api_role": a.role,
        "category_header": a.category_header, "category": cat, "category_rule": "api",
    })


# ---------------------------------------------------------------- controllo sulla sovrapposizione

def validate(corpus, api):
    section("C. Controllo della regola del corpus sulla sovrapposizione (stessi interventi nelle due fonti)")
    if not PAIRS.exists():
        print(f"manca {PAIRS.relative_to(cfg.ROOT)}: lancia prima overlap_corpus_api.py")
        return
    p = pd.read_parquet(PAIRS, columns=["guid", "sid"])
    c = corpus.set_index("source_id")[["category", "category_rule", "category_header"]]
    a = api.drop_duplicates("source_id").set_index("source_id")["category"]
    v = p.join(c, on="guid").join(a.rename("api"), on="sid").dropna(subset=["category", "api"])
    print(f"coppie: {len(v):,}")
    print("\nRegola del paper (category_header) x categoria API:")
    print(pd.crosstab(v.category_header, v.api, margins=True).to_string())
    print("\nRegola omogenea (category) x categoria API:")
    print(pd.crosstab(v.category, v.api, margins=True).to_string())
    for name, col in (("regola del paper", "category_header"), ("regola omogenea", "category")):
        pred, true = v[col].eq(MAIN), v.api.eq(MAIN)
        tp = (pred & true).sum()
        print(f"\n{name}: '{MAIN}'")
        print(f"  del corpus classificati così, quanti lo sono anche per l'API (precisione): "
              f"{tp:,}/{pred.sum():,} = {tp / max(pred.sum(), 1):.1%}")
        print(f"  dei casi API, quanti il corpus li classifica così (copertura): "
              f"{tp:,}/{true.sum():,} = {tp / max(true.sum(), 1):.1%}")
    fb = v[v.category_rule.eq("metadata") & v.category.eq(MAIN)]
    print("\nCasi decisi dai metadati come main sample, categoria API:", fb.api.value_counts().to_dict())


# ---------------------------------------------------------------- riepilogo

def summary(u):
    section("D. Dataset unificato")
    print(f"righe: {len(u):,}; uid unici: {u.uid.is_unique}; "
          f"corpus {u.source.eq('corpus').sum():,} (fino al {u[u.source == 'corpus'].date.max().date()}), "
          f"API {u.source.eq('api').sum():,} (dal {u[u.source == 'api'].date.min().date()} al {u.date.max().date()})")
    by = pd.crosstab(u.year, u.category).reindex(columns=[k for k in ORDER if k in set(u.category)], fill_value=0)
    print("\nCategoria omogenea per anno:")
    print(by.to_string())
    by.to_csv(OUT / "categories_by_year.csv")

    m = u[u.main_series]
    ms = pd.crosstab(m.year, m.category_rule).reindex(columns=["header", "metadata", "api"], fill_value=0)
    ms["total"] = ms.sum(axis=1)
    ms["paper rule"] = u[u.category_header.eq(MAIN)].groupby("year").size().reindex(ms.index, fill_value=0)
    print(f"\nSerie principale ({MAIN}) per anno e regola che l'ha decisa; 'paper rule' = regola della Fig. 1:")
    print(ms.to_string())
    ms.to_csv(OUT / "main_series_by_year.csv")

    mo = m.groupby([m.date.dt.to_period("M"), "source"]).size().unstack(fill_value=0)
    print("\nSerie principale per mese dal 2019:")
    print(mo[mo.index >= pd.Period("2019-01", "M")].to_string())
    mo.to_csv(OUT / "main_series_by_month.csv")
    print("\nTesto inglese disponibile, per fonte:")
    print(pd.crosstab(u.english_text, u.source).to_string())


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meps = sitting_meps()
    corpus = load_corpus(meps)
    api = load_api(meps)
    validate(corpus, api)
    u = pd.concat([corpus[corpus.date < API_START], api[api.date >= API_START]], ignore_index=True)
    u["main_series"] = u.category.eq(MAIN)
    u = u.sort_values(["date", "source", "source_id"]).reset_index(drop=True)
    for col in ["written", "sitting_mep", "main_series"]:
        u[col] = u[col].astype("boolean")
    summary(u)
    u.to_parquet(OUT / "unified.parquet", index=False)
    print(f"\nSalvato {(OUT / 'unified.parquet').relative_to(cfg.ROOT)} ({len(u):,} righe) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
