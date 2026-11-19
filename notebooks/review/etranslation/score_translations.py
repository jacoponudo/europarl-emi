"""REVIEW ONLY. EMI of the speeches machine-translated into English with eTranslation, measured exactly
as in the paper, and the figures that extend the "including translations" series past 2012.

Sources
  data/interim/etranslation/unified.sqlite   translations (translate_unified.py): one row per segment,
                                             text_id = uid of unified.parquet
  data/raw/unified/unified.parquet           metadata of every speech (build_unified.py): date, person,
                                             homogeneous category, original header
  Also scored, with the same measure: the API speeches delivered in English (MEP, oral, from 1 July 2021),
  so that the English series continues where the corpus is incomplete.

Same measure as the paper (steps 2-5)
  step 2  at least 3 words and 16 characters, spaces normalised
  step 3  procedural filter: the classifier in models/procedural_nblr.joblib, texts < 400 characters
  step 4  chunks of <= 150 words (last < 50 merged), all-MiniLM-L6-v2, cosine with the mean lexicon
          vectors, header kept in the text ("Name (GROUP). - ..."; if the translation lost the header,
          the original header from unified.parquet is put back in front)
  step 5  length correction (10-word bins) and z-scores ON THE PAPER'S SCALE: bin means, mean and
          standard deviation come from the paper's chunks (data/processed/05_chunk_emi.parquet), so the
          paper's numbers do not change and the new speeches are directly comparable. The script first
          recomputes the paper's chunk EMI from those parameters and stops if it differs.
Selection: the homogeneous category of unified.parquet ("MEP, oral, other original language" for the
translated speeches, "MEP, oral, original English" for the API speeches in English). The step-6 header
rule applied to the translated text is reported next to it as a check.
With a fixed scale the EMI of a speech does not depend on the other speeches, so only the selected
speeches are scored (--all scores every translated text).

Use (Mac, from the project folder):
    mkdir -p data/review/etranslation
    caffeinate -i uv run python notebooks/review/etranslation/score_translations.py 2>&1 | tee data/review/etranslation/score.log
Options:
    --all            score every translated text, not only MEP oral speeches
    --figures-only   redo tables and figures from data/review/etranslation/speech_emi_mt.parquet

Out (all outside git):
    data/review/etranslation/chunk_scores/part-*.parquet  resumable: rerun to continue
    data/review/etranslation/speech_emi_mt.parquet         one row per scored speech
    data/review/etranslation/yearly.csv, trends.csv, selection.csv
    results/figures/review/etranslation/figT1-figT3 (.pdf, .png)
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from europarl_emi import config as cfg
from europarl_emi import figures as F
from europarl_emi.step06_parse_headers import parse_header
from europarl_emi.step08_models import ANNOTATED, CHUNKS, fit_trend, load_main, normalise_id

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from build_unified import API_START, MAIN, OTHER_LANG  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend already set by figures)

DB = cfg.INTERIM / "etranslation" / "unified.sqlite"
UNIFIED = cfg.DATA / "raw" / "unified" / "unified.parquet"
PROC_MODEL = cfg.MODELS / "procedural_nblr.joblib"
OUT = cfg.DATA / "review" / "etranslation"
SCORES = OUT / "chunk_scores"
SPEECH_OUT = OUT / "speech_emi_mt.parquet"
FIG = cfg.FIGURES / "review" / "etranslation"
PART_SIZE = 20_000
MAX_CHARS_TO_CLASSIFY = 400  # as in step 3
UCOLS = ["uid", "source", "date", "year", "person_id", "original_language", "header", "category", "text"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def section(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


# ---------------------------------------------------------------- input

def load_inputs():
    section("1. Traduzioni e metadati")
    for p in (DB, UNIFIED):
        if not p.exists():
            sys.exit(f"Manca {p.relative_to(cfg.ROOT)}")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    texts = pd.read_sql_query("SELECT text_id, lang, n_segments FROM texts", con)
    seg = pd.read_sql_query("SELECT text_id, idx, translation FROM segments WHERE status = 'done'", con)
    con.close()
    seg = seg.sort_values(["text_id", "idx"])
    tr = seg.groupby("text_id").agg(text_en=("translation", lambda s: " ".join(s.fillna(""))),
                                    n_done=("idx", "size")).reset_index()
    tr = tr.merge(texts, on="text_id", how="left")
    complete = tr["n_done"] == tr["n_segments"]
    print(f"testi in unified.sqlite: {len(texts):,}; tradotti per intero: {complete.sum():,}")
    tr = tr[complete].rename(columns={"text_id": "uid", "lang": "mt_language"})
    tr["mt_language"] = tr["mt_language"].str.lower()

    u = pd.read_parquet(UNIFIED, columns=UCOLS)
    u["year"] = u["year"].astype("Int64")
    mt = tr[["uid", "mt_language", "text_en"]].merge(u.drop(columns="text"), on="uid", how="left")
    lost = mt["source"].isna().sum()
    print(f"tradotti senza riga in unified.parquet: {lost:,}")
    mt = mt[mt["source"].notna()].copy()
    print("\nTesti tradotti per anno e categoria (unified.parquet):")
    ct = pd.crosstab(mt["year"], mt["category"])
    print(ct.to_string())
    ct.to_csv(OUT / "translated_by_year_category.csv")

    en = u[(u["source"] == "api") & (u["category"] == MAIN) & (u["date"] >= API_START)].copy()
    en = en.rename(columns={"text": "text_en"})
    en["mt_language"] = "en"
    print(f"\nInterventi API in inglese (MEP, orali, dal {API_START.date()}): {len(en):,}")
    mt["kind"], en["kind"] = "machine", "api_english"
    return pd.concat([mt, en.drop(columns=[c for c in en.columns if c not in mt.columns])], ignore_index=True)


# ---------------------------------------------------------------- steps 2, 3 and selection

def clean_filter_annotate(df):
    section("2. Intestazione, pulizia e filtro procedurale (passi 2, 3)")
    df = df.copy()
    df["text"] = df["text_en"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop(columns="text_en")
    found = pd.Series([parse_header(t)[0] is not None for t in df["text"]], index=df.index)
    head = df["header"].fillna("").astype(str).str.strip().str.rstrip(".").str.strip()
    put_back = ~found & head.ne("")
    df.loc[put_back, "text"] = head[put_back] + ". – " + df.loc[put_back, "text"]
    print("intestazione originale rimessa davanti (la traduzione non l'aveva):")
    print(pd.crosstab(df["kind"] + "/" + df["source"], put_back.map({True: "rimessa", False: "non serve"}),
                      rownames=["tipo/fonte"], colnames=["intestazione"]).to_string())

    df["n_words"] = df["text"].str.split().str.len()
    keep = (df["n_words"] >= cfg.MIN_WORDS) & (df["text"].str.len() >= cfg.MIN_CHARS)
    print(f"\ntolti perché troppo corti: {(~keep).sum():,}")
    df = df[keep].copy()

    if PROC_MODEL.exists():
        import joblib
        from europarl_emi.step03_filter_procedural import predict, simplify
        model = joblib.load(PROC_MODEL)
        simple = df["text"].map(simplify)
        short = (simple.str.len() < MAX_CHARS_TO_CLASSIFY).to_numpy()
        proba = np.zeros(len(df))
        if short.any():
            proba[short] = predict(model, simple[short].tolist())
        proc = proba >= 0.5
        print(f"testi brevi esaminati: {short.sum():,}; procedurali tolti: {proc.sum():,}")
        df = df[~proc].copy()
    else:
        print(f"ATTENZIONE: manca {PROC_MODEL.relative_to(cfg.ROOT)}, filtro procedurale NON applicato")

    parsed = [parse_header(t) for t in df["text"]]
    df["role_h"] = [p[3] for p in parsed]
    df["written_h"] = [bool(p[2]) for p in parsed]
    df["header_rule_oral"] = (df["role_h"] == "mep") & ~df["written_h"]
    df["selected"] = ((df["kind"] == "machine") & (df["category"] == OTHER_LANG)) | \
                     ((df["kind"] == "api_english") & (df["category"] == MAIN))
    df["mep_id"] = normalise_id(df["person_id"])

    m = df[df["kind"] == "machine"]
    print("\nTradotti: categoria omogenea x regola dell'intestazione (passo 6) sul testo tradotto:")
    sel = pd.crosstab(m["category"], m["header_rule_oral"].map({True: "MEP orale (intest.)", False: "altro"}),
                      margins=True)
    print(sel.to_string())
    sel.to_csv(OUT / "selection.csv")
    by = pd.crosstab(df["year"], df["kind"].where(df["selected"], "non selezionati"))
    print("\nSelezionati per anno:")
    print(by.to_string())
    s = df[df["selected"] & (df["kind"] == "machine")]
    print("\nLingue originali dei tradotti selezionati:", s["mt_language"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------- steps 4, 5

def paper_scale():
    """Length-bin means, mean and sd of the paper (step 5), checked by recomputing the paper's chunk EMI."""
    section("3. Scala del paper (passo 5)")
    ref = pd.read_parquet(CHUNKS, columns=["n_words", "sim_evidence", "sim_intuition", "emi"])
    bins = np.arange(cfg.SCORE_MIN_WORDS, ref["n_words"].max() + 10, 10)
    length_bin = pd.cut(ref["n_words"], bins)
    params, z = {}, {}
    for name in ("evidence", "intuition"):
        sim = ref[f"sim_{name}"]
        adjusted = sim - sim.groupby(length_bin, observed=True).transform("mean")
        means = sim.groupby(length_bin, observed=False).mean()
        params[name] = (means, adjusted.mean(), adjusted.std())
        z[name] = (adjusted - params[name][1]) / params[name][2]
    diff = float(np.abs((z["evidence"] - z["intuition"]) - ref["emi"]).max())
    print(f"pezzi del paper: {len(ref):,}; EMI ricalcolata con questi parametri, differenza massima: {diff:.2e}")
    if diff > 1e-4:
        sys.exit("La scala del paper non si riproduce: controlla 05_chunk_emi.parquet prima di andare avanti.")
    for name, (_, mu, sd) in params.items():
        print(f"  {name}: media dopo la correzione {mu:.2e}, deviazione standard {sd:.5f}")
    return bins, params


def score(df):
    section("4. Similarità con i lessici (passo 4, MiniLM)")
    from sentence_transformers import SentenceTransformer, util
    from tqdm import tqdm

    from europarl_emi.step04_score_chunks import lexicon_vectors, make_chunks, pick_device

    SCORES.mkdir(parents=True, exist_ok=True)
    files = sorted(SCORES.glob("part-*.parquet"))
    done = set(pd.concat([pd.read_parquet(f, columns=["guid"]) for f in files]).guid) if files else set()
    todo = df.loc[~df["uid"].isin(done), ["uid", "text", "n_words"]].rename(columns={"uid": "guid"})
    chunks = make_chunks(todo)
    print(f"testi già con punteggio: {len(done & set(df.uid)):,}; pezzi da calcolare ora: {len(chunks):,}")
    if len(chunks):
        device = pick_device()
        print(f"dispositivo: {device}")
        model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
        reference = lexicon_vectors(model)
        start = max([int(f.stem.split("-")[1]) for f in files], default=-1) + 1
        n_parts = int(np.ceil(len(chunks) / PART_SIZE))
        for p in tqdm(range(n_parts), desc="Parti"):
            part = chunks.iloc[p * PART_SIZE:(p + 1) * PART_SIZE].copy()
            emb = model.encode(part["chunk_text"].tolist(), batch_size=128, convert_to_tensor=True)
            sims = util.cos_sim(emb, reference).cpu().numpy()
            part["sim_evidence"] = sims[:, 0]
            part["sim_intuition"] = sims[:, 1]
            path = SCORES / f"part-{start + p:04d}.parquet"
            tmp = path.with_suffix(".tmp")
            part.drop(columns="chunk_text").to_parquet(tmp, index=False)
            tmp.rename(path)
    files = sorted(SCORES.glob("part-*.parquet"))
    sc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return sc[sc["guid"].isin(set(df["uid"]))].drop_duplicates(["guid", "chunk_id"])


def apply_scale(chunks, bins, params):
    n = chunks["n_words"].clip(lower=cfg.SCORE_MIN_WORDS + 1, upper=bins[-1])
    codes = pd.cut(n, bins).cat.codes.to_numpy()
    for name in ("evidence", "intuition"):
        means, mu, sd = params[name]
        m = means.ffill().bfill().to_numpy()
        adjusted = chunks[f"sim_{name}"].to_numpy() - m[codes]
        chunks[f"{name}_z"] = (adjusted - mu) / sd
    chunks["emi"] = chunks["evidence_z"] - chunks["intuition_z"]
    return chunks


def speeches(df, chunks):
    sp = chunks.groupby("guid").agg(emi=("emi", "mean"), evidence_z=("evidence_z", "mean"),
                                    intuition_z=("intuition_z", "mean"), n_chunks=("chunk_id", "size"),
                                    n_words_scored=("n_words", "sum")).reset_index().rename(columns={"guid": "uid"})
    return df.drop(columns="text").merge(sp, on="uid", how="inner")


# ---------------------------------------------------------------- analysis

def with_cluster(d):
    d = d.dropna(subset=["year", "emi"]).copy()
    d["year"] = d["year"].astype(int)
    d["mep_id"] = d["mep_id"].astype("string")
    d["cluster"] = d["mep_id"].fillna("speech:" + d["uid"].astype("string"))
    d["trend"] = (d["year"] - 2000) / 10
    return d[["uid", "year", "emi", "mep_id", "cluster", "trend", "source", *[c for c in ["lang"] if c in d]]]


def analysis_sets(sp):
    start = pd.Timestamp(API_START, tz="UTC")
    paper = load_main(period=None)
    paper = paper[paper["pubdate"] < start].assign(uid="corpus:" + paper["guid"].astype(str), source="original")
    api_en = sp[sp["selected"] & (sp["kind"] == "api_english")].assign(source="original")
    eng = with_cluster(pd.concat([paper[["uid", "year", "emi", "mep_id", "source"]],
                                  api_en[["uid", "year", "emi", "mep_id", "source"]]], ignore_index=True))

    ann = pd.read_parquet(ANNOTATED, columns=["guid", "year", "pubdate", "emi", "role", "written", "translated",
                                              "detectedmepid"])
    ann["pubdate"] = pd.to_datetime(ann["pubdate"], utc=True)
    oral = ann[(ann["role"] == "mep") & ~ann["written"] & (ann["pubdate"] < start)].copy()
    oral["source"] = np.where(oral["translated"], "human", "original")
    oral["mep_id"] = normalise_id(oral["detectedmepid"])
    oral.loc[oral["detectedmepid"].isna(), "mep_id"] = pd.NA
    oral["uid"] = "corpus:" + oral["guid"].astype(str)
    oral = with_cluster(oral)

    m = sp[sp["selected"] & (sp["kind"] == "machine")].assign(source="machine", lang=lambda d: d["mt_language"])
    mtm = with_cluster(m)
    incl = pd.concat([oral, with_cluster(api_en), mtm.drop(columns="lang")], ignore_index=True)
    return eng, oral, mtm, incl


def yearly_table(eng, mtm, incl):
    section("5. EMI medio per anno (per discorso)")
    g = lambda d: d.groupby("year")["emi"].agg(["size", "mean"])  # noqa: E731
    y = pd.concat({"English": g(eng), "machine": g(mtm), "incl. translations": g(incl)}, axis=1)
    y = y[y.index >= 2009]
    y.to_csv(OUT / "yearly.csv")
    print(y.round(3).to_string())
    for a, b in ((2013, 2019), (2022, 2026)):
        e, m = eng[eng.year.between(a, b)].emi, mtm[mtm.year.between(a, b)].emi
        if len(e) and len(m):
            print(f"differenza di livello {a}-{b}, tradotti a macchina meno inglese: {m.mean() - e.mean():+.3f}")
    print("(deviazione standard dell'EMI per discorso nel paper: 0,636)")


def trends(eng, mtm, incl):
    section("6. Trend per decennio (errori raggruppati per oratore)")
    sel = lambda d, a, b: d[d["year"].between(a, b)]  # noqa: E731
    specs = [
        (sel(eng, 1996, 2019), [], "Delivered in English 1996-2019 (paper: -0.182)"),
        (sel(eng, 1996, 2019), ["mep_id"], "Delivered in English 1996-2019, speaker FE (paper: -0.171)"),
    ]
    for a, b in ((2013, 2019), (2022, 2026)):
        specs += [
            (sel(eng, a, b), [], f"Delivered in English {a}-{b}"),
            (sel(eng, a, b), ["mep_id"], f"Delivered in English {a}-{b}, speaker FE"),
            (sel(mtm, a, b), [], f"Machine-translated {a}-{b}"),
            (sel(mtm, a, b), ["mep_id"], f"Machine-translated {a}-{b}, speaker FE"),
            (sel(incl, a, b), ["source"], f"English + machine-translated {a}-{b}, source FE"),
            (sel(incl, a, b), ["mep_id", "source"], f"English + machine-translated {a}-{b}, speaker + source FE"),
        ]
    specs += [
        (sel(eng, 1996, 2026), [], "Delivered in English 1996-2026"),
        (sel(eng, 1996, 2026), ["mep_id"], "Delivered in English 1996-2026, speaker FE"),
        (sel(incl, 1996, 2026), [], "Including translations 1996-2026"),
        (sel(incl, 1996, 2026), ["source"], "Including translations 1996-2026, source FE"),
        (sel(incl, 1996, 2026), ["mep_id", "source"], "Including translations 1996-2026, speaker + source FE"),
    ]
    rows = []
    for d, fes, label in specs:
        if len(d) < 50 or d["year"].nunique() < 3:
            print(f"  saltato (pochi dati): {label}")
            continue
        rows.append(fit_trend(d, fes, label))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "trends.csv", index=False)
    show = t.copy()
    for c in ("coef", "se", "ci_low", "ci_high"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("(source FE: originale / traduzione umana / traduzione automatica)")


# ---------------------------------------------------------------- figures

LOW = {2020, 2021}  # no complete source: corpus incomplete, API complete only from July 2021


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  salvata {(FIG / name).relative_to(cfg.ROOT)}.pdf / .png")


def fig_t1(eng, oral, incl):
    """Fig. 2 extended to 2026: human translations until 2012, machine translations from 2013."""
    series = [
        ("Including human translations", F.yearly(oral[oral.year <= 2012], "emi"), F.GREY, F.SAGE, F.SAGE, 0.45, "--", "o"),
        ("Including machine translations", F.yearly(incl[incl.year >= 2013], "emi"), F.DARK, "white", F.DARK, 0.12, ":", "s"),
        ("Delivered in English", F.yearly(eng, "emi"), F.OCHRE, F.OCHRE, F.PEACH, 0.5, "-", "o"),
    ]
    n_max = max(s["count"].max() for _, s, *_ in series)
    area = lambda n: 3 + 60 * np.asarray(n) / n_max  # noqa: E731
    fig, ax = plt.subplots(figsize=(F.DOUBLE * 0.75, 75 * F.MM))
    F.zero_line(ax)
    for b in F.TERM_BOUNDARIES + [2024.5]:
        ax.axvline(b, color="#d0d0d0", linewidth=0.4, zorder=0)
    ax.axvspan(min(LOW) - 0.5, max(LOW) + 0.5, color="#f2f2f2", zorder=0, linewidth=0)
    handles = []
    for z, (label, s, line, fill, band, alpha, ls, mk) in enumerate(series):
        ax.fill_between(s.index, s["lo"], s["hi"], color=band, alpha=alpha, linewidth=0, zorder=1 + z)
        ax.plot(s.index, s["mean"], color=line, ls=ls, linewidth=1, zorder=3 + z)
        ok, low = s[~s.index.isin(LOW)], s[s.index.isin(LOW)]
        ax.scatter(ok.index, ok["mean"], s=area(ok["count"]), marker=mk, facecolor=fill, edgecolor=line,
                   linewidth=0.6, zorder=5 + z)
        ax.scatter(low.index, low["mean"], s=area(low["count"]), marker=mk, facecolor="white", edgecolor=line,
                   linewidth=0.6, zorder=5 + z)
        handles.append(plt.Line2D([], [], color=line, ls=ls, marker=mk, markersize=3.5, markerfacecolor=fill,
                                  markeredgecolor=line, label=label))
    ax.xaxis.set_major_locator(F.MaxNLocator(integer=True))
    ax.axvline(2012.5, color=F.INK, linewidth=0.5, ls=":", zorder=1)
    tx = ax.get_xaxis_transform()
    ax.text(2012.3, 0.97, "Human → machine\ntranslation", transform=tx, ha="right", va="top", fontsize=5.5,
            color=F.MUTED)
    for x, lab in zip([1997.75, 2002, 2007, 2012, 2017, 2022, 2025.5], ["4th", "5th", "6th", "7th", "8th", "9th", "10th"]):
        ax.text(x, 1.01, f"{lab} term", transform=tx, ha="center", fontsize=5.5, color=F.MUTED)
    ax.text(2021, 0.02, "Low\ncoverage", transform=tx, ha="center", va="bottom", fontsize=5, color=F.MUTED)
    first = ax.legend(handles=handles[::-1], loc="upper right", fontsize=6)
    ax.add_artist(first)
    ref = [n for n in (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000) if n <= n_max]
    ref = sorted({ref[0], ref[len(ref) // 2], ref[-1]})
    sizes = [ax.scatter([], [], s=area(n), color=F.MUTED, linewidth=0, label=f"{n:,}") for n in ref]
    ax.legend(handles=sizes, title="Speeches per year", loc="lower left", fontsize=5.5, title_fontsize=5.5,
              ncol=len(sizes), columnspacing=1.0, handletextpad=0.3, borderpad=0.6)
    ax.set_ylabel("EMI (evidence minus intuition)")
    ax.set_xlabel("Year")
    save(fig, "figT1_emi_by_year_with_machine_translations")


def fig_t2(eng, mtm, n_lang=6):
    """Machine-translated speeches by original language, against speeches delivered in English."""
    langs = mtm["lang"].value_counts().index[:n_lang].tolist()
    if not langs:
        return
    ref = F.yearly(eng[eng.year >= 2013], "emi")
    ncol = 3
    nrow = int(np.ceil(len(langs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(F.DOUBLE, 45 * F.MM * nrow), sharex=True, sharey=True,
                             squeeze=False)
    for ax, lang in zip(axes.flat, langs):
        s = F.yearly(mtm[mtm.lang == lang], "emi", min_n=20)
        F.zero_line(ax)
        ax.axvspan(min(LOW) - 0.5, max(LOW) + 0.5, color="#f2f2f2", zorder=0, linewidth=0)
        F.plot_series(ax, ref, F.LINES[1], "Delivered in English")
        F.plot_series(ax, s, F.LINES[0], "Machine-translated from this language")
        ax.set_title(f"{lang.upper()} (n = {int((mtm.lang == lang).sum()):,})", loc="left")
    for ax in axes.flat[len(langs):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    for ax in axes[:, 0]:
        ax.set_ylabel("EMI")
    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    fig.tight_layout()
    save(fig, "figT2_machine_translated_by_language")


def fig_t3(eng, mtm):
    """MEP oral speeches per year from 2013: delivered in English vs machine-translated."""
    years = range(2013, int(max(eng.year.max(), mtm.year.max())) + 1)
    a = eng.groupby("year").size().reindex(years, fill_value=0)
    b = mtm.groupby("year").size().reindex(years, fill_value=0)
    fig, ax = plt.subplots(figsize=(F.SINGLE, 55 * F.MM))
    x = np.arange(len(a))
    ax.bar(x - 0.2, a.values, width=0.4, color=F.OCHRE, label="Delivered in English")
    ax.bar(x + 0.2, b.values, width=0.4, color=F.SAGE, edgecolor=F.GREY, linewidth=0.4, label="Machine-translated")
    ax.set_xticks(x, [str(y) for y in years], rotation=90)
    ax.set_ylabel("MEP oral speeches")
    ax.legend(loc="upper left")
    save(fig, "figT3_speeches_per_year")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="score every translated text, not only MEP oral speeches")
    ap.add_argument("--figures-only", action="store_true", help="only tables and figures from the saved file")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        sp = pd.read_parquet(SPEECH_OUT)
    else:
        df = clean_filter_annotate(load_inputs())
        bins, params = paper_scale()
        to_score = df if args.all else df[df["selected"]]
        print(f"\ntesti da valutare: {len(to_score):,} ({'tutti' if args.all else 'solo MEP orali'})")
        sp = speeches(to_score, apply_scale(score(to_score), bins, params))
        sp.to_parquet(SPEECH_OUT, index=False)
        print(f"\nsalvato {SPEECH_OUT.relative_to(cfg.ROOT)}: {len(sp):,} interventi con EMI")
        print(sp[sp.selected].groupby("kind")[["emi", "evidence_z", "intuition_z"]].describe().T.round(3).to_string())

    eng, oral, mtm, incl = analysis_sets(sp)
    yearly_table(eng, mtm, incl)
    trends(eng, mtm, incl)
    section("7. Figure")
    F.set_style()
    fig_t1(eng, oral, incl)
    fig_t2(eng, mtm)
    fig_t3(eng, mtm)
    print(f"\nfatto in {(time.time() - t0) / 60:.1f} minuti")


if __name__ == "__main__":
    main()
