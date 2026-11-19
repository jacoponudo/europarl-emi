"""Figure 1 of the paper. Speeches delivered in English and translated with eTranslation: how many, and their EMI.

Data
  data/raw/unified/unified.parquet          build_unified.py, one row per speech (corpus to 30-06-2021,
                                            Parliament API from 01-07-2021)
  data/interim/etranslation/unified.sqlite  translate_unified.py, read-only; text_id = uid
eTranslation was run on every non-English text of unified from 2013 (prepare --from-year 2013): before
2013 the corpus holds only the English version (the Parliament's human translation), so there is no
original-language text to translate.

Population: MEP oral speeches (category starts with "MEP, oral").
Two groups:
  Delivered in English          english_text == "text (original)"; text = column `text`
  Translated with eTranslation  detected as non-English, all segments translated; text = the segments
                                joined in order (header included, as in the baseline measure)

Panel a  speeches per year in each group (stacked bars). --show-human adds the Parliament's human
         translations (english_text == "text (human translation)", to 2012) as a third series.
Panel b  mean EMI per year, with percentile bootstrap 95% intervals (resampling speeches within year,
         B = 2000, seed 42), for: all speeches (both groups), delivered in English, translated.

EMI, same measure as the paper baseline (src/europarl_emi/emi.py): spaces normalised, speeches with
more than 10 words, chunks of <= 150 words (a last chunk < 50 words joins the previous one), MiniLM
all-MiniLM-L6-v2, cosine with the mean embedding of each lexicon; length correction in 10-word bins and
z-score computed ONCE on all chunks of both groups together; EMI = z(evidence) - z(intuition); speech
EMI = mean of its chunks. Not applied: the procedural filter of step 3.

The scoring (embeddings) is the slow part. It is saved in parts and resumes if interrupted; if the set
of speeches changes (new unified.parquet or new translations), delete the chunk_scores folder.

    uv run python notebooks/paper/fig1.py
    uv run python notebooks/paper/fig1.py --show-human

Out:
  results/figures/paper/fig1.pdf / .png
  data/paper/fig1/
      by_year.csv           panel a: speeches per year in each group
      emi_by_year.csv       panel b: mean EMI, bootstrap interval and n per year and group
      speech_emi.parquet    EMI per speech (uid, year, group, emi, evidence_z, intuition_z, n_chunks)
      chunk_scores/         raw similarities per chunk (cache)
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nhb_style import BAND, DOUBLE, INK, MUTED, PALETTE, apply_style, mm, panel_label, save, ygrid  # noqa: E402

from europarl_emi import config as cfg  # noqa: E402

UNIFIED = cfg.DATA / "raw" / "unified" / "unified.parquet"
ETR_DB = cfg.DATA / "interim" / "etranslation" / "unified.sqlite"
OUT = cfg.DATA / "paper" / "fig1"
CACHE = OUT / "chunk_scores"
FIG = cfg.ROOT / "results" / "figures" / "paper" / "fig1"
PART_SIZE = 20_000
B = 2000

# (key, legend label, colour)
ORIGINAL = ("original", "Delivered in English", PALETTE["Cherry Rose"])
ETRANSLATION = ("etranslation", "Translated with eTranslation", PALETTE["Pacific Cyan"])
HUMAN = ("human", "Human translation (European Parliament)", PALETTE["Dusty Grape"])
ALL = ("all", "All speeches", PALETTE["Dusty Grape"])


# ---------------------------------------------------------------- data

def translations(db: Path) -> pd.DataFrame:
    """English text of the non-English texts whose segments are all translated."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tx = pd.read_sql("SELECT text_id, lang FROM texts WHERE lang IS NOT NULL AND lang != 'EN'", con)
    sg = pd.read_sql("SELECT text_id, idx, status, translation FROM segments", con)
    con.close()
    print(f"{db.name}: {len(tx):,} non-English texts, {len(sg):,} segments "
          f"({sg.status.value_counts().to_dict()})")
    full = sg.groupby("text_id").status.agg(lambda s: s.eq("done").all())
    sg = sg[sg.text_id.isin(full[full].index) & sg.text_id.isin(tx.text_id)].sort_values(["text_id", "idx"])
    out = sg.groupby("text_id").translation.agg(lambda s: " ".join(s.fillna(""))).rename("text_en").reset_index()
    print(f"  fully translated: {len(out):,}; not complete: {tx.text_id.nunique() - len(out):,}")
    return out.rename(columns={"text_id": "uid"})


def load(path: Path, db: Path) -> pd.DataFrame:
    u = pd.read_parquet(path, columns=["uid", "year", "date", "category", "english_text", "text"])
    print(f"{path.name}: {len(u):,} rows, {u.date.min().date()} to {u.date.max().date()}")
    u = u[u.category.astype("string").str.startswith("MEP, oral", na=False)].copy()
    print(f"MEP oral speeches: {len(u):,}")
    tr = translations(db)
    u = u.merge(tr, on="uid", how="left")
    et = u.english_text.astype("string").fillna("none")
    u["group"] = "other"
    u.loc[et.eq("text (original)"), "group"] = "original"
    u.loc[et.eq("text (human translation)"), "group"] = "human"
    u.loc[u.text_en.notna() & ~et.eq("text (original)"), "group"] = "etranslation"
    u["en"] = u.text.where(u.group.eq("original"), u.text_en)
    u["year"] = u.year.astype(int)
    print("groups:", u.group.value_counts().to_dict())
    return u.drop(columns=["text", "text_en"])


def by_year(u: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(u.year.to_numpy(), u.group.to_numpy())
    years = range(int(t.index.min()), int(t.index.max()) + 1)
    t = t.reindex(index=years, columns=["original", "etranslation", "human", "other"], fill_value=0)
    t.index.name, t.columns.name = "year", None
    return t


# ---------------------------------------------------------------- EMI

def chunk_table(u: pd.DataFrame) -> pd.DataFrame:
    from europarl_emi.emi import split_into_chunks

    s = u[u.group.isin(["original", "etranslation"])].sort_values("uid")
    text = s.en.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    keep = text.str.split().str.len() > cfg.SCORE_MIN_WORDS
    rows = [(uid, i, c) for uid, t in zip(s.uid[keep], text[keep]) for i, c in enumerate(split_into_chunks(t))]
    ch = pd.DataFrame(rows, columns=["uid", "chunk_id", "chunk_text"])
    ch["n_words"] = ch.chunk_text.str.split().str.len()
    print(f"speeches to score: {keep.sum():,} (dropped with <= {cfg.SCORE_MIN_WORDS} words: {(~keep).sum():,}); "
          f"chunks: {len(ch):,}")
    return ch


def score(ch: pd.DataFrame) -> pd.DataFrame:
    """Raw similarities per chunk, saved in parts; resumes from the missing parts."""
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = CACHE / "manifest.json"
    sig = {"n_chunks": len(ch), "n_speeches": int(ch.uid.nunique()), "first": ch.uid.iloc[0],
           "last": ch.uid.iloc[-1], "part_size": PART_SIZE, "model": cfg.EMBEDDING_MODEL}
    if manifest.exists() and json.loads(manifest.read_text()) != sig:
        raise SystemExit(f"The speeches differ from those in {CACHE}: delete that folder and run again.")
    manifest.write_text(json.dumps(sig))

    n_parts = math.ceil(len(ch) / PART_SIZE)
    todo = [p for p in range(n_parts) if not (CACHE / f"part-{p:04d}.parquet").exists()]
    if todo:
        from sentence_transformers import SentenceTransformer, util
        from tqdm import tqdm

        from europarl_emi.emi import lexicon_vectors, pick_device

        device = pick_device()
        print(f"scoring {len(todo)} of {n_parts} parts on {device}")
        model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
        ref = lexicon_vectors(model)
        for p in tqdm(todo, desc="parts"):
            part = ch.iloc[p * PART_SIZE:(p + 1) * PART_SIZE].copy()
            emb = model.encode(part.chunk_text.tolist(), batch_size=128, convert_to_tensor=True)
            sims = util.cos_sim(emb, ref).cpu().numpy()
            part["sim_evidence"], part["sim_intuition"] = sims[:, 0], sims[:, 1]
            path = CACHE / f"part-{p:04d}.parquet"
            part.drop(columns="chunk_text").to_parquet(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").rename(path)
    s = pd.concat([pd.read_parquet(CACHE / f"part-{p:04d}.parquet") for p in range(n_parts)], ignore_index=True)
    assert len(s) == len(ch), f"chunks in cache {len(s):,} != expected {len(ch):,}"
    return s


def speech_emi(u: pd.DataFrame) -> pd.DataFrame:
    from europarl_emi.emi import compute_emi

    ch = chunk_table(u)
    s = compute_emi(score(ch))  # length correction and z-score on all chunks of both groups together
    sp = s.groupby("uid").agg(emi=("emi", "mean"), evidence_z=("evidence_z", "mean"),
                              intuition_z=("intuition_z", "mean"), n_chunks=("chunk_id", "size")).reset_index()
    sp = sp.merge(u[["uid", "year", "group"]], on="uid", how="left")
    print(f"speeches with EMI: {len(sp):,}; SD of speech EMI: {sp.emi.std():.3f}")
    return sp


def boot_mean(x: np.ndarray, rng: np.random.Generator, b: int = B) -> tuple[float, float]:
    means = np.empty(b)
    step = max(1, 2_000_000 // max(len(x), 1))  # resamples per batch, to keep memory low
    for i in range(0, b, step):
        k = min(step, b - i)
        means[i:i + k] = x[rng.integers(0, len(x), size=(k, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def emi_by_year(sp: pd.DataFrame, b: int) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for key, sel in (("all", sp), ("original", sp[sp.group.eq("original")]),
                     ("etranslation", sp[sp.group.eq("etranslation")])):
        for year, x in sel.groupby("year").emi:
            x = x.to_numpy()
            lo, hi = boot_mean(x, rng, b) if len(x) > 1 else (np.nan, np.nan)
            rows.append((key, int(year), len(x), x.mean(), lo, hi))
    return pd.DataFrame(rows, columns=["group", "year", "n", "mean", "ci_low", "ci_high"])


# ---------------------------------------------------------------- figure

def coverage_band(ax, label: bool) -> None:
    # corpus incomplete from 2020 until the API takes over (July 2021)
    ax.axvspan(2019.5, 2021.5, color=BAND, zorder=0, linewidth=0)
    if label:
        ax.text(2020.5, 1.01, "Incomplete coverage", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.5, color=MUTED, clip_on=False)


def panel_a(ax, t: pd.DataFrame, show_human: bool) -> None:
    x = t.index.to_numpy()
    bottom = 0
    for key, label, color in [ORIGINAL, ETRANSLATION] + ([HUMAN] if show_human else []):
        ax.bar(x, t[key], bottom=bottom, width=0.8, color=color, label=label, edgecolor="white", linewidth=0.4)
        bottom = bottom + t[key]
    coverage_band(ax, label=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    ax.set_ylabel("Speeches per year")
    ygrid(ax)
    ax.legend(loc="upper left", handlelength=1.0, handleheight=0.8, borderaxespad=0.2, labelspacing=0.3)


def panel_b(ax, e: pd.DataFrame) -> None:
    coverage_band(ax, label=False)
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    # all speeches: dashed line on top, because before 2013 it coincides with "delivered in English"
    for key, label, color in (ORIGINAL, ETRANSLATION, ALL):
        d = e[e.group.eq(key)].sort_values("year")
        dashed = key == "all"
        ax.fill_between(d.year, d.ci_low, d.ci_high, color=color, alpha=0.12 if dashed else 0.2,
                        linewidth=0, zorder=2)
        ax.plot(d.year, d["mean"], color=color, label=label, zorder=4 if dashed else 3,
                linewidth=0.9 if dashed else 1.0, linestyle=(0, (3, 1.5)) if dashed else "-",
                marker=None if dashed else "o", markersize=2.6, markeredgecolor="white", markeredgewidth=0.3)
    ax.set_ylabel("Mean EMI")
    ax.set_xlabel("Year")
    ygrid(ax)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)


def figure(t: pd.DataFrame, e: pd.DataFrame, show_human: bool) -> None:
    apply_style()
    fig, (a, b) = plt.subplots(2, 1, figsize=(mm(DOUBLE), mm(130)), sharex=True,
                               gridspec_kw={"hspace": 0.18})
    panel_a(a, t, show_human)
    panel_b(b, e)
    x = t.index.to_numpy()
    b.set_xlim(x.min() - 0.7, x.max() + 0.7)
    b.xaxis.set_major_locator(MultipleLocator(4))
    b.xaxis.set_minor_locator(MultipleLocator(1))
    a.tick_params(labelbottom=False)
    fig.align_ylabels([a, b])
    panel_label(a, "a", x=-0.055)
    panel_label(b, "b", x=-0.055)
    save(fig, FIG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=UNIFIED)
    ap.add_argument("--db", type=Path, default=ETR_DB)
    ap.add_argument("--show-human", action="store_true", help="panel a: add the Parliament's human translations")
    ap.add_argument("--bootstrap", type=int, default=B, help=f"bootstrap resamples (default {B})")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    u = load(args.input, args.db)
    t = by_year(u)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "by_year.csv")
    print("\nPanel a, speeches per year (other = no English text, not in the figure):")
    print(t.to_string())

    sp = speech_emi(u)
    sp.to_parquet(OUT / "speech_emi.parquet", index=False)
    e = emi_by_year(sp, args.bootstrap)
    e.to_csv(OUT / "emi_by_year.csv", index=False)
    print(f"\nPanel b, mean EMI per year (percentile bootstrap 95%, B = {args.bootstrap}):")
    print(e.pivot(index="year", columns="group", values=["mean", "n"]).round(3).to_string())

    figure(t, e, args.show_human)
    print(f"tables in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
