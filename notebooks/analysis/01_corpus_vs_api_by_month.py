"""Analisi 1. Quanti interventi abbiamo per mese dal corpus originale e quanti dall'API del Parlamento,
e quanti sono lo stesso intervento presente in entrambe le fonti (duplicati).

Fonti (fuori da git)
  corpus  data/raw/corpus/all_years_combined.csv.gzip   una riga per intervento, tutte le lingue
  API     data/raw/api/speeches/YYYY-MM-DD.jsonl         una riga per intervento (download_speeches.py)
Si contano tutti gli interventi con un testo non vuoto, di qualsiasi tipo e lingua.

Duplicato: un intervento del corpus e uno dell'API il cui testo e uguale per piu del 50%.
  Testo normalizzato: html.unescape, senza tag, NFKC, minuscolo, solo parole. Per l'API si usa
  intestazione + testo, perche nel corpus l'intestazione ("Nome (GRUPPO). - ...") fa parte del testo.
  Somiglianza = sequenze di 5 parole in comune / sequenze di 5 parole del testo PIU LUNGO
  (quindi "piu del 50%" vuol dire che piu di meta del testo piu lungo compare anche nell'altro).
  Testi sotto le 5 parole: duplicati solo se identici.
  Si cercano i duplicati nella finestra in cui ci sono entrambe le fonti, con qualsiasi data
  (le date del corpus potrebbero non essere i giorni di seduta), e l'abbinamento e uno a uno:
  ogni intervento conta al massimo una volta.
  Ogni intervento va nel mese della sua data; un duplicato va nel mese della seduta (data API).

    uv run python notebooks/analysis/01_corpus_vs_api_by_month.py
    uv run python notebooks/analysis/01_corpus_vs_api_by_month.py --threshold 0.6

Esce:
  results/figures/analysis/01_corpus_vs_api_by_month.pdf / .png
  data/analysis/01_corpus_vs_api/monthly.csv     per mese: solo corpus, entrambi, solo API, totali
  data/analysis/01_corpus_vs_api/pairs.parquet   coppie abbinate: guid, speech_id, somiglianza, giorni di distanza
  data/analysis/01_corpus_vs_api/best_match.parquet   per ogni intervento della finestra: miglior somiglianza
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
import zlib
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from europarl_emi import config as cfg  # noqa: E402

CORPUS = cfg.RAW_CORPUS / "all_years_combined.csv.gzip"
API_DIR = cfg.DATA / "raw" / "api" / "speeches"
OUT = cfg.DATA / "analysis" / "01_corpus_vs_api"
FIG = cfg.ROOT / "results" / "figures" / "analysis" / "01_corpus_vs_api_by_month"

K = 5               # parole per sequenza
SAMPLE_MOD = 4      # per cercare i candidati si indicizza 1 sequenza su 4 (scelta dal valore hash, stabile)
MAX_DF = 200        # sequenze presenti in piu di 200 testi (formule ricorrenti) non servono a trovare candidati
TOP = 5             # candidati controllati per ogni intervento API

# colori (validati per daltonismo con lo strumento della skill dataviz): corpus, entrambi, API
C_CORPUS, C_BOTH, C_API = "#a40606", "#c8741a", "#2f6db3"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
LABELS = {"corpus": "solo corpus", "both": "in entrambi (duplicati)", "api": "solo API"}

TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"\w+")


def norm_words(text):
    t = html.unescape(TAG.sub(" ", str(text or "")))
    return WORD.findall(unicodedata.normalize("NFKC", t).lower())


def shingles(words):
    return {zlib.crc32(" ".join(words[i:i + K]).encode()) for i in range(len(words) - K + 1)}


def similarity(wa, wb):
    if len(wa) < K or len(wb) < K:
        return 1.0 if wa == wb and wa else 0.0
    a, b = shingles(wa), shingles(wb)
    return len(a & b) / max(len(a), len(b))


# ---------------------------------------------------------------- lettura

def read_corpus():
    t0 = time.time()
    df = pd.read_csv(CORPUS, compression="gzip", usecols=["guid", "language", "pubdate", "description"], dtype=str)
    df["date"] = pd.to_datetime(df["pubdate"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
    df["language"] = df["language"].str.strip().str.lower()
    n0 = len(df)
    df = df[df["description"].fillna("").str.strip().ne("") & df["date"].notna()]
    print(f"corpus: {n0:,} righe, {len(df):,} con testo e data, {df.date.min():%Y-%m-%d} .. {df.date.max():%Y-%m-%d}"
          f"  ({time.time() - t0:.0f} s)", flush=True)
    return df.rename(columns={"description": "text"})[["guid", "language", "date", "text"]]


def read_api():
    t0 = time.time()
    rows, no_text = [], 0
    for f in sorted(API_DIR.glob("*.jsonl")):
        if f.stat().st_size == 0:
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if not (r.get("text") or "").strip():
                    no_text += 1
                    continue
                rows.append((r["speech_id"], r["date"], r.get("activity_type"), r.get("role"),
                             r.get("original_language"), f"{r.get('header') or ''} {r['text']}"))
    df = pd.DataFrame(rows, columns=["speech_id", "date", "activity_type", "role", "original_language", "text"])
    df["date"] = pd.to_datetime(df["date"])
    print(f"API: {len(df):,} interventi con testo ({no_text:,} senza testo esclusi), "
          f"{df.date.min():%Y-%m-%d} .. {df.date.max():%Y-%m-%d}  ({time.time() - t0:.0f} s)", flush=True)
    return df


# ---------------------------------------------------------------- abbinamento

def match(corpus, api, threshold):
    """Coppie uno a uno corpus-API con somiglianza > threshold, nella finestra in cui ci sono entrambe."""
    start = max(corpus.date.min(), api.date.min())
    end = min(corpus.date.max(), api.date.max())
    # margine di 31 giorni ai bordi, per le date del corpus che non sono giorni di seduta
    c = corpus[corpus.date.between(start - pd.Timedelta(days=31), end + pd.Timedelta(days=31))].reset_index(drop=True)
    a = api[api.date.between(start - pd.Timedelta(days=31), end + pd.Timedelta(days=31))].reset_index(drop=True)
    print(f"\nfinestra comune: {start:%Y-%m-%d} .. {end:%Y-%m-%d}  "
          f"(corpus {len(c):,}, API {len(a):,} interventi, con 31 giorni di margine)", flush=True)
    t0 = time.time()
    cw = [norm_words(x) for x in c.text]
    aw = [norm_words(x) for x in a.text]

    # indice delle sequenze campionate del corpus
    index = defaultdict(list)
    for i, w in enumerate(cw):
        for h in shingles(w):
            if h % SAMPLE_MOD == 0:
                index[h].append(i)
    index = {h: ids for h, ids in index.items() if len(ids) <= MAX_DF}
    exact = defaultdict(list)  # testi corti: confronto esatto
    for i, w in enumerate(cw):
        if 0 < len(w) < K:
            exact[" ".join(w)].append(i)
    print(f"  indice: {len(index):,} sequenze  ({time.time() - t0:.0f} s)", flush=True)

    cand = []
    best_api = [0.0] * len(aw)
    best_cor = [0.0] * len(cw)
    for j, w in enumerate(aw):
        if len(w) < K:
            for i in exact.get(" ".join(w), []):
                cand.append((1.0, i, j))
                best_api[j] = best_cor[i] = 1.0
            continue
        hits = Counter()
        for h in shingles(w):
            if h % SAMPLE_MOD == 0 and h in index:
                hits.update(index[h])
        for i, _ in hits.most_common(TOP):
            s = similarity(cw[i], w)
            best_api[j] = max(best_api[j], s)
            best_cor[i] = max(best_cor[i], s)
            if s > threshold:
                cand.append((s, i, j))
        if j and j % 10_000 == 0:
            print(f"  {j:,} / {len(aw):,} interventi API  ({time.time() - t0:.0f} s)", flush=True)

    # uno a uno: prima le coppie piu simili
    used_c, used_a, pairs = set(), set(), []
    for s, i, j in sorted(cand, reverse=True):
        if i not in used_c and j not in used_a:
            used_c.add(i)
            used_a.add(j)
            pairs.append((c.guid[i], a.speech_id[j], s, (c.date[i] - a.date[j]).days))
    pairs = pd.DataFrame(pairs, columns=["guid", "speech_id", "similarity", "days_corpus_minus_api"])
    best = pd.concat([pd.DataFrame({"source": "api", "id": a.speech_id, "date": a.date, "best": best_api}),
                      pd.DataFrame({"source": "corpus", "id": c.guid, "date": c.date, "best": best_cor})])
    print(f"  coppie oltre la soglia {threshold:.0%}: {len(pairs):,}  ({time.time() - t0:.0f} s)", flush=True)
    return pairs, best, (start, end)


# ---------------------------------------------------------------- tabelle

def monthly(corpus, api, pairs):
    dup_c, dup_a = set(pairs.guid), set(pairs.speech_id)
    api_m = api.assign(month=api.date.dt.to_period("M"))
    cor_m = corpus.assign(month=corpus.date.dt.to_period("M"))
    t = pd.DataFrame({
        "corpus": cor_m[~cor_m.guid.isin(dup_c)].groupby("month").size(),
        "both": api_m[api_m.speech_id.isin(dup_a)].groupby("month").size(),
        "api": api_m[~api_m.speech_id.isin(dup_a)].groupby("month").size(),
        "corpus_total": cor_m.groupby("month").size(),
        "api_total": api_m.groupby("month").size(),
    }).fillna(0).astype(int)
    full = pd.period_range(t.index.min(), t.index.max(), freq="M")
    return t.reindex(full, fill_value=0).rename_axis("month")


def report(t, pairs, best, corpus, api, window, threshold):
    start, end = window
    w = t.loc[start.to_period("M"):end.to_period("M")]
    print("\n" + "=" * 78)
    print("TOTALI")
    print("=" * 78)
    print(f"  corpus: {t.corpus_total.sum():,}   API: {t.api_total.sum():,}   duplicati: {len(pairs):,}   "
          f"interventi distinti: {t[['corpus', 'both', 'api']].sum().sum():,}")
    print(f"\nFINESTRA COMUNE {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    ct, at, both = w.corpus_total.sum(), w.api_total.sum(), w.both.sum()
    print(f"  corpus {ct:,}, API {at:,}, in entrambi {both:,}: "
          f"{both / ct:.1%} del corpus e {both / at:.1%} dell'API sono duplicati")
    print("\n  per mese:")
    ww = w.assign(**{"% corpus duplicato": (w.both / w.corpus_total.where(w.corpus_total > 0)).round(3),
                     "% API duplicata": (w.both / w.api_total.where(w.api_total > 0)).round(3)})
    print(ww.to_string())

    if len(pairs):
        print("\n  somiglianza delle coppie: " + ", ".join(
            f"{q:.0%} {v:.2f}" for q, v in pairs.similarity.quantile([0.01, 0.1, 0.5]).items()))
        d = pairs.days_corpus_minus_api.value_counts().head(6)
        print("  giorni tra data del corpus e seduta (corpus - API), i piu frequenti: "
              + ", ".join(f"{k:+d}: {v:,}" for k, v in d.items()))
    bw = best[best.date.between(start, end)]
    bins = [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    print("\n  miglior somiglianza trovata per ogni intervento della finestra (quota per fascia):")
    for src, g in bw.groupby("source"):
        h = pd.cut(g.best, bins).value_counts(normalize=True, sort=False)
        print(f"    {src:7}" + "  ".join(f"{str(k):>12} {v:5.1%}" for k, v in h.items()))

    aw = api[api.date.between(start, end)].assign(dup=lambda x: x.speech_id.isin(set(pairs.speech_id)))
    print("\n  API nella finestra, per tipo di intervento: quota con un duplicato nel corpus")
    print(aw.groupby(["activity_type", "role"], dropna=False).dup.agg(["size", "mean"])
          .rename(columns={"size": "interventi", "mean": "% duplicati"}).round(3).to_string())
    cw = corpus[corpus.date.between(start, end)].assign(dup=lambda x: x.guid.isin(set(pairs.guid)))
    print("\n  corpus nella finestra, per lingua: quota con un duplicato nell'API")
    print(cw.groupby("language").dup.agg(["size", "mean"]).rename(columns={"size": "interventi", "mean": "% duplicati"})
          .sort_values("interventi", ascending=False).head(12).round(3).to_string())


# ---------------------------------------------------------------- figura

def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.frameon": False,
    })


def stacked(ax, t, width):
    x = t.index.to_timestamp() + pd.Timedelta(days=15)
    bottom = 0
    for key, color in (("corpus", C_CORPUS), ("both", C_BOTH), ("api", C_API)):
        ax.bar(x, t[key], width=width, bottom=bottom, color=color, label=LABELS[key], edgecolor="white", linewidth=0.2)
        bottom = bottom + t[key]
    ax.set_ylabel("interventi al mese")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def figure(t, best, window, threshold):
    style()
    start, end = window
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.8], hspace=0.55, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    stacked(ax, t, width=25)
    ax.axvspan(start, end, color="#f3f1b5", zorder=0, linewidth=0)
    ax.set_title("a  Interventi per mese e fonte (in giallo la finestra con entrambe le fonti)",
                 loc="left", color=INK, fontsize=7, fontweight="bold", pad=22)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, borderaxespad=0, handlelength=1.2)

    w = t.loc[(start - pd.Timedelta(days=31)).to_period("M"):(end + pd.Timedelta(days=31)).to_period("M")]
    ax = fig.add_subplot(gs[1, :])
    stacked(ax, w, width=22)
    ax.set_title("b  Finestra comune, ingrandita", loc="left", color=INK, fontsize=7, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    ax = fig.add_subplot(gs[2, 0])
    ww = t.loc[start.to_period("M"):end.to_period("M")]
    x = ww.index.to_timestamp() + pd.Timedelta(days=15)
    pc = (ww.both / ww.corpus_total.where(ww.corpus_total > 0)) * 100
    pa = (ww.both / ww.api_total.where(ww.api_total > 0)) * 100
    ax.plot(x, pc, color=C_CORPUS, linewidth=1.2, marker="o", markersize=2.5, label="% del corpus che è anche nell'API")
    ax.plot(x, pa, color=C_API, linewidth=1.2, marker="o", markersize=2.5, label="% dell'API che è anche nel corpus")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% duplicati")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("c  Quota di duplicati per mese", loc="left", color=INK, fontsize=7, fontweight="bold")

    ax = fig.add_subplot(gs[2, 1])
    bw = best[best.date.between(start, end)]
    bins = [i / 20 for i in range(21)]
    for src, color, lab in (("corpus", C_CORPUS, "interventi del corpus"), ("api", C_API, "interventi dell'API")):
        v = bw.loc[bw.source == src, "best"]
        ax.hist(v, bins=bins, weights=[100 / max(len(v), 1)] * len(v), histtype="step", linewidth=1.2,
                color=color, label=lab)
    ax.axvline(threshold, color=INK, linewidth=0.7, linestyle="--")
    ax.text(threshold, ax.get_ylim()[1] * 0.02, f" soglia {threshold:.0%}", color=INK, fontsize=6, va="bottom")
    ax.set_xlabel("miglior somiglianza con l'altra fonte")
    ax.set_ylabel("% degli interventi")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.52, 0.85), fontsize=6)
    ax.set_title("d  Quanto è simile il testo più vicino", loc="left", color=INK, fontsize=7, fontweight="bold")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nfigura: {FIG.relative_to(cfg.ROOT)}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.5, help="somiglianza minima per un duplicato (default 0.5)")
    args = ap.parse_args()
    t0 = time.time()
    corpus, api = read_corpus(), read_api()
    pairs, best, window = match(corpus, api, args.threshold)
    t = monthly(corpus, api, pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "monthly.csv")
    pairs.to_parquet(OUT / "pairs.parquet", index=False)
    best.to_parquet(OUT / "best_match.parquet", index=False)
    report(t, pairs, best, corpus, api, window, args.threshold)
    figure(t, best, window, args.threshold)
    print(f"tabelle in {OUT.relative_to(cfg.ROOT)}/   fatto in {(time.time() - t0) / 60:.1f} minuti")


if __name__ == "__main__":
    main()
