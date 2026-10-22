"""Paper figures (Scientific Reports style).

Run all:        uv run python -m europarl_emi.figures
Run some:       uv run python -m europarl_emi.figures --only fig2 fig5

Requires steps 1-8. Output: results/figures/paper/<name>.pdf and .png
(PDF with editable text for submission, PNG at 600 dpi for viewing).

Main figures
  fig1    Corpus composition by year
  fig2    EMI by year, main sample
  fig3    Composition vs within-speaker change
  fig4    EMI by political family and parliamentary term
  fig5    Robustness: change in EMI per decade across specifications
Supplementary
  figS1   Processing pipeline with counts
  figS2   Manual validation of the procedural filter
  tableS3 Most evidence-like / intuition-like chunks + lexicons (tables, not a figure)
  figS4   Evidence and intuition components
  figS5   Original English vs translated speeches, 1996-2012
  figS6   United Kingdom, Ireland and other member states
  figS7   EMI distribution by parliamentary term
  figS8   Speeches per year in the main sample
  figS9   Old (batch-standardised) vs new (globally standardised) EMI
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

from europarl_emi import config as cfg
from europarl_emi.step08_models import ANNOTATED, CHUNKS, SPEECHES, TABLES, load_main

OUT = cfg.FIGURES / "paper"

# ---------------------------------------------------------------- style

MM = 1 / 25.4
SINGLE, ONE_HALF, DOUBLE = 89 * MM, 120 * MM, 183 * MM

MINT, SAGE, GREY, OCHRE, PEACH = "#e2efde", "#afd0bf", "#808f87", "#9b7e46", "#f4b266"
INK, MUTED, DARK = "#1a1a1a", "#5a5a5a", "#3a3a3a"

# Only OCHRE and GREY are dark enough for lines on white; light colours are fills only.
# Series are also told apart by line style and marker, and labelled directly.
LINES = [
    {"color": OCHRE, "ls": "-", "marker": "o"},
    {"color": GREY, "ls": "--", "marker": "s"},
    {"color": DARK, "ls": ":", "marker": "^"},
]
FAMILY_ORDER = ["Left", "Greens/EFA", "S&D", "Liberals", "EPP", "Conservatives", "Eurosceptics", "Non-attached"]
TERMS_MAIN = ["4th", "5th", "6th", "7th", "8th"]
TERM_BOUNDARIES = [1999.5, 2004.5, 2009.5, 2014.5, 2019.5]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.0, "lines.markersize": 2.5,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def panel_label(ax, letter):
    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def zero_line(ax, axis="h"):
    (ax.axhline if axis == "h" else ax.axvline)(0, color=MUTED, linewidth=0.4, zorder=0)


def yearly(df, col, min_n=1):
    g = df.groupby("year")[col].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    g["lo"] = g["mean"] - 1.96 * g["se"]
    g["hi"] = g["mean"] + 1.96 * g["se"]
    g = g[g["count"] >= min_n]
    g.index = g.index.astype(int)
    return g


def plot_series(ax, s, style, label, band=True):
    if band:
        alpha = 0.07 if style["color"] == DARK else 0.15
        ax.fill_between(s.index, s["lo"], s["hi"], color=style["color"], alpha=alpha, linewidth=0)
    ax.plot(s.index, s["mean"], color=style["color"], ls=style["ls"], marker=style["marker"],
            markersize=2.5, linewidth=1, label=label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def direct_labels(ax, items, dx=0.4):
    """items: list of (label, x, y). Labels in ink colour, spread apart to avoid overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    placed = []
    for label, x, y in sorted(items, key=lambda t: t[2]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.text(x + dx, y, label, va="center", ha="left", fontsize=6, color=MUTED)


def nrows(path):
    return pq.ParquetFile(path).metadata.num_rows


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(cfg.ROOT)} not found: run step 8 first.")


# ---------------------------------------------------------------- main figures

def fig1():
    """Corpus composition by year: counts (a) and shares (b)."""
    df = pd.read_parquet(ANNOTATED, columns=["year", "role", "written", "translated", "header_found"])
    df = df.reset_index(drop=True)
    cat = pd.Series("Unrecognised header", index=df.index)
    cat[df["role"].isin(["chair", "commission", "council"])] = "Chair, Commission or Council"
    mep = (df["role"] == "mep") & df["header_found"]
    cat[mep & df["written"]] = "MEP, written statement"
    cat[mep & ~df["written"] & df["translated"]] = "MEP, oral, translated"
    cat[mep & ~df["written"] & ~df["translated"]] = "MEP, oral, original English (main sample)"
    order = ["MEP, oral, original English (main sample)", "MEP, oral, translated", "MEP, written statement",
             "Chair, Commission or Council", "Unrecognised header"]
    fills = [OCHRE, PEACH, SAGE, GREY, MINT]
    counts = pd.crosstab(df["year"].astype(int), cat).reindex(columns=order, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 105 * MM), sharex=True)
    for ax, data, ylabel, letter in ((axes[0], counts / 1000, "Speeches (thousands)", "a"),
                                     (axes[1], shares, "Share of speeches (%)", "b")):
        bottom = np.zeros(len(data))
        for col, color in zip(order, fills):
            ax.bar(data.index, data[col], bottom=bottom, color=color, width=0.8,
                   edgecolor="white", linewidth=0.3, label=col)
            bottom += data[col].to_numpy()
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.axvline(2012.5, color=INK, linewidth=0.6, ls="--")
    axes[1].text(2012.7, 102, "End of full translation\nof verbatim reports", fontsize=5.5, color=MUTED, va="bottom")
    axes[1].set_xlabel("Year")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, fontsize=6, columnspacing=1.2)
    save(fig, "fig1")


def fig2():
    """EMI by year in the main sample; years after 2019 shown as low coverage."""
    s = yearly(load_main(period=None), "emi")
    core, late = s[s.index <= 2019], s[s.index >= 2019]

    fig, ax = plt.subplots(figsize=(ONE_HALF, 65 * MM))
    zero_line(ax)
    for b in TERM_BOUNDARIES:
        ax.axvline(b, color="#d0d0d0", linewidth=0.4, zorder=0)
    ax.fill_between(core.index, core["lo"], core["hi"], color=PEACH, alpha=0.55, linewidth=0)
    ax.plot(core.index, core["mean"], color=OCHRE, marker="o", linewidth=1)
    ax.fill_between(late.index, late["lo"], late["hi"], color=GREY, alpha=0.18, linewidth=0)
    ax.plot(late.index, late["mean"], color=GREY, ls="--", marker="o", markerfacecolor="white", linewidth=0.8)

    ymin, ymax = ax.get_ylim()
    for x, label in zip([1997.75, 2002, 2007, 2012, 2017, 2021.5], ["4th", "5th", "6th", "7th", "8th", "9th"]):
        ax.text(x, 1.01, f"{label} term", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color=MUTED)
    ax.text(2004.7, ymin + 0.03 * (ymax - ymin), "2004 enlargement", fontsize=5.5, color=MUTED)
    if len(late) > 1:
        ax.text(late.index.to_numpy().mean(), late["hi"].max() + 0.02 * (ymax - ymin), "Low coverage",
                fontsize=5.5, color=MUTED, ha="center", va="bottom")
    ax.set_ylabel("EMI (evidence minus intuition)")
    ax.set_xlabel("Year")
    save(fig, "fig2")


def fig3():
    """Change since 1996-1998: all speeches, excluding UK MEPs, within speaker."""
    require(TABLES / "year_effects_speaker_fe.csv")
    df = load_main()

    def relative(sub):
        s = yearly(sub, "emi")
        base = sub.loc[sub["year"] <= 1998, "emi"].mean()
        s[["mean", "lo", "hi"]] -= base
        return s

    fe = pd.read_csv(TABLES / "year_effects_speaker_fe.csv").set_index("year")
    fe["mean"] = fe["coef"] - fe.loc[fe.index <= 1998, "coef"].mean()
    fe["lo"], fe["hi"] = fe["mean"] - 1.96 * fe["se"], fe["mean"] + 1.96 * fe["se"]

    series = [
        ("All speeches", relative(df)),
        ("Excluding UK MEPs", relative(df[df["country_group"] != "United Kingdom"])),
        ("Within speaker", fe),
    ]
    fig, ax = plt.subplots(figsize=(ONE_HALF, 65 * MM))
    zero_line(ax)
    for (label, s), style in zip(series, LINES):
        plot_series(ax, s, style, label)
    ax.set_xlim(1995.5, 2023.5)
    direct_labels(ax, [(label, s.index[-1], s["mean"].iloc[-1]) for label, s in series])
    ax.legend(loc="lower left")
    ax.set_ylabel("Change in EMI since 1996–1998")
    ax.set_xlabel("Year")
    save(fig, "fig3")


def fig4():
    """EMI by political family and parliamentary term (4th-8th), shared y-axis."""
    df = load_main()
    df = df[df["term"].isin(TERMS_MAIN)]
    g = df.groupby(["family", "term"], observed=True)["emi"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    g = g[g["count"] >= 50]

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 85 * MM), sharex=True, sharey=True)
    for ax, family in zip(axes.flat, FAMILY_ORDER):
        sub = g[g["family"] == family].set_index("term").reindex(TERMS_MAIN)
        x = np.arange(len(TERMS_MAIN))
        zero_line(ax)
        ax.errorbar(x, sub["mean"], yerr=1.96 * sub["se"], color=OCHRE, marker="o", markersize=3,
                    linewidth=0.9, elinewidth=0.6, capsize=0)
        ax.set_title(family, loc="left", fontsize=7)
        ax.set_xticks(x, TERMS_MAIN)
        if sub["mean"].isna().all():
            ax.text(0.5, 0.5, "Too few speeches", transform=ax.transAxes, ha="center", color=MUTED, fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("EMI")
    fig.supxlabel("Parliamentary term", fontsize=7)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, "fig4")


def fig5():
    """Coefficient plot: change in EMI per decade across specifications."""
    require(TABLES / "trend_models.csv")
    m = pd.read_csv(TABLES / "trend_models.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(SINGLE, 62 * MM))
    zero_line(ax, axis="v")
    y = np.arange(len(m))
    ax.errorbar(m["coef"], y, xerr=[m["coef"] - m["ci_low"], m["ci_high"] - m["coef"]], fmt="o",
                color=OCHRE, markersize=3.5, elinewidth=0.9, capsize=0)
    ax.set_yticks(y, m["spec"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Change in EMI per decade (95% CI)")
    save(fig, "fig5")


# ---------------------------------------------------------------- supplementary

def figS1():
    """Processing pipeline with the number of texts at each stage."""
    raw = len(pd.read_csv(cfg.RAW_CORPUS / "all_years_combined.csv.gzip", compression="gzip", usecols=["language"]))
    stages = [
        ("Raw corpus, all languages", raw),
        ("English-language speeches", nrows(cfg.INTERIM / "01_speeches_en.parquet")),
        ("Very short texts removed", nrows(cfg.INTERIM / "02_speeches_clean.parquet")),
        ("Procedural texts removed", nrows(cfg.INTERIM / "03_speeches_substantive.parquet")),
        ("Scored speeches", nrows(cfg.PROCESSED / "05_speech_emi.parquet")),
        ("Main sample: oral speeches by MEPs,\noriginally delivered in English", nrows(cfg.PROCESSED / "07_speech_main_sample.parquet")),
    ]
    notes = ["language = English", "≥ 3 words and ≥ 16 characters", "NB-LR classifier, texts < 400 characters",
             "> 10 words; ≤ 150-word chunks; MiniLM similarity", "speech header: role, original language, written"]

    fig, ax = plt.subplots(figsize=(ONE_HALF, 110 * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(stages))
    ax.axis("off")
    for i, (label, n) in enumerate(stages):
        yc = len(stages) - i - 0.5
        fill = SAGE if i == len(stages) - 1 else MINT
        ax.add_patch(FancyBboxPatch((0.03, yc - 0.33), 0.5, 0.66, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor=fill, edgecolor=GREY, linewidth=0.5))
        ax.text(0.28, yc + (0.14 if "\n" in label else 0.1), label, ha="center", va="center", fontsize=6.5, linespacing=1.1)
        ax.text(0.28, yc - 0.2, f"n = {n:,}", ha="center", va="center", fontsize=6.5, fontweight="bold")
        if i < len(notes):
            ax.annotate("", xy=(0.28, yc - 0.67), xytext=(0.28, yc - 0.33),
                        arrowprops={"arrowstyle": "-|>", "color": GREY, "linewidth": 0.6})
            ax.text(0.56, yc - 0.5, notes[i], va="center", fontsize=6, color=MUTED, style="italic")
    save(fig, "figS1")


def parse_manual(value):
    v = str(value).strip().lower()
    if v in {"1", "yes", "y", "si", "sì", "true", "procedural", "p"}:
        return True
    if v in {"0", "no", "n", "false", "substantive", "s"}:
        return False
    return np.nan


def figS2():
    """Manual validation of the procedural filter (stratified sample of 200 short texts)."""
    sample = pd.read_csv(cfg.ROOT / "results" / "procedural_check_sample.csv")
    manual = sample["controllo_manuale"].map(parse_manual)
    fig, ax = plt.subplots(figsize=(SINGLE, 70 * MM))
    if manual.notna().sum() == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Manual labels not yet available.\nFill column 'controllo_manuale' in\n"
                "results/procedural_check_sample.csv\n(1 = procedural, 0 = substantive).",
                ha="center", va="center", fontsize=7, color=MUTED)
        save(fig, "figS2")
        return
    ok = manual.notna()
    truth, pred = manual[ok].astype(bool), sample.loc[ok, "procedural"].astype(bool)
    cm = pd.crosstab(truth, pred).reindex(index=[False, True], columns=[False, True], fill_value=0)
    cmap = LinearSegmentedColormap.from_list("palette", [MINT, SAGE, OCHRE])
    ax.imshow(cm.to_numpy(), cmap=cmap)
    for (i, j), v in np.ndenumerate(cm.to_numpy()):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, fontweight="bold")
    labels = ["Substantive", "Procedural"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Classifier prediction")
    ax.set_ylabel("Manual label")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    tp, fp, fn = cm.loc[True, True], cm.loc[False, True], cm.loc[True, False]
    acc = (cm.loc[True, True] + cm.loc[False, False]) / cm.to_numpy().sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    ax.text(0.5, -0.32, f"n = {ok.sum()}   accuracy = {acc:.2f}   precision = {prec:.2f}   recall = {rec:.2f}",
            transform=ax.transAxes, ha="center", fontsize=6, color=MUTED)
    save(fig, "figS2")


def tableS3():
    """Most evidence-like and intuition-like chunks in the main sample, plus the two lexicons."""
    from europarl_emi.step04_score_chunks import split_into_chunks

    tables = cfg.ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    main = load_main()
    chunks = pd.read_parquet(CHUNKS, columns=["guid", "chunk_id", "n_words", "emi"])
    chunks = chunks[chunks["guid"].isin(set(main["guid"])) & (chunks["n_words"] >= 100)]
    picks = pd.concat([
        chunks.nlargest(10, "emi").assign(end="Most evidence-like"),
        chunks.nsmallest(10, "emi").assign(end="Most intuition-like"),
    ])
    texts = pd.read_parquet(SPEECHES, columns=["guid", "text"])
    texts = texts[texts["guid"].isin(set(picks["guid"]))].set_index("guid")["text"]
    picks["excerpt"] = [
        split_into_chunks(texts[g])[int(c)][:400] + "…" for g, c in zip(picks["guid"], picks["chunk_id"])
    ]
    picks = picks.merge(main[["guid", "year", "family", "country"]], on="guid", how="left")
    out = picks[["end", "emi", "year", "family", "country", "excerpt"]].round({"emi": 2})
    out.to_csv(tables / "S3_examples.csv", index=False)

    lex = {name: pd.read_csv(cfg.LEXICONS / f"{name}_lexicon.csv")[f"{name}_keywords"].dropna().tolist()
           for name in ("evidence", "intuition")}
    width = max(len(v) for v in lex.values())
    pd.DataFrame({k: v + [""] * (width - len(v)) for k, v in lex.items()}).to_csv(tables / "S3_lexicons.csv", index=False)
    print("  saved results/tables/S3_examples.csv and S3_lexicons.csv")


def figS4():
    """Evidence and intuition components (z-scores) by year."""
    df = load_main()
    series = [("Evidence", yearly(df, "evidence_z")), ("Intuition", yearly(df, "intuition_z"))]
    fig, ax = plt.subplots(figsize=(ONE_HALF, 62 * MM))
    zero_line(ax)
    for (label, s), style in zip(series, LINES):
        plot_series(ax, s, style, label)
    ax.set_xlim(1995.5, 2022)
    direct_labels(ax, [(label, s.index[-1], s["mean"].iloc[-1]) for label, s in series])
    ax.legend(loc="lower left")
    ax.set_ylabel("Mean component score (z)")
    ax.set_xlabel("Year")
    save(fig, "figS4")


def figS5():
    """Oral MEP speeches 1996-2012: originally in English vs translated into English."""
    df = pd.read_parquet(ANNOTATED, columns=["emi", "year", "role", "written", "translated"])
    df = df[(df["role"] == "mep") & ~df["written"] & df["year"].between(1996, 2012)]
    series = [("Original English", yearly(df[~df["translated"]], "emi")),
              ("Translated into English", yearly(df[df["translated"]], "emi"))]
    fig, ax = plt.subplots(figsize=(ONE_HALF, 62 * MM))
    zero_line(ax)
    for (label, s), style in zip(series, LINES):
        plot_series(ax, s, style, label)
    ax.set_xlim(1995.5, 2016)
    direct_labels(ax, [(label, s.index[-1], s["mean"].iloc[-1]) for label, s in series])
    ax.legend(loc="lower left")
    ax.set_ylabel("EMI")
    ax.set_xlabel("Year")
    save(fig, "figS5")


def figS6():
    """Main sample by country group."""
    df = load_main()
    groups = ["United Kingdom", "Ireland", "Other EU"]
    series = [(g, yearly(df[df["country_group"] == g], "emi", min_n=50)) for g in groups]
    fig, ax = plt.subplots(figsize=(ONE_HALF, 62 * MM))
    zero_line(ax)
    for (label, s), style in zip(series, LINES):
        plot_series(ax, s, style, label)
    ax.set_xlim(1995.5, 2022.5)
    direct_labels(ax, [(label, s.index[-1], s["mean"].iloc[-1]) for label, s in series if len(s)])
    ax.legend(loc="lower left")
    ax.set_ylabel("EMI")
    ax.set_xlabel("Year")
    save(fig, "figS6")


def figS7():
    """Distribution of speech-level EMI by parliamentary term (box: IQR; whiskers: 5th-95th percentile)."""
    df = load_main()
    data = [df.loc[df["term"] == t, "emi"].dropna() for t in TERMS_MAIN]
    fig, ax = plt.subplots(figsize=(SINGLE, 62 * MM))
    zero_line(ax)
    ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
               boxprops={"facecolor": SAGE, "edgecolor": DARK, "linewidth": 0.6},
               medianprops={"color": DARK, "linewidth": 0.9},
               whiskerprops={"color": DARK, "linewidth": 0.6}, capprops={"color": DARK, "linewidth": 0.6})
    ax.plot(range(1, len(data) + 1), [d.mean() for d in data], "o", color=OCHRE, markersize=3, label="Mean")
    ax.set_xticks(range(1, len(data) + 1), TERMS_MAIN)
    ax.set_xlabel("Parliamentary term")
    ax.set_ylabel("Speech-level EMI")
    ax.legend(loc="lower left")
    save(fig, "figS7")


def figS8():
    """Speeches per year in the main sample by country group."""
    df = load_main(period=None)
    order = ["United Kingdom", "Ireland", "Other EU", "Unknown"]
    counts = pd.crosstab(df["year"].astype(int), df["country_group"]).reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(ONE_HALF, 62 * MM))
    bottom = np.zeros(len(counts))
    for col, color in zip(order, [OCHRE, PEACH, SAGE, MINT]):
        ax.bar(counts.index, counts[col], bottom=bottom, color=color, width=0.8,
               edgecolor="white", linewidth=0.3, label=col)
        bottom += counts[col].to_numpy()
    ax.set_ylabel("Speeches")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left")
    save(fig, "figS8")


def figS9():
    """Chunk-level EMI by year: original pipeline (z-scores within 1,000-row batches) vs this pipeline."""
    old = pd.read_csv(cfg.INTERIM / "outputv2.csv.gzip", compression="gzip",
                      usecols=["pubdate", "evidence_minus_intuition_score"])
    old["year"] = pd.to_datetime(old["pubdate"], errors="coerce", utc=True, format="mixed").dt.year
    old = old.rename(columns={"evidence_minus_intuition_score": "emi"})
    new = pd.read_parquet(CHUNKS, columns=["guid", "emi"]).merge(
        pd.read_parquet(cfg.PROCESSED / "05_speech_emi.parquet", columns=["guid", "year"]), on="guid")
    series = [("Global standardisation (this study)", yearly(new, "emi")),
              ("Batch standardisation (original code)", yearly(old.dropna(subset=["year"]), "emi"))]
    fig, ax = plt.subplots(figsize=(ONE_HALF, 62 * MM))
    zero_line(ax)
    for (label, s), style in zip(series, LINES):
        plot_series(ax, s, style, label)
    ax.legend(loc="lower left")
    ax.set_ylabel("Mean chunk-level EMI")
    ax.set_xlabel("Year")
    save(fig, "figS9")


# ---------------------------------------------------------------- run

FIGURES = {
    "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
    "figS1": figS1, "figS2": figS2, "tableS3": tableS3, "figS4": figS4, "figS5": figS5,
    "figS6": figS6, "figS7": figS7, "figS8": figS8, "figS9": figS9,
}


def main():
    parser = argparse.ArgumentParser(description="Build paper figures.")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), help="build only these")
    args = parser.parse_args()
    set_style()
    for name, build in FIGURES.items():
        if args.only and name not in args.only:
            continue
        print(f"{name}…")
        try:
            build()
        except Exception as e:
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
