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

import sys

import matplotlib

if "ipykernel" not in sys.modules:
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

# Paper palette. Names are kept for compatibility with existing code and notebooks;
# each colour has one role in every figure.
OCHRE = "#b3001b"   # primary series: lines and points (red)
PEACH = "#baacbd"   # band of the primary series (mauve)
GREY = "#646e68"    # secondary series: lines and edges (dark grey-green)
SAGE = "#a3bba8"    # secondary fills (sage)
MINT = "#c7efcf"    # lightest fill (mint)
WINE, SLATE, MIST, ASH, PLUM = "#8c3742", "#7f8780", "#c1cec6", "#8f8d93", "#371e30"
INK, MUTED, DARK = "#1a1a1a", "#5a5a5a", PLUM  # DARK: third line series

# Only OCHRE (red), GREY (dark grey-green) and DARK (plum) are dark enough for lines on white;
# the light colours are fills only, and SAGE and PEACH are never placed next to each other.
# Series are also told apart by line style and marker, and labelled directly.
LINES = [
    {"color": OCHRE, "ls": "-", "marker": "o"},
    {"color": GREY, "ls": "--", "marker": "s"},
    {"color": DARK, "ls": ":", "marker": "^"},
]
FAMILY_ORDER = ["Left", "Greens/EFA", "S&D", "Liberals", "EPP", "Conservatives", "Eurosceptics", "Non-attached"]
TERMS_MAIN = ["4th", "5th", "6th", "7th", "8th"]
TERM_BOUNDARIES = [1999.5, 2004.5, 2009.5, 2014.5, 2019.5]
LOW_COVERAGE = "#e6e6e6"  # bands and points after 2019: white-grey, whatever the series


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


BOOT_B = 2000  # bootstrap replicates for every confidence interval in the figures


def boot_ci(values, b=BOOT_B, seed=None, chunk=200):
    """Mean and 95% percentile-bootstrap interval of the mean (speeches resampled with replacement)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(cfg.SEED if seed is None else seed)
    means = np.concatenate([v[rng.integers(0, len(v), size=(min(chunk, b - i), len(v)))].mean(axis=1)
                            for i in range(0, b, chunk)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return v.mean(), lo, hi


def yearly(df, col, min_n=1):
    """Yearly mean with 95% percentile-bootstrap interval."""
    rows = []
    for year, g in df.groupby("year"):
        v = g[col].dropna().to_numpy()
        if len(v) >= min_n:
            m, lo, hi = boot_ci(v, seed=cfg.SEED + int(year))
            rows.append((int(year), m, lo, hi, len(v)))
    return pd.DataFrame(rows, columns=["year", "mean", "lo", "hi", "count"]).set_index("year")


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
    fills = [OCHRE, SAGE, GREY, PEACH, MINT]
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
    """EMI by year: speeches delivered in English and all oral MEP speeches including translations.
    Same marker for both series; point area proportional to speeches per year;
    years after 2019 (low coverage) drawn with white markers and white-grey bands."""
    original = load_main(period=None)[["year", "emi"]]
    oral = pd.read_parquet(ANNOTATED, columns=["year", "emi", "role", "written"])
    oral = oral[(oral["role"] == "mep") & ~oral["written"]]
    # (label, data, line colour, marker fill, band colour, band alpha, line style); drawn in this order
    series = [
        ("Including translations", yearly(oral, "emi"), GREY, SAGE, SAGE, 0.45, "--"),
        ("Delivered in English", yearly(original, "emi"), OCHRE, OCHRE, PEACH, 0.5, "-"),
    ]
    n_max = max(s_["count"].max() for _, s_, *_ in series)

    def area(n):
        return 3 + 60 * np.asarray(n) / n_max

    fig, ax = plt.subplots(figsize=(ONE_HALF, 72 * MM))
    last = max(s_.index.max() for _, s_, *_ in series)
    zero_line(ax)
    for b in TERM_BOUNDARIES:
        ax.axvline(b, color="#d0d0d0", linewidth=0.4, zorder=0)
    handles = []
    for z, (label, s_, line, fill, band, alpha, ls) in enumerate(series):
        core, late = s_[s_.index <= 2019], s_[s_.index > 2019]
        edge = s_[s_.index >= 2019]  # low-coverage band starts at 2019 so the two bands join
        ax.fill_between(core.index, core["lo"], core["hi"], color=band, alpha=alpha, linewidth=0, zorder=1 + z)
        ax.fill_between(edge.index, edge["lo"], edge["hi"], color=LOW_COVERAGE, alpha=0.8, linewidth=0, zorder=1 + z)
        ax.plot(s_.index, s_["mean"], color=line, ls=ls, linewidth=1, zorder=3 + z)
        ax.scatter(core.index, core["mean"], s=area(core["count"]), marker="o", facecolor=fill,
                   edgecolor=line, linewidth=0.6, zorder=5 + z)
        ax.scatter(late.index, late["mean"], s=area(late["count"]), marker="o", facecolor="white",
                   edgecolor=line, linewidth=0.6, zorder=5 + z)
        handles.append(plt.Line2D([], [], color=line, ls=ls, marker="o", markersize=3.5,
                                  markerfacecolor=fill, markeredgecolor=line, label=label))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ymin, ymax = ax.get_ylim()
    for x, label in zip([1997.75, 2002, 2007, 2012, 2017], ["4th", "5th", "6th", "7th", "8th"]):
        ax.text(x, 1.01, f"{label} term", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color=MUTED)
    ax.text((2019.5 + last) / 2, 1.01, "Low coverage", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=5.5, color=MUTED)
    ax.axvline(2012.5, color=INK, linewidth=0.5, ls=":", zorder=1)
    ax.text(2012.7, ymin + 0.03 * (ymax - ymin), "End of\ntranslations", fontsize=5.5, color=MUTED, va="bottom")

    first = ax.legend(handles=handles[::-1], loc="upper right", bbox_to_anchor=(1, 0.93), fontsize=6)
    ax.add_artist(first)
    ref = [n for n in (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000) if n <= n_max]
    ref = sorted({ref[0], ref[len(ref) // 2], ref[-1]})
    sizes = [ax.scatter([], [], s=area(n), color=MUTED, linewidth=0, label=f"{n:,}") for n in ref]
    ax.legend(handles=sizes, title="Speeches per year", loc="lower left", fontsize=5.5, title_fontsize=5.5,
              ncol=len(sizes), columnspacing=1.0, handletextpad=0.3, borderpad=0.6)
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
    rows = []
    for (family, term), grp in df.groupby(["family", "term"], observed=True):
        if len(grp) >= 50:
            m, lo, hi = boot_ci(grp["emi"].to_numpy())
            rows.append((family, term, m, lo, hi))
    g = pd.DataFrame(rows, columns=["family", "term", "mean", "lo", "hi"])

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 85 * MM), sharex=True, sharey=True)
    for ax, family in zip(axes.flat, FAMILY_ORDER):
        sub = g[g["family"] == family].set_index("term").reindex(TERMS_MAIN)
        x = np.arange(len(TERMS_MAIN))
        zero_line(ax)
        ax.errorbar(x, sub["mean"], yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]], color=OCHRE,
                    marker="o", markersize=3,
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
    """Processing pipeline with the number of texts at each stage, and why texts leave the main sample."""
    raw = len(pd.read_csv(cfg.RAW_CORPUS / "all_years_combined.csv.gzip", compression="gzip", usecols=["language"]))
    main_all = nrows(cfg.PROCESSED / "07_speech_main_sample.parquet")
    main_period = len(load_main())
    stages = [
        ("Raw corpus, all languages", raw, None),
        ("English-language speeches", nrows(cfg.INTERIM / "01_speeches_en.parquet"), None),
        ("Very short texts removed", nrows(cfg.INTERIM / "02_speeches_clean.parquet"), None),
        ("Procedural texts removed", nrows(cfg.INTERIM / "03_speeches_substantive.parquet"), None),
        ("Scored speeches", nrows(cfg.PROCESSED / "05_speech_emi.parquet"), None),
        ("Main sample, 1996–2019: oral speeches\nby MEPs, delivered in English", main_period,
         f"({main_all:,} including 2020–2023)"),
    ]

    ann = pd.read_parquet(ANNOTATED, columns=["role", "written", "translated", "header_found"]).reset_index(drop=True)
    unrecognised = ~ann["header_found"] | (ann["role"] == "unknown")
    non_mep = ~unrecognised & ann["role"].isin(["chair", "commission", "council"])
    mep = ~unrecognised & (ann["role"] == "mep")
    excluded = [
        ("translated into English", int((mep & ~ann["written"] & ann["translated"]).sum())),
        ("written statements", int((mep & ann["written"]).sum())),
        ("chair, Commission or Council", int(non_mep.sum())),
        ("header not recognised", int(unrecognised.sum())),
        ("speeches after 2019", main_all - main_period),
    ]
    last_note = "Speech header parsed. Excluded:\n" + "\n".join(f"{label}: {n:,}" for label, n in excluded)
    notes = ["language = English", "≥ 3 words and ≥ 16 characters", "NB-LR classifier, texts < 400 characters",
             "> 10 words; ≤ 150-word chunks; MiniLM similarity", last_note]

    fig, ax = plt.subplots(figsize=(ONE_HALF, 125 * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, len(stages))
    ax.axis("off")
    for i, (label, n, extra) in enumerate(stages):
        yc = len(stages) - i - 0.5
        fill = SAGE if i == len(stages) - 1 else MINT
        ax.add_patch(FancyBboxPatch((0.03, yc - 0.33), 0.5, 0.66, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor=fill, edgecolor=GREY, linewidth=0.5))
        ax.text(0.28, yc + (0.14 if "\n" in label else 0.1), label, ha="center", va="center", fontsize=6.5, linespacing=1.1)
        ax.text(0.28, yc - 0.2, f"n = {n:,}", ha="center", va="center", fontsize=6.5, fontweight="bold")
        if extra:
            ax.text(0.28, yc - 0.47, extra, ha="center", va="center", fontsize=5.5, color=MUTED)
        if i < len(notes):
            ax.annotate("", xy=(0.28, yc - 0.67), xytext=(0.28, yc - 0.33),
                        arrowprops={"arrowstyle": "-|>", "color": GREY, "linewidth": 0.6})
            ax.text(0.56, yc - 0.5, notes[i], va="center", fontsize=6, color=MUTED, style="italic", linespacing=1.25)
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

    write_latex_tables(out, lex, tables)
    print("  saved results/tables/S3_examples.csv/.tex and S3_lexicons.csv/.tex")


LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
SAFE_EXTRA = set("‘’“”–—…€șțȘȚ")


def latex_escape(text):
    """Escape LaTeX special characters and drop characters that pdflatex (T1) cannot typeset."""
    import unicodedata

    out = []
    for ch in str(text):
        if ch in LATEX_SPECIAL:
            out.append(LATEX_SPECIAL[ch])
        elif ch in "\u00a0\u2009\u202f":
            out.append(" ")
        elif ch in "\u2010\u2011\u2212":
            out.append("-")
        elif ch == "…":
            out.append(r"\ldots{}")
        elif ord(ch) < 0x180 or ch in SAFE_EXTRA:  # Latin-1 + Latin Extended-A (ł, ś, č, ő ...)
            out.append(ch)
        else:
            out.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode())
    return "".join(out)


def write_latex_tables(examples, lexicons, folder, per_end=5, chars=200):
    """LaTeX versions of Supplementary Table S3, to \\input{} in the paper (compact: fits one page)."""
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}r r l l p{8.8cm}@{}}", r"\toprule",
             r"EMI & Year & Family & Country & Excerpt \\", r"\midrule"]
    for end, group in examples.groupby("end", sort=False):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{end}}}}} \\")
        for _, r in group.head(per_end).iterrows():
            country = "--" if pd.isna(r["country"]) else str(r["country"]).title()
            family = "--" if pd.isna(r["family"]) else r["family"]
            excerpt = str(r["excerpt"]).rstrip("…")[:chars].rsplit(" ", 1)[0] + "…"
            lines.append(f"{r['emi']:.2f} & {int(r['year'])} & {latex_escape(family)} & "
                         f"{latex_escape(country)} & {latex_escape(excerpt)} \\\\[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder / "S3_examples.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lex_lines = [r"\footnotesize", r"\begin{tabular}{@{}p{7.6cm} p{7.6cm}@{}}", r"\toprule",
                 rf"Evidence ({len(lexicons['evidence'])} terms) & Intuition ({len(lexicons['intuition'])} terms) \\",
                 r"\midrule",
                 ", ".join(map(latex_escape, lexicons["evidence"])) + " & "
                 + ", ".join(map(latex_escape, lexicons["intuition"])) + r" \\",
                 r"\bottomrule", r"\end{tabular}"]
    (folder / "S3_lexicons.tex").write_text("\n".join(lex_lines) + "\n", encoding="utf-8")


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
    for col, color in zip(order, [OCHRE, PEACH, GREY, MINT]):
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
