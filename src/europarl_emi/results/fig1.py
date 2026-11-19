"""Figure 1. Speeches delivered in English and translated with eTranslation: how many, and their EMI.

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (translation/unified.py)
  data/scores/minilm_150_<date>/speech_emi.parquet                         (scores/score_unified.py; --scores)
eTranslation was run on every non-English text of unified from 2013: before 2013 the corpus holds only the
English version (the Parliament's human translation), so there is no original-language text to translate.

Population: MEP oral speeches. Groups: delivered in English, translated with eTranslation.
Panel a  speeches per year in each group (stacked bars). --show-human adds the Parliament's human
         translations (to 2012) as a third series.
Panel b  mean EMI per year, with percentile bootstrap 95% intervals (resampling speeches within year,
         B = 2000, seed 42), for: all speeches (both groups), delivered in English, translated.

    uv run python -m europarl_emi.scores.score_unified    # once: EMI scores (slow part)
    uv run python -m europarl_emi.results.fig1            # latest data/scores/minilm_150_* folder
    uv run python -m europarl_emi.results.fig1 --scores minilm_150_20261119 --show-human

Out:
  results/figures/paper/fig1.pdf / .png
  results/tables/fig1/by_year.csv       panel a: speeches per year in each group
  results/tables/fig1/emi_by_year.csv   panel b: mean EMI, bootstrap interval and n per year and group
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from europarl_emi import config as cfg  # noqa: E402
from europarl_emi.results.style import (  # noqa: E402
    BAND, DOUBLE, INK, MUTED, PALETTE, apply_style, mm, panel_label, save, ygrid,
)
from europarl_emi.translation.unified import load  # noqa: E402

OUT = cfg.TABLES / "fig1"
FIG = cfg.FIGURES / "paper" / "fig1"
B = 2000

# (key, legend label, colour)
ORIGINAL = ("original", "Delivered in English", PALETTE["Cherry Rose"])
ETRANSLATION = ("etranslation", "Translated with eTranslation", PALETTE["Pacific Cyan"])
HUMAN = ("human", "Human translation (European Parliament)", PALETTE["Dusty Grape"])
ALL = ("all", "All speeches", PALETTE["Dusty Grape"])


# ---------------------------------------------------------------- data

def by_year(u: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(u.year.to_numpy(), u.group.to_numpy())
    years = range(int(t.index.min()), int(t.index.max()) + 1)
    t = t.reindex(index=years, columns=["original", "etranslation", "human", "other"], fill_value=0)
    t.index.name, t.columns.name = "year", None
    return t


# ---------------------------------------------------------------- EMI

def latest_scores() -> Path:
    runs = sorted(cfg.SCORES.glob(f"{cfg.MODEL_SHORT}_{cfg.CHUNK_MAX_WORDS}_*/speech_emi.parquet"))
    if not runs:
        raise SystemExit("No EMI scores: run first  uv run python -m europarl_emi.scores.score_unified")
    return runs[-1].parent


def speech_emi(u: pd.DataFrame, scores: Path) -> pd.DataFrame:
    sp = pd.read_parquet(scores / "speech_emi.parquet", columns=["uid", "emi"])
    sp = sp.merge(u[["uid", "year", "group"]], on="uid", how="inner")
    sp = sp[sp.group.isin(["original", "etranslation"])]
    print(f"EMI from {scores.relative_to(cfg.ROOT)}: {len(sp):,} speeches {sp.group.value_counts().to_dict()}")
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
    ap.add_argument("--scores", help="folder in data/scores (default: the latest minilm_150_*)")
    ap.add_argument("--show-human", action="store_true", help="panel a: add the Parliament's human translations")
    ap.add_argument("--bootstrap", type=int, default=B, help=f"bootstrap resamples (default {B})")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    u = load("mep-oral")
    t = by_year(u)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "by_year.csv")
    print("\nPanel a, speeches per year (other = no English text, not in the figure):")
    print(t.to_string())

    sp = speech_emi(u, cfg.SCORES / args.scores if args.scores else latest_scores())
    e = emi_by_year(sp, args.bootstrap)
    e.to_csv(OUT / "emi_by_year.csv", index=False)
    print(f"\nPanel b, mean EMI per year (percentile bootstrap 95%, B = {args.bootstrap}):")
    print(e.pivot(index="year", columns="group", values=["mean", "n"]).round(3).to_string())

    figure(t, e, args.show_human)
    print(f"tables in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
