"""Figure 2. Mean EMI per year of MEP oral speeches, by the source of their English text.

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (code/translation/unified.py)
  data/scores/minilm_150_<date>/speech_emi.parquet                         (code/scores/score_unified.py; --scores)

Population: MEP oral speeches, in three groups by the source of their English text (translation/unified.py):
delivered in English; translated by hand by the Parliament (to 2012); translated with eTranslation (from 2013).
Mean EMI per year with 95% confidence intervals from the data (mean +/- t(0.975, n-1) * sd / sqrt(n),
one speech = one observation), for each group and for all speeches together. The area of each point is
proportional to the number of speeches in that year and group.
--without-human leaves the Parliament's human translations out of the groups and of "all speeches".

    uv run python code/results/fig2.py                                   # latest data/scores/minilm_150_* folder
    uv run python code/results/fig2.py --scores minilm_150_20261119 --without-human

Out:
  results/figures/paper/fig2.pdf
  results/tables/fig2/emi_by_year.csv   n, mean, sd, se and 95% interval per year and group
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
import config as cfg  # noqa: E402
from results.style import ALL, DOUBLE, MUTED, apply_style, mm, save, series, ygrid  # noqa: E402
from translation.unified import load  # noqa: E402

OUT = cfg.TABLES / "fig2"
FIG = cfg.FIGURES / "paper" / "fig2"
MAX_AREA = 60  # marker area (pt^2) of the largest year-group; area is proportional to the number of speeches


# ---------------------------------------------------------------- EMI

def latest_scores() -> Path:
    runs = sorted(cfg.SCORES.glob(f"{cfg.MODEL_SHORT}_{cfg.CHUNK_MAX_WORDS}_*/speech_emi.parquet"))
    if not runs:
        raise SystemExit("No EMI scores: run first  uv run python code/scores/score_unified.py")
    return runs[-1].parent


def speech_emi(u: pd.DataFrame, scores: Path, groups: list[str]) -> pd.DataFrame:
    sp = pd.read_parquet(scores / "speech_emi.parquet", columns=["uid", "emi"])
    sp = sp.merge(u[["uid", "year", "group"]], on="uid", how="inner")
    sp = sp[sp.group.isin(groups)]
    missing = [g for g in groups if not sp.group.eq(g).any()]
    if missing:
        raise SystemExit(f"{scores.name} has no scores for {missing}: run score_unified.py again.")
    print(f"EMI from {scores.relative_to(cfg.ROOT)}: {len(sp):,} speeches {sp.group.value_counts().to_dict()}")
    return sp


def emi_by_year(sp: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Mean EMI per year and group with its 95% confidence interval from the data:
    mean +/- t(0.975, n-1) * sd / sqrt(n), sd with n-1 degrees of freedom."""
    rows = []
    for key, sel in [("all", sp)] + [(g, sp[sp.group.eq(g)]) for g in groups]:
        for year, x in sel.groupby("year").emi:
            n, mean = len(x), x.mean()
            if n > 1:
                sd = x.std(ddof=1)
                se = sd / np.sqrt(n)
                half = stats.t.ppf(0.975, n - 1) * se
            else:
                sd = se = half = np.nan
            rows.append((key, int(year), n, mean, sd, se, mean - half, mean + half))
    return pd.DataFrame(rows, columns=["group", "year", "n", "mean", "sd", "se", "ci_low", "ci_high"])


# ---------------------------------------------------------------- figure

def size_key(n_max: int) -> list[int]:
    """Three round numbers of speeches for the size legend: the largest 1-2-5 number <= n_max, its half
    and its tenth."""
    exp = 10 ** int(np.floor(np.log10(n_max)))
    top = max(m * exp for m in (1, 2, 5) if m * exp <= n_max)
    return [int(v) for v in (top / 10, top / 2, top) if v >= 1]


def figure(e: pd.DataFrame, groups: list) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(mm(DOUBLE), mm(70)))
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    k = MAX_AREA / e.n.max()
    handles = []
    for key, label, color in groups + [ALL]:
        d = e[e.group.eq(key)].sort_values("year")
        dashed = key == "all"
        ax.fill_between(d.year, d.ci_low, d.ci_high, color=color, alpha=0.12 if dashed else 0.2,
                        linewidth=0, zorder=2)
        (line,) = ax.plot(d.year, d["mean"], color=color, zorder=4 if dashed else 3,
                          linewidth=0.9 if dashed else 1.0, linestyle=(0, (3, 1.5)) if dashed else "-")
        if dashed:  # all speeches: open rings, so they do not hide the groups underneath
            ax.scatter(d.year, d["mean"], s=k * d.n, facecolor="none", edgecolor=color, linewidth=0.6, zorder=5)
        else:
            ax.scatter(d.year, d["mean"], s=k * d.n, color=color, edgecolor="white", linewidth=0.3, zorder=4)
        handles.append((line, label))
    ax.set_ylabel("Mean EMI (95% CI)")
    ax.set_xlabel("Year")
    years = e.year.to_numpy()
    ax.set_xlim(years.min() - 0.7, years.max() + 0.7)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ygrid(ax)
    leg = ax.legend([h for h, _ in handles], [lab for _, lab in handles], loc="lower left",
                    handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    ax.add_artist(leg)
    vals = size_key(int(e.n.max()))
    dots = [ax.scatter([], [], s=k * v, color=MUTED, edgecolor="white", linewidth=0.3) for v in vals]
    ax.legend(dots, [f"{v:,}" for v in vals], title="Speeches", loc="lower left", bbox_to_anchor=(0.27, 0),
              borderaxespad=0.2, labelspacing=0.6, handletextpad=0.6, title_fontsize=6, scatterpoints=1)
    save(fig, FIG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", help="folder in data/scores (default: the latest minilm_150_*)")
    ap.add_argument("--without-human", action="store_true",
                    help="leave out the Parliament's human translations (to 2012)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    groups = series(args.without_human)
    keys = [k for k, _, _ in groups]
    u = load("mep-oral")
    sp = speech_emi(u, cfg.SCORES / args.scores if args.scores else latest_scores(), keys)
    e = emi_by_year(sp, keys)
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_csv(OUT / "emi_by_year.csv", index=False)
    print("\nMean EMI per year (95% interval: mean +/- t * sd / sqrt(n)):")
    print(e.pivot(index="year", columns="group", values=["mean", "n"]).round(3).to_string())
    print("\nhalf-width of the 95% interval (t * se):")
    print(e.assign(half=(e.ci_high - e.ci_low) / 2).pivot(index="year", columns="group", values="half").round(3).to_string())

    figure(e, groups)
    print(f"table in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
