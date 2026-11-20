"""Figure 3. The two components of EMI per year: mean evidence score and mean intuition score, all speeches.

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (code/translation/unified.py)
  data/scores/minilm_150_<date>/speech_emi.parquet                         (code/scores/score_unified.py; --scores)

Population: all MEP oral speeches with an English text, with no distinction between speeches delivered in
English and translated ones (delivered in English, translated by hand by the Parliament to 2012, translated
with eTranslation from 2013, pooled).
For every chunk, the cosine similarity with each lexicon is corrected for length and standardised over all
chunks (z); EMI = z(evidence) - z(intuition). A speech's components are the means over its chunks
(evidence_z, intuition_z), so speech EMI = evidence_z - intuition_z exactly.
One panel: mean evidence_z and mean intuition_z per year, with 95% confidence intervals from the data
(mean +/- t(0.975, n-1) * sd / sqrt(n), one speech = one observation).
The area of each point is proportional to the number of speeches in that year (the same for both lines).
--without-human leaves the Parliament's human translations out.

    uv run python code/results/fig3.py                                   # latest data/scores/minilm_150_* folder
    uv run python code/results/fig3.py --scores minilm_150_20261119 --without-human

Out:
  results/figures/paper/fig3.pdf
  results/tables/fig3/components_by_year.csv   component, year, n, mean, sd, se and 95% interval
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
from results.style import (DOUBLE, EVIDENCE, INTUITION, MUTED, apply_style, mm, save,  # noqa: E402
                           series, ygrid)
from translation.unified import load  # noqa: E402

OUT = cfg.TABLES / "fig3"
FIG = cfg.FIGURES / "paper" / "fig3"
MAX_AREA = 60  # marker area (pt^2) of the largest year, as in Figure 2
COMPONENTS = [EVIDENCE, INTUITION]


# ---------------------------------------------------------------- data

def latest_scores() -> Path:
    runs = sorted(cfg.SCORES.glob(f"{cfg.MODEL_SHORT}_{cfg.CHUNK_MAX_WORDS}_*/speech_emi.parquet"))
    if not runs:
        raise SystemExit("No EMI scores: run first  uv run python code/scores/score_unified.py")
    return runs[-1].parent


def speech_scores(u: pd.DataFrame, scores: Path, groups: list[str]) -> pd.DataFrame:
    sp = pd.read_parquet(scores / "speech_emi.parquet", columns=["uid", "emi", "evidence_z", "intuition_z"])
    sp = sp.merge(u[["uid", "year", "group"]], on="uid", how="inner")
    sp = sp[sp.group.isin(groups)]
    missing = [g for g in groups if not sp.group.eq(g).any()]
    if missing:
        raise SystemExit(f"{scores.name} has no scores for {missing}: run score_unified.py again.")
    print(f"scores from {scores.relative_to(cfg.ROOT)}: {len(sp):,} speeches, pooled "
          f"{sp.group.value_counts().to_dict()}")
    # check: speech EMI is the difference of its two components
    gap = (sp.emi - (sp.evidence_z - sp.intuition_z)).abs().max()
    print(f"check, max |emi - (evidence_z - intuition_z)| over speeches: {gap:.2e}")
    return sp


def by_year(sp: pd.DataFrame) -> pd.DataFrame:
    """Mean of each component per year, all speeches pooled, with its 95% confidence interval from the data:
    mean +/- t(0.975, n-1) * sd / sqrt(n), sd with n-1 degrees of freedom."""
    rows = []
    for col, _, _ in COMPONENTS:
        for year, x in sp.groupby("year")[col]:
            n, mean = len(x), x.mean()
            if n > 1:
                sd = x.std(ddof=1)
                se = sd / np.sqrt(n)
                half = stats.t.ppf(0.975, n - 1) * se
            else:
                sd = se = half = np.nan
            rows.append((col, int(year), n, mean, sd, se, mean - half, mean + half))
    return pd.DataFrame(rows, columns=["component", "year", "n", "mean", "sd", "se", "ci_low", "ci_high"])


# ---------------------------------------------------------------- figure

def size_key(n_max: int) -> list[int]:
    """Three round numbers of speeches for the size legend: the largest 1-2-5 number <= n_max, its half
    and its tenth."""
    exp = 10 ** int(np.floor(np.log10(n_max)))
    top = max(m * exp for m in (1, 2, 5) if m * exp <= n_max)
    return [int(v) for v in (top / 10, top / 2, top) if v >= 1]


def figure(e: pd.DataFrame) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(mm(DOUBLE), mm(75)))
    fig.subplots_adjust(right=0.89)  # room for the size key on the right, inside the 183 mm width
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    k = MAX_AREA / e.n.max()
    handles = []
    for key, label, color in COMPONENTS:
        d = e[e.component.eq(key)].sort_values("year")
        ax.fill_between(d.year, d.ci_low, d.ci_high, color=color, alpha=0.2, linewidth=0, zorder=2)
        (line,) = ax.plot(d.year, d["mean"], color=color, linewidth=1.0, zorder=3)
        ax.scatter(d.year, d["mean"], s=k * d.n, color=color, edgecolor="white", linewidth=0.3, zorder=4)
        handles.append((line, label))
    ax.set_ylabel("Mean score, z (95% CI)")
    ax.set_xlabel("Year")
    years = e.year.to_numpy()
    ax.set_xlim(years.min() - 0.7, years.max() + 0.7)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ygrid(ax)
    leg = ax.legend([h for h, _ in handles], [lab for _, lab in handles], loc="lower right",
                    bbox_to_anchor=(1, 1.0), ncol=len(handles), handlelength=1.6, borderaxespad=0.2,
                    columnspacing=1.2)
    ax.add_artist(leg)
    vals = size_key(int(e.n.max()))
    dots = [ax.scatter([], [], s=k * v, color=MUTED, edgecolor="white", linewidth=0.3) for v in vals]
    ax.legend(dots, [f"{v:,}" for v in vals], title="Speeches", loc="lower left", bbox_to_anchor=(1.005, 0),
              borderaxespad=0.2, labelspacing=0.8, handletextpad=0.6, title_fontsize=6, scatterpoints=1)
    save(fig, FIG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", help="folder in data/scores (default: the latest minilm_150_*)")
    ap.add_argument("--without-human", action="store_true",
                    help="leave out the Parliament's human translations (to 2012)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    keys = [k for k, _, _ in series(args.without_human)]
    u = load("mep-oral")
    sp = speech_scores(u, cfg.SCORES / args.scores if args.scores else latest_scores(), keys)
    e = by_year(sp)
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_csv(OUT / "components_by_year.csv", index=False)
    w = e.pivot(index="year", columns="component", values="mean")
    w["n"] = e.groupby("year").n.first()
    w["emi"] = sp.groupby("year").emi.mean()
    print("\nMean per year, all speeches pooled:")
    print(w.round(3).to_string())
    # check: per year, mean evidence - mean intuition = mean EMI
    gap = (w.evidence_z - w.intuition_z - w.emi).abs().max()
    print(f"\ncheck, max |mean evidence - mean intuition - mean EMI| over years: {gap:.2e}")

    figure(e)
    print(f"table in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
