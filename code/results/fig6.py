"""Figure 6. Mean EMI per year by place of the sitting: Strasbourg and Brussels.

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (code/translation/unified.py)
  data/scores/minilm_150_<date>/speech_emi.parquet                         (code/scores/score_unified.py; --scores)
  data/raw/api/sittings.csv                                                (code/raw/download_sittings.py)

The place of each sitting day comes from the Parliament's Open Data API (vcard:hasLocality of MTG-PL-<day>),
available from 2014, so the figure covers 2014 onwards.
Population: MEP oral speeches with an English text, pooled (delivered in English, translated with eTranslation;
the Parliament's human translations end in 2012), on a sitting day with a known place.
One line per place: mean EMI per year with a 95% percentile bootstrap interval over MEPs (B = --boot); the area
of each point is proportional to the number of speeches. A year with fewer than --min-speeches speeches or
--min-meps MEPs at a place is not drawn. Grey band: March 2020 to December 2021, when all sittings were held
in Brussels with remote participation (Covid).

    uv run python code/raw/download_sittings.py                         # once: place of every sitting day
    uv run python code/results/fig6.py --scores minilm_150_20261119

Out:
  results/figures/paper/fig6.pdf
  results/tables/fig6/emi_by_place_year.csv   place, year, n_speeches, n_meps, days, mean, ci_low, ci_high
  results/tables/fig6/days_by_year.csv        sitting days per year and place (from sittings.csv)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
import config as cfg  # noqa: E402
from results.meps import category_means, drop_small_cells, latest_scores  # noqa: E402
from results.style import BAND, DOUBLE, MUTED, PALETTE, apply_style, mm, save, series, ygrid  # noqa: E402
from translation.unified import load  # noqa: E402

OUT = cfg.TABLES / "fig6"
FIG = cfg.FIGURES / "paper" / "fig6"
SITTINGS = cfg.DATA / "raw" / "api" / "sittings.csv"
PLACES = {"FRA_SXB": "Strasbourg", "BEL_BRU": "Brussels"}
COLORS = {"Strasbourg": PALETTE["orange"], "Brussels": PALETTE["teal-dark"]}
# March 2020 to December 2021; a yearly point at x = year stands for the interval [year - 0.5, year + 0.5]
COVID = (2020 - 0.5 + 2 / 12, 2021 + 0.5)
MAX_AREA = 60  # marker area (pt^2) of the largest year-place, as in Figure 2


def places() -> pd.DataFrame:
    if not SITTINGS.exists():
        raise SystemExit(f"{SITTINGS.relative_to(cfg.ROOT)} missing: run  uv run python code/raw/download_sittings.py")
    s = pd.read_csv(SITTINGS, dtype=str).fillna("")
    s["place"] = s.locality.map(PLACES)  # other codes or several codes: not used
    days = pd.crosstab(s.date.str[:4], s.locality.replace("", "(not in API)"))
    days.index.name = "year"
    print(f"{SITTINGS.name}: {len(s):,} days; Strasbourg or Brussels for {s.place.notna().sum():,}")
    print(days.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    days.to_csv(OUT / "days_by_year.csv")
    return s.loc[s.place.notna(), ["date", "place"]].rename(columns={"date": "day"})


def size_key(n_max: int) -> list[int]:
    exp = 10 ** int(np.floor(np.log10(n_max)))
    top = max(m * exp for m in (1, 2, 5) if m * exp <= n_max)
    return [int(v) for v in (top / 10, top / 2, top) if v >= 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", help="folder in data/scores (default: the latest minilm_150_*)")
    ap.add_argument("--min-speeches", type=int, default=20, help="minimum speeches for a place-year")
    ap.add_argument("--min-meps", type=int, default=3, help="minimum MEPs for a place-year")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (default 2000)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    pl = places()
    folder = cfg.SCORES / args.scores if args.scores else latest_scores()
    keys = [k for k, _, _ in series(False)]
    u = load("mep-oral")
    sp = pd.read_parquet(folder / "speech_emi.parquet", columns=["uid", "emi"])
    sp = sp.merge(u[["uid", "year", "date", "group"]], on="uid", how="inner")
    sp = sp[sp.group.isin(keys)]
    sp["day"] = pd.to_datetime(sp.date).dt.strftime("%Y-%m-%d")
    pid = pd.read_parquet(cfg.UNIFIED, columns=["uid", "person_id"])
    pid["person_id"] = pid.person_id.astype("string").str.extract(r"(\d+)", expand=False)
    sp = sp.merge(pid, on="uid", how="left").merge(pl, on="day", how="inner")
    print(f"\nspeeches with a known place: {len(sp):,}, {sp.day.min()} to {sp.day.max()}")

    e = category_means(sp, ["place", "year"], "emi", B=args.boot)
    e = e.merge(sp.groupby(["place", "year"]).day.nunique().rename("days").reset_index(), on=["place", "year"])
    e = drop_small_cells(e, args.min_speeches, args.min_meps)
    e = e.drop(columns="emi_raw")
    e.to_csv(OUT / "emi_by_place_year.csv", index=False)
    print("\nMean EMI per year and place (95% bootstrap over MEPs):")
    print(e.pivot(index="year", columns="place", values=["mean", "n_speeches", "days"]).round(3).to_string())
    m = e.pivot(index="year", columns="place", values="mean")
    if {"Strasbourg", "Brussels"} <= set(m.columns):
        print("\ndifference Strasbourg - Brussels per year:")
        print((m.Strasbourg - m.Brussels).round(3).to_string())

    apply_style()
    fig, ax = plt.subplots(figsize=(mm(DOUBLE), mm(70)))
    ax.axvspan(*COVID, color=BAND, zorder=0, linewidth=0)
    ax.text(np.mean(COVID), 0.98, "Covid: all sittings\nin Brussels", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.5, color=MUTED, zorder=6,
            bbox={"facecolor": BAND, "edgecolor": "none", "pad": 1.0})
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    k = MAX_AREA / e.n_speeches.max()
    handles = []
    all_years = range(int(e.year.min()), int(e.year.max()) + 1)
    for place in ["Strasbourg", "Brussels"]:
        d = e[e.place.eq(place)].set_index("year").reindex(all_years).rename_axis("year").reset_index()
        if d["mean"].isna().all():
            continue  # a year without data is a gap in the line, not a straight segment across it
        c = COLORS[place]
        ax.fill_between(d.year, d.ci_low, d.ci_high, color=c, alpha=0.2, linewidth=0, zorder=2)
        (line,) = ax.plot(d.year, d["mean"], color=c, linewidth=1.0, zorder=3)
        ax.scatter(d.year, d["mean"], s=k * d.n_speeches, color=c, edgecolor="white", linewidth=0.3, zorder=4)
        handles.append((line, place))
    years = e.year.to_numpy()
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean EMI (95% CI)")
    ygrid(ax)
    leg = ax.legend([h for h, _ in handles], [lab for _, lab in handles], loc="lower left",
                    handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    ax.add_artist(leg)
    vals = size_key(int(e.n_speeches.max()))
    dots = [ax.scatter([], [], s=k * v, color=MUTED, edgecolor="white", linewidth=0.3) for v in vals]
    ax.legend(dots, [f"{v:,}" for v in vals], title="Speeches", loc="lower left", bbox_to_anchor=(0.16, 0),
              borderaxespad=0.2, labelspacing=0.6, handletextpad=0.6, title_fontsize=6, scatterpoints=1)
    save(fig, FIG)
    print(f"tables in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
