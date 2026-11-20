"""Figure 1. MEP oral speeches per year, by the source of their English text (stacked bars).

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (code/translation/unified.py)

Population: MEP oral speeches, in three groups by the source of their English text (translation/unified.py):
delivered in English; translated by hand by the Parliament (to 2012); translated with eTranslation (from 2013).
eTranslation was run on every non-English text of unified from 2013: before 2013 the corpus holds only the
English version (the Parliament's human translation), so there is no original-language text to translate.
Speeches with no English text (group "other") are counted in the table but not drawn.
--without-human leaves the Parliament's human translations out.

    uv run python code/results/fig1.py
    uv run python code/results/fig1.py --without-human

Out:
  results/figures/paper/fig1.pdf
  results/tables/fig1/by_year.csv       speeches per year in each group
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
import config as cfg  # noqa: E402
from results.style import DOUBLE, apply_style, mm, save, series, ygrid  # noqa: E402
from translation.unified import load  # noqa: E402

OUT = cfg.TABLES / "fig1"
FIG = cfg.FIGURES / "paper" / "fig1"


def by_year(u: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(u.year.to_numpy(), u.group.to_numpy())
    years = range(int(t.index.min()), int(t.index.max()) + 1)
    t = t.reindex(index=years, columns=["original", "etranslation", "human", "other"], fill_value=0)
    t.index.name, t.columns.name = "year", None
    return t


def figure(t: pd.DataFrame, groups: list) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(mm(DOUBLE), mm(70)))
    x = t.index.to_numpy()
    bottom = 0
    for key, label, color in groups:
        ax.bar(x, t[key], bottom=bottom, width=0.8, color=color, label=label, edgecolor="white", linewidth=0.4)
        bottom = bottom + t[key]
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    ax.set_ylabel("Speeches per year")
    ax.set_xlabel("Year")
    ax.set_xlim(x.min() - 0.7, x.max() + 0.7)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ygrid(ax)
    # above the plot, one row: inside, it covers the bars of the busiest years
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(groups), handlelength=1.0,
              handleheight=0.8, borderaxespad=0.2, columnspacing=1.2)
    save(fig, FIG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--without-human", action="store_true",
                    help="leave out the Parliament's human translations (to 2012)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    t = by_year(load("mep-oral"))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "by_year.csv")
    print("\nSpeeches per year (other = no English text, not in the figure):")
    print(t.to_string())

    figure(t, series(args.without_human))
    print(f"table in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
