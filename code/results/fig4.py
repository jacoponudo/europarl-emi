"""Figure 4. Mean EMI per year in each country of the MEP (one panel per country).

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (code/translation/unified.py)
  data/scores/minilm_150_<date>/speech_emi.parquet                         (code/scores/score_unified.py; --scores)
  data/raw/inherited/meps/mep_database_1994_present.csv                    (country of each MEP and term)

Population: MEP oral speeches with an English text, pooled (delivered in English, translated by hand by the
Parliament, translated with eTranslation), whose MEP has a country (results/meps.py). Speeches from a country
before it joined the EU are left out (counted in the terminal).
One panel per country with at least --min-total speeches, in alphabetical order, all on the same scales: mean
EMI per year with a 95% percentile bootstrap interval over MEPs (B = --boot); a year with fewer than
--min-speeches speeches or --min-meps MEPs is not drawn (gap). Dashed teal in every panel: the yearly mean
of all speeches, for comparison.
--without-human leaves the Parliament's human translations out.

    uv run python code/results/fig4.py --scores minilm_150_20261119

Out:
  results/figures/paper/fig4.pdf
  results/tables/fig4/emi_by_country_year.csv   country, year, n_speeches, n_meps, mean, ci_low, ci_high
  results/tables/fig4/country_by_year.csv       speeches per year and country (to check the assignment)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
import config as cfg  # noqa: E402
from results.meps import (category_means, drop_small_cells, small_multiples, speeches,  # noqa: E402
                          yearly_reference)
from results.style import ALL, INK, apply_style, save  # noqa: E402

OUT = cfg.TABLES / "fig4"
FIG = cfg.FIGURES / "paper" / "fig4"
NCOLS = 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", help="folder in data/scores (default: the latest minilm_150_*)")
    ap.add_argument("--without-human", action="store_true",
                    help="leave out the Parliament's human translations (to 2012)")
    ap.add_argument("--min-total", type=int, default=100, help="countries with fewer speeches get no panel")
    ap.add_argument("--min-speeches", type=int, default=20, help="minimum speeches for a country-year")
    ap.add_argument("--min-meps", type=int, default=3, help="minimum MEPs for a country-year")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (default 2000)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    sp = speeches(args.scores, args.without_human)
    ref = yearly_reference(sp)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.crosstab(sp.year, sp.country.fillna("missing")).to_csv(OUT / "country_by_year.csv")

    ok = sp[sp.country_check.eq("ok")]
    total = ok.country_name.value_counts()
    keep = sorted(total[total >= args.min_total].index)
    print(f"\ncountries with a panel: {len(keep)}; left out (< {args.min_total} speeches): "
          f"{total[total < args.min_total].to_dict()}")
    e = category_means(ok[ok.country_name.isin(keep)], ["country_name", "year"], "emi", B=args.boot)
    e = drop_small_cells(e, args.min_speeches, args.min_meps)
    e.to_csv(OUT / "emi_by_country_year.csv", index=False)
    print("\nMean EMI per year and country (blank: cell not drawn):")
    print(e.pivot(index="year", columns="country_name", values="mean").round(2).to_string())

    apply_style()
    nrows = -(-len(keep) // NCOLS)
    fig = small_multiples(e, ref, "country_name", keep, NCOLS, height_mm=12 + 30 * nrows, color=INK,
                          series_label="Speeches of MEPs from the country", ref_color=ALL[2],
                          ref_label="All speeches")
    save(fig, FIG)
    print(f"tables in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
