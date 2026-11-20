"""Figure 5. Mean EMI per year in each political family of the MEP (one panel per family).

Data
  data/translation/unified.parquet, data/translation/etranslation.sqlite   (code/translation/unified.py)
  data/scores/minilm_150_<date>/speech_emi.parquet                         (code/scores/score_unified.py; --scores)
  data/raw/inherited/meps/mep_database_1994_present.csv                    (group of each MEP and term, fallback)

Population: MEP oral speeches with an English text, pooled (delivered in English, translated by hand by the
Parliament, translated with eTranslation), whose political group is known (results/meps.py): the European
group at the time of the speech, from the acronym in the header, or else from the database for that MEP and
term. Groups are joined into families stable over time (results/meps.py: FAMILIES, FAMILY_WORDS).
One panel per family with at least --min-total speeches, from left to right, all on the same scales: mean EMI
per year with a 95% percentile bootstrap interval over MEPs (B = --boot); a year with fewer than
--min-speeches speeches or --min-meps MEPs is not drawn (gap). Dashed teal in every panel: the yearly mean
of all speeches, for comparison.
--without-human leaves the Parliament's human translations out.

    uv run python code/results/fig5.py --scores minilm_150_20261119

Out:
  results/figures/paper/fig5.pdf
  results/tables/fig5/emi_by_family_year.csv   family, year, n_speeches, n_meps, mean, ci_low, ci_high
  results/tables/fig5/groups.csv               acronym in the header -> family, with counts (to check the mapping)
  results/tables/fig5/family_by_year.csv       speeches per year and family
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
from results.meps import (FAMILY_ORDER, category_means, drop_small_cells, small_multiples,  # noqa: E402
                          speeches, yearly_reference)
from results.style import ALL, INK, apply_style, save  # noqa: E402

OUT = cfg.TABLES / "fig5"
FIG = cfg.FIGURES / "paper" / "fig5"
NCOLS = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", help="folder in data/scores (default: the latest minilm_150_*)")
    ap.add_argument("--without-human", action="store_true",
                    help="leave out the Parliament's human translations (to 2012)")
    ap.add_argument("--min-total", type=int, default=100, help="families with fewer speeches get no panel")
    ap.add_argument("--min-speeches", type=int, default=20, help="minimum speeches for a family-year")
    ap.add_argument("--min-meps", type=int, default=3, help="minimum MEPs for a family-year")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (default 2000)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    sp = speeches(args.scores, args.without_human)
    ref = yearly_reference(sp)
    OUT.mkdir(parents=True, exist_ok=True)
    g = sp.groupby([sp.group_raw.fillna("(none in header)"), sp.family.fillna("missing")]).size()
    g.rename("speeches").sort_values(ascending=False).to_csv(OUT / "groups.csv")
    pd.crosstab(sp.year, sp.family.fillna("missing")).to_csv(OUT / "family_by_year.csv")

    ok = sp[sp.family.notna()]
    total = ok.family.value_counts()
    keep = [f for f in FAMILY_ORDER if total.get(f, 0) >= args.min_total]
    print(f"\nfamilies with a panel: {keep}; speeches per family: {total.to_dict()}")
    e = category_means(ok[ok.family.isin(keep)], ["family", "year"], "emi", B=args.boot)
    e = drop_small_cells(e, args.min_speeches, args.min_meps)
    e.to_csv(OUT / "emi_by_family_year.csv", index=False)
    print("\nMean EMI per year and family (blank: cell not drawn):")
    print(e.pivot(index="year", columns="family", values="mean")[keep].round(2).to_string())

    apply_style()
    nrows = -(-len(keep) // NCOLS)
    fig = small_multiples(e, ref, "family", keep, NCOLS, height_mm=12 + 38 * nrows, color=INK,
                          series_label="Speeches of MEPs of the family", ref_color=ALL[2],
                          ref_label="All speeches")
    save(fig, FIG)
    print(f"tables in {OUT.relative_to(cfg.ROOT)}/")


if __name__ == "__main__":
    main()
