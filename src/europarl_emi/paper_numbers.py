"""All numbers needed to write the paper, in one text report.

Run after steps 1-8:
    uv run python -m europarl_emi.paper_numbers | tee results/paper_numbers.txt
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from europarl_emi import config as cfg
from europarl_emi.step08_models import ANNOTATED, CHUNKS, TABLES, load_main

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 220)
EARLY, LATE = (1996, 1998), (2017, 2019)


def title(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def ci(s):
    m, se = s.mean(), s.std() / np.sqrt(s.count())
    return f"{m:.3f} [{m - 1.96 * se:.3f}, {m + 1.96 * se:.3f}] (n={s.count():,})"


def period(df, col, span):
    return df.loc[df["year"].between(*span), col]


def nrows(path):
    return pq.ParquetFile(path).metadata.num_rows


def main():
    title("A. Corpus flow")
    raw = len(pd.read_csv(cfg.RAW_CORPUS / "all_years_combined.csv.gzip", compression="gzip", usecols=["language"]))
    print(f"raw corpus: {raw:,}")
    for name in ("01_speeches_en", "02_speeches_clean", "03_speeches_substantive"):
        print(f"{name}: {nrows(cfg.INTERIM / f'{name}.parquet'):,}")
    print(f"chunks scored: {nrows(CHUNKS):,}")
    print(f"speeches scored: {nrows(cfg.PROCESSED / '05_speech_emi.parquet'):,}")

    ann = pd.read_parquet(ANNOTATED, columns=["year", "role", "written", "translated", "header_found"])
    print(f"header recognised: {ann['header_found'].mean():.1%}")
    ann["period"] = pd.cut(ann["year"], [1995, 2012, 2016, 2019, 2023], labels=["1996-2012", "2013-2016", "2017-2019", "2020-2023"])
    found = ann[ann["header_found"]]
    comp = found.groupby("period", observed=True).agg(
        n=("year", "size"), translated=("translated", "mean"), written=("written", "mean"),
        mep=("role", lambda s: (s == "mep").mean()),
        non_mep=("role", lambda s: s.isin(["chair", "commission", "council"]).mean()))
    print("\ncomposition by period (shares):\n" + comp.round(3).to_string())

    title("B. Main sample")
    all_years = load_main(period=None)
    main = load_main()
    print(f"main sample, all years: {len(all_years):,}   1996-2019: {len(main):,}")
    print(f"distinct speakers (1996-2019): {main['mep_id'].nunique():,}   with MEP id: {main['mep_id'].notna().mean():.1%}")
    print(f"agenda items: {main['agenda_item'].nunique():,}")
    print("\ncountry group shares by term:")
    print(pd.crosstab(main["term"], main["country_group"], normalize="index").round(3).to_string())
    print("\nfamily shares by term:")
    print(pd.crosstab(main["term"], main["family"], normalize="index").round(3).to_string())

    title("C. EMI by year, main sample")
    g = all_years.groupby("year")["emi"].agg(["mean", "std", "count"])
    g["lo"] = g["mean"] - 1.96 * g["std"] / np.sqrt(g["count"])
    g["hi"] = g["mean"] + 1.96 * g["std"] / np.sqrt(g["count"])
    print(g.round(3).to_string())
    print(f"\n{EARLY}: {ci(period(main, 'emi', EARLY))}")
    print(f"{LATE}: {ci(period(main, 'emi', LATE))}")
    print(f"speech-level EMI sd (1996-2019): {main['emi'].std():.3f}")

    title("D. Trend models (change in EMI per decade)")
    models = pd.read_csv(TABLES / "trend_models.csv")
    print(models.round(4).to_string(index=False))
    sd = main["emi"].std()
    base = models.loc[models["spec"] == "Baseline", "coef"].iat[0]
    fe = models.loc[models["spec"] == "Speaker fixed effects", "coef"].iat[0]
    print(f"\nbaseline in sd units per decade: {base / sd:.3f}")
    print(f"within-speaker / baseline ratio: {fe / base:.2f}")

    title("E. Within-speaker year effects vs raw change")
    eff = pd.read_csv(TABLES / "year_effects_speaker_fe.csv").set_index("year")
    within = eff.loc[LATE[0]:LATE[1], "coef"].mean() - eff.loc[EARLY[0]:EARLY[1], "coef"].mean()
    raw_change = period(main, "emi", LATE).mean() - period(main, "emi", EARLY).mean()
    no_uk = main[main["country_group"] != "United Kingdom"]
    nouk_change = period(no_uk, "emi", LATE).mean() - period(no_uk, "emi", EARLY).mean()
    print(f"raw change 1996-98 -> 2017-19: {raw_change:.3f}")
    print(f"excluding UK: {nouk_change:.3f}")
    print(f"within speaker: {within:.3f}  (share of raw change: {within / raw_change:.2f})")
    print("\nyear effects (speaker FE):\n" + eff.round(3).to_string())

    title("F. Family x term")
    df = main[main["term"].isin(["4th", "5th", "6th", "7th", "8th"])]
    ft = df.groupby(["family", "term"], observed=True)["emi"].agg(["mean", "std", "count"])
    ft["ci"] = 1.96 * ft["std"] / np.sqrt(ft["count"])
    print(ft[["mean", "ci", "count"]].round(3).to_string())
    print("\nfamily means over 1996-2019:")
    print(main.groupby("family")["emi"].agg(["mean", "count"]).sort_values("mean").round(3).to_string())

    title("G. Components")
    for col in ("evidence_z", "intuition_z"):
        e, l = period(main, col, EARLY).mean(), period(main, col, LATE).mean()
        print(f"{col}: {EARLY} {e:.3f} -> {LATE} {l:.3f}   change {l - e:+.3f}")

    title("H. Original vs translated, oral MEP speeches 1996-2012")
    tr = pd.read_parquet(ANNOTATED, columns=["emi", "year", "role", "written", "translated"])
    tr = tr[(tr["role"] == "mep") & ~tr["written"] & tr["year"].between(1996, 2012)]
    for flag, name in ((False, "original"), (True, "translated")):
        sub = tr[tr["translated"] == flag]
        print(f"{name}: 1996-98 {ci(period(sub, 'emi', (1996, 1998)))}   2010-12 {ci(period(sub, 'emi', (2010, 2012)))}")
    print("original languages of translations (top 10):")
    print(pd.read_parquet(ANNOTATED, columns=["orig_lang"])["orig_lang"].value_counts().head(10).to_string())

    title("I. Country groups")
    for grp in ("United Kingdom", "Ireland", "Other EU", "Unknown"):
        sub = main[main["country_group"] == grp]
        print(f"{grp}: {EARLY} {ci(period(sub, 'emi', EARLY))}   {LATE} {ci(period(sub, 'emi', LATE))}")

    title("J. Distribution by term")
    d = df.groupby("term", observed=True)["emi"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    d["share_negative"] = df.groupby("term", observed=True)["emi"].apply(lambda s: (s < 0).mean())
    print(d.round(3).to_string())

    title("K. Procedural filter manual validation")
    s = pd.read_csv(cfg.ROOT / "results" / "procedural_check_sample.csv")
    filled = s["controllo_manuale"].notna().sum()
    print(f"manual labels filled: {filled} / {len(s)}")
    if filled:
        print(pd.crosstab(s["controllo_manuale"], s["procedural"]).to_string())

    title("L. Examples (most evidence-like / intuition-like chunks)")
    path = TABLES / "S3_examples.csv"
    if path.exists():
        ex = pd.read_csv(path)
        ex["excerpt"] = ex["excerpt"].str.slice(0, 220)
        print(ex.groupby("end").head(4).to_string(index=False))


if __name__ == "__main__":
    main()
