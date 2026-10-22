"""Step 8: trend models with fixed effects and speaker-clustered standard errors.

Outcome: speech-level EMI. Main sample: oral speeches by MEPs originally delivered
in English, 1996-2019. The trend is expressed as change in EMI per decade.

Specifications (one row each in results/tables/trend_models.csv):
  1. Baseline                              no fixed effects
  2. Excluding UK MEPs                     no fixed effects
  3. Speaker fixed effects                 within-speaker change
  4. Agenda-item fixed effects             same agenda-item title across years
  5. Speaker + agenda-item fixed effects
  6. Original English, 1996-2012           same period as the translated sample
  7. Translated, 1996-2012                 MEP oral speeches translated into English
  8. Chunk-level outcome                   150-word chunks instead of speeches

Also saved: year-by-year effects with speaker fixed effects
(results/tables/year_effects_speaker_fe.csv), used in Figure 3.

Fixed effects are absorbed by alternating projections (iterative demeaning);
standard errors are cluster-robust (CR1) by speaker.
"""

import re

import numpy as np
import pandas as pd

from europarl_emi import config as cfg

MAIN = cfg.PROCESSED / "07_speech_main_sample.parquet"
ANNOTATED = cfg.PROCESSED / "06_speech_annotated.parquet"
CHUNKS = cfg.PROCESSED / "05_chunk_emi.parquet"
SPEECHES = cfg.INTERIM / "03_speeches_substantive.parquet"
TABLES = cfg.ROOT / "results" / "tables"
PERIOD = (1996, 2019)

UK = {"GB", "GBR", "UK", "UNITED KINGDOM"}
IE = {"IE", "IRL", "IRELAND"}
TERM_STARTS = {
    "4th": "1994-07-19", "5th": "1999-07-20", "6th": "2004-07-20", "7th": "2009-07-14",
    "8th": "2014-07-01", "9th": "2019-07-02", "10th": "2024-07-16",
}


# ---------------------------------------------------------------- data

def normalise_id(s):
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def agenda_item(title):
    """'6.10. Annual reports 2012-2013 (A8-0301/2015 - X)' -> 'annual reports 2012-2013'."""
    if not isinstance(title, str):
        return None
    t = re.sub(r"^\s*[\d.]+\s*", "", title)
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .-–").lower()
    return t or None


def country_group(country):
    c = country.astype("string").str.upper()
    out = pd.Series("Other EU", index=country.index, dtype="object")
    out[c.isin(UK)] = "United Kingdom"
    out[c.isin(IE)] = "Ireland"
    out[c.isna()] = "Unknown"
    return out


def add_term(df):
    edges = [pd.Timestamp(d, tz="UTC") for d in TERM_STARTS.values()] + [pd.Timestamp("2029-07-31", tz="UTC")]
    df["term"] = pd.cut(df["pubdate"], bins=edges, labels=list(TERM_STARTS), right=False)
    return df


def load_main(period=PERIOD):
    df = pd.read_parquet(MAIN)
    if "title" not in df.columns:
        df = df.merge(pd.read_parquet(SPEECHES, columns=["guid", "title"]), on="guid", how="left")
    df["pubdate"] = pd.to_datetime(df["pubdate"], utc=True)
    df["agenda_item"] = df["title"].map(agenda_item)
    df["country_group"] = country_group(df["country"])
    df["mep_id"] = normalise_id(df["mep_id"])
    df["cluster"] = df["mep_id"].fillna("speech:" + df["guid"].astype("string"))
    df = add_term(df)
    if period:
        df = df[df["year"].between(*period)]
    df["trend"] = (df["year"] - 2000) / 10
    return df.reset_index(drop=True)


def load_translated(period=(1996, 2012)):
    df = pd.read_parquet(ANNOTATED, columns=["guid", "emi", "year", "role", "written", "translated", "detectedmepid"])
    df = df[(df["role"] == "mep") & ~df["written"] & df["translated"] & df["year"].between(*period)].copy()
    df["mep_id"] = normalise_id(df["detectedmepid"])
    df["cluster"] = df["mep_id"].fillna("speech:" + df["guid"].astype("string"))
    df["trend"] = (df["year"] - 2000) / 10
    return df.reset_index(drop=True)


def load_chunks(main):
    chunks = pd.read_parquet(CHUNKS, columns=["guid", "emi"])
    return chunks.merge(main[["guid", "year", "trend", "cluster"]], on="guid", how="inner")


# ---------------------------------------------------------------- estimation

def drop_singletons(d, fes):
    while fes:
        keep = np.ones(len(d), dtype=bool)
        for fe in fes:
            keep &= d.groupby(fe)[fe].transform("size").to_numpy() > 1
        if keep.all():
            break
        d = d[keep]
    return d


def demean(d, cols, fes, tol=1e-9, max_iter=1000):
    out = d[cols].astype(float).reset_index(drop=True)
    keys = [d[fe].to_numpy() for fe in fes]
    for _ in range(max_iter):
        change = 0.0
        for k in keys:
            means = out.groupby(k).transform("mean")
            out = out - means
            change = max(change, float(np.abs(means.to_numpy()).max()))
        if change < tol:
            break
    return out


def ols_cluster(y, X, clusters):
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    scores = pd.DataFrame(X * resid[:, None]).groupby(clusters).sum().to_numpy()
    n, k = X.shape
    g = len(scores)
    adj = g / (g - 1) * (n - 1) / (n - k)
    vcov = adj * xtx_inv @ (scores.T @ scores) @ xtx_inv
    return beta, np.sqrt(np.diag(vcov))


def fit_trend(d, fes, label):
    d = d.dropna(subset=["emi", "trend", "cluster", *fes])
    d = drop_singletons(d, fes)
    if fes:
        dm = demean(d, ["emi", "trend"], fes)
        y, X = dm["emi"].to_numpy(), dm[["trend"]].to_numpy()
    else:
        y = d["emi"].to_numpy(float)
        X = np.column_stack([np.ones(len(d)), d["trend"].to_numpy(float)])
    beta, se = ols_cluster(y, X, d["cluster"].to_numpy())
    b, s = beta[-1], se[-1]
    return {
        "spec": label, "coef": b, "se": s, "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s,
        "n": len(d), "clusters": d["cluster"].nunique(),
    }


def year_effects(d, fes):
    d = d.dropna(subset=["emi", "cluster", *fes]).copy()
    d = drop_singletons(d, fes)
    years = sorted(d["year"].unique())
    cols = [f"y{y}" for y in years[1:]]
    for y, c in zip(years[1:], cols):
        d[c] = (d["year"] == y).astype(float)
    dm = demean(d, ["emi", *cols], fes)
    beta, se = ols_cluster(dm["emi"].to_numpy(), dm[cols].to_numpy(), d["cluster"].to_numpy())
    return pd.DataFrame({"year": years, "coef": [0.0, *beta], "se": [0.0, *se]})


# ---------------------------------------------------------------- main

def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    main_sample = load_main()
    no_uk = main_sample[main_sample["country_group"] != "United Kingdom"]
    print(f"Main sample {PERIOD[0]}-{PERIOD[1]}: {len(main_sample):,} speeches")

    specs = [
        (main_sample, [], "Baseline"),
        (no_uk, [], "Excluding UK MEPs"),
        (main_sample, ["mep_id"], "Speaker fixed effects"),
        (main_sample, ["agenda_item"], "Agenda-item fixed effects"),
        (main_sample, ["mep_id", "agenda_item"], "Speaker + agenda-item fixed effects"),
        (main_sample[main_sample["year"] <= 2012], [], "Original English, 1996–2012"),
        (load_translated(), [], "Translated, 1996–2012"),
        (load_chunks(main_sample), [], "Chunk-level outcome"),
    ]
    rows = []
    for data, fes, label in specs:
        print(f"  fitting: {label}")
        rows.append(fit_trend(data, fes, label))
    models = pd.DataFrame(rows)
    models.to_csv(TABLES / "trend_models.csv", index=False)

    effects = year_effects(main_sample, ["mep_id"])
    effects.to_csv(TABLES / "year_effects_speaker_fe.csv", index=False)

    print("\nChange in EMI per decade (95% CI), standard errors clustered by speaker:")
    show = models.copy()
    for c in ("coef", "se", "ci_low", "ci_high"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print(f"\nSaved: {(TABLES / 'trend_models.csv').relative_to(cfg.ROOT)}, "
          f"{(TABLES / 'year_effects_speaker_fe.csv').relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
