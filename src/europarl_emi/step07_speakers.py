"""Passo 7: paese e gruppo di ogni oratore, e il calo dell'EMI a parita di oratori.

Campione principale: interventi ORALI, ORIGINALI IN INGLESE, di EURODEPUTATI.

- Paese: da detectedmepid, collegato al database degli eurodeputati (mep_id).
- Gruppo: dalla sigla nell'intestazione del testo (e' il gruppo al momento
  dell'intervento), raggruppata in famiglie politiche stabili nel tempo.

Tre versioni del trend nel tempo:
1. tutti gli interventi del campione principale;
2. senza i parlamentari britannici;
3. "a parita di oratori": l'EMI di ogni intervento meno la media del suo oratore.
   Se il calo resta, non dipende da chi entra o esce dal Parlamento,
   ma dal fatto che gli stessi oratori cambiano linguaggio nel tempo.

Entra:  data/processed/06_speech_annotated.parquet, data/processed/mep_database_1994_present.csv
Esce:   data/processed/07_speech_main_sample.parquet
        results/emi_by_year_speakers.csv, results/emi_by_year_family.csv
        results/figures/emi_by_year_speakers.png
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from europarl_emi import config as cfg

SPEECHES = cfg.PROCESSED / "06_speech_annotated.parquet"
MEPS = cfg.PROCESSED / "mep_database_1994_present.csv"
OUTPUT = cfg.PROCESSED / "07_speech_main_sample.parquet"
TABLE_TREND = cfg.ROOT / "results" / "emi_by_year_speakers.csv"
TABLE_FAMILY = cfg.ROOT / "results" / "emi_by_year_family.csv"
FIGURE = cfg.FIGURES / "emi_by_year_speakers.png"
MIN_PER_CELL = 100

FAMILIES = {
    "PPE": "EPP", "PPE-DE": "EPP", "EPP": "EPP",
    "PSE": "S&D", "S&D": "S&D", "S-D": "S&D", "PES": "S&D",
    "ELDR": "Liberals", "ALDE": "Liberals", "RENEW": "Liberals",
    "V": "Greens/EFA", "VERTS/ALE": "Greens/EFA", "GREENS/ALE": "Greens/EFA", "GREENS/EFA": "Greens/EFA",
    "GUE/NGL": "Left", "GUE": "Left", "LEFT": "Left", "THE LEFT": "Left",
    "UPE": "Conservatives", "UEN": "Conservatives", "ECR": "Conservatives",
    "EDN": "Eurosceptics", "I-EDN": "Eurosceptics", "EDD": "Eurosceptics", "IND/DEM": "Eurosceptics",
    "EFD": "Eurosceptics", "EFDD": "Eurosceptics", "ENF": "Eurosceptics", "ID": "Eurosceptics",
    "ITS": "Eurosceptics",
    "NI": "Non-attached", "TDI": "Non-attached",
    "ARE": "Other",
}
GROUP_PAREN = re.compile(r"\(([^()]{1,25})\)")


def clean_group(header):
    """Ultima sigla tra parentesi nell'intestazione (evita casi come 'The Earl of')."""
    if not isinstance(header, str):
        return None
    found = GROUP_PAREN.findall(header)
    if not found:
        m = re.search(r"on behalf of the (.{1,25}?) group", header, re.IGNORECASE)
        found = [m.group(1)] if m else []
    for raw in reversed(found):
        key = re.sub(r"[‐‑–−]", "-", raw).strip().upper()
        if key in FAMILIES:
            return key
    return found[-1].strip() if found else None


def country_of(meps):
    code = meps["country"].astype(str).str.strip().str.rstrip("/").str.split("/").str[-1].str.upper()
    ids = meps["mep_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    table = pd.DataFrame({"mep_id": ids, "country": code})
    table = table[~table["country"].isin(["NAN", "NONE", ""])]
    return table.groupby("mep_id")["country"].agg(lambda s: s.mode().iat[0])


def build_sample():
    df = pd.read_parquet(SPEECHES)
    df = df[(df["role"] == "mep") & ~df["written"] & ~df["translated"]].copy()

    df["group"] = df["header"].map(clean_group)
    df["family"] = df["group"].map(FAMILIES).fillna("Other")

    countries = country_of(pd.read_csv(MEPS))
    df["mep_id"] = df["detectedmepid"].astype(str).str.replace(r"\.0$", "", regex=True)
    df.loc[df["detectedmepid"].isna(), "mep_id"] = pd.NA
    df["country"] = df["mep_id"].map(countries)
    df["year"] = df["year"].astype(int)
    return df


def is_uk(country):
    return country.isin(["GB", "GBR", "UK", "UNITED KINGDOM"])


def trends(df):
    known = df[df["mep_id"].notna()].copy()
    known["emi_within"] = known["emi"] - known.groupby("mep_id")["emi"].transform("mean") + known["emi"].mean()

    series = {
        "tutti": df.groupby("year")["emi"].mean(),
        "senza britannici": df[~is_uk(df["country"].fillna(""))].groupby("year")["emi"].mean(),
        "a parita di oratori": known.groupby("year")["emi_within"].mean(),
    }
    counts = {
        "n tutti": df.groupby("year").size(),
        "n senza britannici": df[~is_uk(df["country"].fillna(""))].groupby("year").size(),
    }
    table = pd.DataFrame({**series, **counts})
    table.loc[table["n senza britannici"] < MIN_PER_CELL, "senza britannici"] = np.nan
    table.loc[table["n tutti"] < MIN_PER_CELL, ["tutti", "a parita di oratori"]] = np.nan
    return table


def plot(table):
    colors = {"tutti": "#2a78d6", "senza britannici": "#eb6834", "a parita di oratori": "#1baf7a"}
    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(0, color=muted, linewidth=0.8, linestyle="--")
    for name, color in colors.items():
        s = table[name].dropna()
        ax.plot(s.index, s.values, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.annotate(name, (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=muted, fontsize=9)
    ax.set_title("EMI per anno: interventi orali in inglese originale di eurodeputati",
                 loc="left", color=ink, fontsize=12)
    ax.set_ylabel("EMI (evidenza − intuizione)", color=muted)
    ax.legend(frameon=False, loc="lower left", labelcolor=muted)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, length=0)
    ax.set_xlim(table.index.min() - 0.5, table.index.max() + 4)
    fig.text(0.01, 0.01, f"Anni con meno di {MIN_PER_CELL} interventi esclusi. "
             "'A parita di oratori' = EMI meno la media di ciascun oratore.", color=muted, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURE, dpi=200)


def main():
    df = build_sample()
    df.to_parquet(OUTPUT, index=False)
    print(f"Campione principale: {len(df):,} interventi")
    print(f"con ID eurodeputato: {df['mep_id'].notna().mean():.1%}  |  con paese: {df['country'].notna().mean():.1%}")
    print("\nPaesi (i 10 piu frequenti):")
    print(df["country"].value_counts().head(10).to_string())
    print("\nFamiglie politiche:")
    print(df["family"].value_counts().to_string())
    print("\nSigle finite in 'Other' (le 10 piu frequenti):")
    print(df.loc[df["family"] == "Other", "group"].value_counts(dropna=False).head(10).to_string())

    table = trends(df)
    table.round(3).to_csv(TABLE_TREND)
    print("\nEMI per anno:")
    print(table.round(3).to_string())

    fam = df.pivot_table(index="year", columns="family", values="emi", aggfunc=["mean", "count"])
    means = fam["mean"].where(fam["count"] >= MIN_PER_CELL)
    means.round(3).to_csv(TABLE_FAMILY)
    print(f"\nEMI per anno e famiglia politica (celle con almeno {MIN_PER_CELL} interventi):")
    print(means.round(2).to_string())

    plot(table)
    print(f"\nSalvati: {OUTPUT.relative_to(cfg.ROOT)}, {TABLE_TREND.relative_to(cfg.ROOT)}, "
          f"{TABLE_FAMILY.relative_to(cfg.ROOT)}, {FIGURE.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
