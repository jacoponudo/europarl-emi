"""Passo 6: chi parla, in che ruolo, e se il testo e' una traduzione.

Ogni intervento nei resoconti del Parlamento inizia con un'intestazione, ad esempio:
    "De Esteban Martin (PPE). - (ES) Mr President, ..."      -> tradotto dallo spagnolo
    "Syed Kamall (PPE-DE). - Mr President, ..."              -> originale in inglese
    "Pavel Telicka (ALDE), in writing. - Many Europeans ..."  -> dichiarazione scritta
    "Janez Lenarcic, President-in-Office of the Council. - (SL) ..." -> Consiglio
    "President. - Thank you ..."                              -> presidenza di seduta

Da questa intestazione ricaviamo:
- orig_lang:  lingua originale se il testo e' tradotto (es. "ES"), vuota se originale inglese
- written:    True per le dichiarazioni scritte ("in writing")
- role:       mep / chair / commission / council / unknown
- group_raw:  sigla del gruppo politico come scritta nel testo (es. "PPE-DE")

Entra:  data/processed/05_speech_emi.parquet, data/interim/03_speeches_substantive.parquet
Esce:   data/processed/06_speech_annotated.parquet
        results/emi_by_year_original_vs_translated.csv
        results/figures/emi_by_year_original_vs_translated.png
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from europarl_emi import config as cfg

EMI = cfg.PROCESSED / "05_speech_emi.parquet"
SPEECHES = cfg.INTERIM / "03_speeches_substantive.parquet"
OUTPUT = cfg.PROCESSED / "06_speech_annotated.parquet"
TABLE = cfg.ROOT / "results" / "emi_by_year_original_vs_translated.csv"
FIGURE = cfg.FIGURES / "emi_by_year_original_vs_translated.png"
MIN_SPEECHES_PER_YEAR = 200

HEADER = re.compile(r"^(?P<header>[^\n]{1,250}?)\.\s*[–−‒—-]\s*(?:\((?P<lang>[A-Z]{2})\)\s*)?")
GROUP_PAREN = re.compile(r"\(([^()]{1,25})\)")
GROUP_BEHALF = re.compile(r"on behalf of the (.{1,25}?) group", re.IGNORECASE)


def parse_header(text):
    m = HEADER.match(text or "")
    if not m:
        return None, None, False, "unknown", None
    header = m.group("header").strip()
    low = header.lower()
    written = "in writing" in low

    group = None
    g = GROUP_PAREN.search(header) or GROUP_BEHALF.search(header)
    if g:
        group = g.group(1).strip()

    if "president-in-office" in low or "of the council" in low or "minister" in low:
        role = "council"
    elif "commission" in low:
        role = "commission"
    elif low in ("president", "the president") or low.startswith("president "):
        role = "chair"
    elif group or "rapporteur" in low or "draftsman" in low or "on behalf of" in low:
        role = "mep"
    else:
        role = "unknown"
    return header, m.group("lang"), written, role, group


def annotate():
    emi = pd.read_parquet(EMI)
    extra = pd.read_parquet(SPEECHES, columns=["guid", "text", "speaker", "speaker-title"])
    df = emi.merge(extra, on="guid", how="left")
    parsed = pd.DataFrame(
        [parse_header(t) for t in df["text"]],
        columns=["header", "orig_lang", "written", "role", "group_raw"],
        index=df.index,
    )
    df = pd.concat([df, parsed], axis=1)
    df["header_found"] = df["header"].notna()
    df["translated"] = df["orig_lang"].notna()
    return df.drop(columns="text")


def summary(df):
    print(f"Interventi: {len(df):,}")
    print(f"Intestazione riconosciuta: {df['header_found'].mean():.1%}")
    print("\nRuoli:")
    print(df["role"].value_counts().to_string())
    print("\nLingue originali dei testi tradotti (le 10 piu frequenti):")
    print(df["orig_lang"].value_counts().head(10).to_string())

    found = df[df["header_found"]]
    by_year = pd.DataFrame({
        "interventi": found.groupby("year").size(),
        "tradotti": found.groupby("year")["translated"].mean(),
        "scritti": found.groupby("year")["written"].mean(),
        "eurodeputati": found.groupby("year")["role"].apply(lambda s: (s == "mep").mean()),
        "presidenza": found.groupby("year")["role"].apply(lambda s: (s == "chair").mean()),
        "commissione": found.groupby("year")["role"].apply(lambda s: (s == "commission").mean()),
        "consiglio": found.groupby("year")["role"].apply(lambda s: (s == "council").mean()),
    })
    by_year.index = by_year.index.astype(int)
    print("\nComposizione per anno (quote):")
    print(by_year.round(2).to_string())


def compare(df):
    oral_mep = df[(df["role"] == "mep") & ~df["written"]]
    oral_mep = oral_mep.assign(versione=np.where(oral_mep["translated"], "tradotto", "originale inglese"))
    stats = oral_mep.groupby(["year", "versione"])["emi"].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["year"] = stats["year"].astype(int)
    stats.loc[stats["count"] < MIN_SPEECHES_PER_YEAR, ["mean", "se"]] = np.nan
    table = stats.pivot(index="year", columns="versione", values=["mean", "count"]).round(3)
    table.to_csv(TABLE)
    print("\nEMI per anno, solo interventi ORALI di EURODEPUTATI:")
    print(table.to_string())
    return stats


def plot(stats):
    colors = {"originale inglese": "#2a78d6", "tradotto": "#eb6834"}
    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color=muted, linewidth=0.8, linestyle="--")
    for name, g in stats.groupby("versione"):
        g = g.sort_values("year")
        ax.fill_between(g["year"], g["mean"] - 1.96 * g["se"], g["mean"] + 1.96 * g["se"],
                        color=colors[name], alpha=0.15, linewidth=0)
        ax.plot(g["year"], g["mean"], color=colors[name], linewidth=2, marker="o", markersize=4, label=name)
        last = g.dropna(subset=["mean"]).iloc[-1]
        ax.annotate(name, (last["year"], last["mean"]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=muted, fontsize=9)
    ax.set_title("EMI medio per anno: interventi orali di eurodeputati, originali vs tradotti",
                 loc="left", color=ink, fontsize=12)
    ax.set_ylabel("EMI (evidenza − intuizione)", color=muted)
    ax.legend(frameon=False, loc="lower left", labelcolor=muted)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, length=0)
    ax.set_xlim(stats["year"].min() - 0.5, stats["year"].max() + 3)
    fig.text(0.01, 0.01, f"Banda = IC 95%. Anni con meno di {MIN_SPEECHES_PER_YEAR} interventi esclusi.",
             color=muted, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURE, dpi=200)


def main():
    df = annotate()
    df.to_parquet(OUTPUT, index=False)
    summary(df)
    stats = compare(df)
    plot(stats)
    print(f"\nSalvati: {OUTPUT.relative_to(cfg.ROOT)}, {TABLE.relative_to(cfg.ROOT)}, {FIGURE.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
