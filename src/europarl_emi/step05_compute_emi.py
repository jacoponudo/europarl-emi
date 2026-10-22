"""Passo 5: calcolo dell'EMI su tutto il corpus e primo risultato nel tempo.

Entra:  data/interim/04_chunk_scores/            (similarita grezze per pezzo)
        data/interim/03_speeches_substantive.parquet  (data e metadati degli interventi)
Esce:   data/processed/05_chunk_emi.parquet       (EMI per pezzo di testo)
        data/processed/05_speech_emi.parquet      (EMI per intervento + metadati)
        results/emi_by_year.csv, results/figures/emi_by_year.png

Metodo (Aroyehun et al. 2025), applicato UNA volta a tutto il corpus:
1. correzione per lunghezza: a ogni similarita si toglie la media dei pezzi
   di lunghezza simile (fasce da 10 parole);
2. standardizzazione (z-score) sull'intero corpus;
3. EMI = z(evidenza) - z(intuizione).
Poi ogni intervento riceve la media dei suoi pezzi, cosi ogni intervento conta una volta.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from europarl_emi import config as cfg

SCORES = cfg.INTERIM / "04_chunk_scores"
SPEECHES = cfg.INTERIM / "03_speeches_substantive.parquet"
OUT_CHUNKS = cfg.PROCESSED / "05_chunk_emi.parquet"
OUT_SPEECHES = cfg.PROCESSED / "05_speech_emi.parquet"
TABLE = cfg.ROOT / "results" / "emi_by_year.csv"
FIGURE = cfg.FIGURES / "emi_by_year.png"

META = [
    "guid", "pubdate", "year", "detectedmepid", "detectedspeaker",
    "speaker-country", "detectedPoliticalGroupEU", "detectedNationalPoliticalGroup",
]


def compute_emi(chunks):
    bins = np.arange(cfg.SCORE_MIN_WORDS, chunks["n_words"].max() + 10, 10)
    length_bin = pd.cut(chunks["n_words"], bins)
    for name in ("evidence", "intuition"):
        sim = chunks[f"sim_{name}"]
        adjusted = sim - sim.groupby(length_bin, observed=True).transform("mean")
        chunks[f"{name}_z"] = (adjusted - adjusted.mean()) / adjusted.std()
    chunks["emi"] = chunks["evidence_z"] - chunks["intuition_z"]
    return chunks


def to_speeches(chunks):
    speech = chunks.groupby("guid").agg(
        emi=("emi", "mean"),
        evidence_z=("evidence_z", "mean"),
        intuition_z=("intuition_z", "mean"),
        n_chunks=("chunk_id", "size"),
        n_words=("n_words", "sum"),
    ).reset_index()
    available = pq.read_schema(SPEECHES).names
    meta = pd.read_parquet(SPEECHES, columns=[c for c in META if c in available])
    return speech.merge(meta, on="guid", how="left")


def by_year(speech):
    yearly = speech.groupby("year")["emi"].agg(["mean", "std", "count"])
    yearly["se"] = yearly["std"] / np.sqrt(yearly["count"])
    yearly["ci_low"] = yearly["mean"] - 1.96 * yearly["se"]
    yearly["ci_high"] = yearly["mean"] + 1.96 * yearly["se"]
    yearly.index = yearly.index.astype(int)
    return yearly


def plot(yearly):
    ink, muted, line = "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0, color=muted, linewidth=0.8, linestyle="--")
    ax.fill_between(yearly.index, yearly["ci_low"], yearly["ci_high"], color=line, alpha=0.18, linewidth=0)
    ax.plot(yearly.index, yearly["mean"], color=line, linewidth=2, marker="o", markersize=4)
    ax.set_title("EMI medio per anno, interventi in inglese al Parlamento europeo", loc="left", color=ink, fontsize=12)
    ax.set_ylabel("EMI (evidenza − intuizione)", color=muted)
    ax.set_xlabel("")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, length=0)
    fig.text(0.01, 0.01, "Media per intervento; banda = intervallo di confidenza al 95%.", color=muted, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=200)


def main():
    chunks = pd.read_parquet(SCORES)
    print(f"Pezzi di testo: {len(chunks):,}")
    chunks = compute_emi(chunks)
    chunks.to_parquet(OUT_CHUNKS, index=False)

    speech = to_speeches(chunks)
    speech.to_parquet(OUT_SPEECHES, index=False)
    print(f"Interventi con EMI: {len(speech):,}")
    print(f"Senza data: {speech['year'].isna().sum():,}")

    yearly = by_year(speech.dropna(subset=["year"]))
    yearly.round(4).to_csv(TABLE)
    plot(yearly)

    print("\nEMI medio per anno (per intervento):")
    print(yearly[["mean", "ci_low", "ci_high", "count"]].round(3).to_string())
    print(f"\nSalvati: {OUT_SPEECHES.relative_to(cfg.ROOT)}, {TABLE.relative_to(cfg.ROOT)}, {FIGURE.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
