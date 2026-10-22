"""Passo 2: togliamo i testi troppo corti e sistemiamo gli spazi.

Entra:  data/interim/01_speeches_en.parquet
Esce:   data/interim/02_speeches_clean.parquet  (nuova colonna "text")

Maiuscole e punteggiatura restano: servono al modello di embedding.
"""

import pandas as pd

from europarl_emi import config as C

INPUT = C.INTERIM / "01_speeches_en.parquet"
OUTPUT = C.INTERIM / "02_speeches_clean.parquet"


def main():
    df = pd.read_parquet(INPUT)
    print(f"Interventi in ingresso: {len(df):,}")

    words = df["description"].str.split().str.len()
    chars = df["description"].str.len()
    keep = (words >= C.MIN_WORDS) & (chars >= C.MIN_CHARS)
    df = df[keep].copy()
    print(f"Tolti perche troppo corti: {(~keep).sum():,}")
    print(f"Interventi rimasti: {len(df):,}  (atteso: 340,606)")

    df["text"] = df["description"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["n_words"] = df["text"].str.split().str.len()

    df.to_parquet(OUTPUT, index=False)
    print(f"\nSalvato in: {OUTPUT.relative_to(C.ROOT)}")
    print("\nLunghezza degli interventi (parole):")
    print(df["n_words"].describe().round(0).to_string())


if __name__ == "__main__":
    main()
