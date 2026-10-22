"""Passo 1: dal corpus grezzo teniamo solo gli interventi in inglese.

Entra:  data/raw/corpus/all_years_combined.csv.gzip  (tutte le lingue)
Esce:   data/interim/01_speeches_en.parquet          (solo inglese, testo intatto)
"""

import pandas as pd

from europarl_emi import config as C

INPUT = C.RAW_CORPUS / "all_years_combined.csv.gzip"
OUTPUT = C.INTERIM / "01_speeches_en.parquet"


def main():
    df = pd.read_csv(INPUT, compression="gzip", low_memory=False)
    print(f"Corpus grezzo:      {len(df):,} interventi")

    df = df[df["language"].str.lower() == "en"].copy()
    print(f"In inglese:         {len(df):,} interventi  (atteso: 340,771)")

    df = df[df["description"].notna()]
    print(f"Con testo presente: {len(df):,} interventi")

    df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce", utc=True, format="mixed")
    df["year"] = df["pubdate"].dt.year

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string")

    df.to_parquet(OUTPUT, index=False)
    print(f"\nSalvato in: {OUTPUT.relative_to(C.ROOT)}")

    print("\nInterventi in inglese per anno:")
    print(df["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
