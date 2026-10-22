"""Passo 4: similarita di ogni pezzo di testo con i due lessici.

Entra:  data/interim/03_speeches_substantive.parquet
        data/raw/lexicons/evidence_lexicon.csv, intuition_lexicon.csv
Esce:   data/interim/04_chunk_scores/part-XXXX.parquet
        (una riga per pezzo: guid, chunk_id, n_words, sim_evidence, sim_intuition)

Qui NON calcoliamo l'EMI: salviamo solo le similarita grezze.
L'EMI si calcola al passo 5, su tutto il corpus insieme.

Se lo script si interrompe, rilancialo: riparte dalle parti mancanti.
"""

import math

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from europarl_emi import config as cfg

INPUT = cfg.INTERIM / "03_speeches_substantive.parquet"
OUT_DIR = cfg.INTERIM / "04_chunk_scores"
PART_SIZE = 20_000


def split_into_chunks(text):
    """Pezzi da massimo CHUNK_MAX_WORDS parole; un ultimo pezzo troppo corto si unisce al precedente."""
    words = text.split()
    size = cfg.CHUNK_MAX_WORDS
    chunks = [words[i:i + size] for i in range(0, len(words), size)]
    if len(chunks) > 1 and len(chunks[-1]) < cfg.CHUNK_MIN_WORDS:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + last
    return [" ".join(c) for c in chunks]


def make_chunks(df):
    df = df[df["n_words"] > cfg.SCORE_MIN_WORDS]
    rows = [
        (guid, i, chunk)
        for guid, text in zip(df["guid"], df["text"])
        for i, chunk in enumerate(split_into_chunks(text))
    ]
    chunks = pd.DataFrame(rows, columns=["guid", "chunk_id", "chunk_text"])
    chunks["n_words"] = chunks["chunk_text"].str.split().str.len()
    return chunks


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def lexicon_vectors(model):
    vectors = []
    for name in ("evidence", "intuition"):
        lex = pd.read_csv(cfg.LEXICONS / f"{name}_lexicon.csv")
        words = lex[f"{name}_keywords"].dropna().astype(str).tolist()
        vectors.append(model.encode(words, convert_to_tensor=True).mean(dim=0))
        print(f"Lessico {name}: {len(words)} parole")
    return torch.stack(vectors)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT, columns=["guid", "text", "n_words"])
    chunks = make_chunks(df)
    print(f"Interventi in ingresso: {len(df):,}")
    print(f"Pezzi di testo da analizzare: {len(chunks):,}")

    device = pick_device()
    print(f"Dispositivo di calcolo: {device}")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
    reference = lexicon_vectors(model)

    n_parts = math.ceil(len(chunks) / PART_SIZE)
    for p in tqdm(range(n_parts), desc="Parti"):
        path = OUT_DIR / f"part-{p:04d}.parquet"
        if path.exists():
            continue
        part = chunks.iloc[p * PART_SIZE:(p + 1) * PART_SIZE].copy()
        emb = model.encode(part["chunk_text"].tolist(), batch_size=128, convert_to_tensor=True)
        sims = util.cos_sim(emb, reference).cpu().numpy()
        part["sim_evidence"] = sims[:, 0]
        part["sim_intuition"] = sims[:, 1]
        tmp = path.with_suffix(".tmp")
        part.drop(columns="chunk_text").to_parquet(tmp, index=False)
        tmp.rename(path)

    scores = pd.read_parquet(OUT_DIR)
    print(f"\nPezzi analizzati: {len(scores):,}  (attesi: {len(chunks):,})")
    print(scores[["sim_evidence", "sim_intuition"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
