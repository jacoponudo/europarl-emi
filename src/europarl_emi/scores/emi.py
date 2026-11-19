"""EMI measure (evidence minus intuition, Aroyehun et al. 2025), as in the paper baseline.

Functions taken unchanged from the original pipeline (steps 4 and 5, in the git history up to ba52117):
    split_into_chunks  chunks of <= CHUNK_MAX_WORDS words; a last chunk < CHUNK_MIN_WORDS joins the previous one
    pick_device        mps / cuda / cpu
    lexicon_vectors    mean embedding of the evidence and intuition lexicons
    compute_emi        length correction (10-word bins), z-score over all chunks given, EMI = z(ev) - z(int)
"""

import numpy as np
import pandas as pd

from europarl_emi import config as cfg


def split_into_chunks(text):
    """Pezzi da massimo CHUNK_MAX_WORDS parole; un ultimo pezzo troppo corto si unisce al precedente."""
    words = text.split()
    size = cfg.CHUNK_MAX_WORDS
    chunks = [words[i:i + size] for i in range(0, len(words), size)]
    if len(chunks) > 1 and len(chunks[-1]) < cfg.CHUNK_MIN_WORDS:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + last
    return [" ".join(c) for c in chunks]


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def lexicon_vectors(model):
    import torch

    vectors = []
    for name in ("evidence", "intuition"):
        lex = pd.read_csv(cfg.LEXICONS / f"{name}_lexicon.csv")
        words = lex[f"{name}_keywords"].dropna().astype(str).tolist()
        vectors.append(model.encode(words, convert_to_tensor=True).mean(dim=0))
        print(f"Lessico {name}: {len(words)} parole")
    return torch.stack(vectors)


def compute_emi(chunks):
    bins = np.arange(cfg.SCORE_MIN_WORDS, chunks["n_words"].max() + 10, 10)
    length_bin = pd.cut(chunks["n_words"], bins)
    for name in ("evidence", "intuition"):
        sim = chunks[f"sim_{name}"]
        adjusted = sim - sim.groupby(length_bin, observed=True).transform("mean")
        chunks[f"{name}_z"] = (adjusted - adjusted.mean()) / adjusted.std()
    chunks["emi"] = chunks["evidence_z"] - chunks["intuition_z"]
    return chunks
