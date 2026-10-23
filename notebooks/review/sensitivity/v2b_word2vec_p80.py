"""REVIEW ONLY. Sensitivity variant 2b: word2vec, 80th percentile of word similarities.

Same text and model as variant 2 (header removed, 50-word windows, word2vec-google-news-300).
The only change is how a chunk is scored:
  variant 2   cosine between the mean word vector of the chunk and the lexicon vector;
  variant 2b  cosine of EACH word with the lexicon vector, then the 80th percentile within the chunk.
The percentile gives weight to the few words most related to the lexicon, instead of the
overall tone of the chunk. Then length correction and z-scores exactly as in step 5.
gensim is not a dependency of the project: run with `uv run --with gensim`.

    uv run --with gensim python notebooks/review/sensitivity/v2b_word2vec_p80.py --smoke
    uv run --with gensim python notebooks/review/sensitivity/v2b_word2vec_p80.py
"""

import re

import numpy as np
import pandas as pd

from common import run
from europarl_emi import config as cfg

NAME = "v2b_word2vec_p80"
PERCENTILE = 80
SETTINGS = {
    "strip_header": True,
    "chunk_max": 50,
    "chunk_min": 17,
    "model": "word2vec-google-news-300",
    "representation": f"{PERCENTILE}th percentile, within the chunk, of the cosine of each word with the lexicon vector",
}
TOKEN = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")  # same tokens as variant 2
BATCH = 2_000


def make_scorer():
    import gensim.downloader

    kv = gensim.downloader.load(SETTINGS["model"])
    vectors = kv.vectors
    norms = np.linalg.norm(vectors, axis=1)
    index = kv.key_to_index
    cache = {}

    def lookup(token):
        if token not in cache:
            cache[token] = index.get(token, index.get(token.lower(), -1))
        return cache[token]

    def unit(i):
        return vectors[i] / norms[i]

    def term_vector(term):  # as in variant 2: phrase vector if it exists, else mean of its words
        words = term.split()
        phrase = lookup("_".join(words))
        if phrase >= 0:
            return unit(phrase)
        found = [unit(i) for i in map(lookup, words) if i >= 0]
        return np.mean(found, axis=0) if found else None

    reference = []
    for name in ("evidence", "intuition"):
        terms = pd.read_csv(cfg.LEXICONS / f"{name}_lexicon.csv")[f"{name}_keywords"].dropna().astype(str)
        vecs = [v for v in map(term_vector, terms) if v is not None]
        print(f"Lexicon {name}: {len(terms)} terms, {len(vecs)} found")
        m = np.mean(vecs, axis=0)
        reference.append(m / np.linalg.norm(m))
    reference = np.stack(reference, axis=1)  # 300 x 2

    def score(texts):
        out = np.full((len(texts), 2), np.nan)
        for b in range(0, len(texts), BATCH):
            ids = [[i for i in map(lookup, TOKEN.findall(text)) if i >= 0] for text in texts[b:b + BATCH]]
            rows = [r for r, row in enumerate(ids) if row]
            if not rows:
                continue
            flat = np.concatenate([ids[r] for r in rows])
            owner = np.repeat(rows, [len(ids[r]) for r in rows])
            sims = (vectors[flat] / norms[flat, None]) @ reference  # one row per word
            q = pd.DataFrame(sims, columns=["e", "i"]).groupby(owner).quantile(PERCENTILE / 100)
            out[b + q.index.to_numpy()] = q.to_numpy()
        return out[:, 0], out[:, 1]

    return score


if __name__ == "__main__":
    run(NAME, SETTINGS, make_scorer)
