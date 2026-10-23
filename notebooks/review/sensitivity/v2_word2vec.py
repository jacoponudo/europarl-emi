"""REVIEW ONLY. Sensitivity variant 2: as variant 1, but with word2vec instead of a sentence transformer.

Same text as variant 1 (header removed, 50-word windows). Each chunk and each lexicon is represented
by the mean of its (unit-length) word2vec vectors, the "distributed dictionary representation"
approach; similarity is the cosine between the two means. Then length correction and z-scores
exactly as in step 5.
  - model: word2vec-google-news-300 (3 million words and phrases, 300 dimensions), downloaded
    once by gensim (about 1.6 GB, into ~/gensim-data);
  - tokens: words of letters (with inner hyphens or apostrophes), looked up as written and then
    in lower case; words not in the vocabulary are skipped;
  - multi-word lexicon terms ("common sense") use the phrase vector when it exists
    ("common_sense"), otherwise the mean of their words.
gensim is not a dependency of the project: run with `uv run --with gensim`.

    uv run --with gensim python notebooks/review/sensitivity/v2_word2vec.py --smoke
    uv run --with gensim python notebooks/review/sensitivity/v2_word2vec.py
"""

import re

import numpy as np
import pandas as pd

from common import run
from europarl_emi import config as cfg

NAME = "v2_word2vec"
SETTINGS = {
    "strip_header": True,
    "chunk_max": 50,
    "chunk_min": 17,
    "model": "word2vec-google-news-300",
    "representation": "mean of unit-length word vectors; cosine with the mean lexicon vector",
}
TOKEN = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
BATCH = 2_000  # chunks summed at once (keeps memory low)


def make_scorer():
    import gensim.downloader

    kv = gensim.downloader.load(SETTINGS["model"])
    vectors = kv.vectors
    norms = np.linalg.norm(vectors, axis=1)
    index = kv.key_to_index
    cache = {}

    def lookup(token):
        if token not in cache:
            i = index.get(token, index.get(token.lower(), -1))
            cache[token] = i
        return cache[token]

    def unit(i):
        return vectors[i] / norms[i]

    def term_vector(term):
        words = term.split()
        phrase = lookup("_".join(words))
        if phrase >= 0:
            return unit(phrase)
        found = [lookup(w) for w in words]
        found = [unit(i) for i in found if i >= 0]
        return np.mean(found, axis=0) if found else None

    reference = []
    for name in ("evidence", "intuition"):
        terms = pd.read_csv(cfg.LEXICONS / f"{name}_lexicon.csv")[f"{name}_keywords"].dropna().astype(str)
        vecs = {t: term_vector(t) for t in terms}
        missing = [t for t, v in vecs.items() if v is None]
        print(f"Lexicon {name}: {len(terms)} terms, {len(terms) - len(missing)} found"
              + (f"; missing: {', '.join(missing)}" if missing else ""))
        m = np.mean([v for v in vecs.values() if v is not None], axis=0)
        reference.append(m / np.linalg.norm(m))
    reference = np.stack(reference, axis=1)  # 300 x 2
    stats = {"tokens": 0, "found": 0}

    def score(texts):
        out = np.full((len(texts), 2), np.nan)
        for b in range(0, len(texts), BATCH):
            ids = [[lookup(t) for t in TOKEN.findall(text)] for text in texts[b:b + BATCH]]
            stats["tokens"] += sum(map(len, ids))
            ids = [[i for i in row if i >= 0] for row in ids]
            stats["found"] += sum(map(len, ids))
            has = np.array([len(row) > 0 for row in ids])
            if not has.any():
                continue
            flat = np.concatenate([row for row in ids if row])
            starts = np.cumsum([0] + [len(row) for row in ids if row])[:-1]
            sums = np.add.reduceat(vectors[flat] / norms[flat, None], starts, axis=0)
            sums /= np.linalg.norm(sums, axis=1, keepdims=True)
            out[b + np.flatnonzero(has)] = sums @ reference
        print(f"  word coverage so far: {stats['found'] / max(stats['tokens'], 1):.1%} of tokens", flush=True)
        return out[:, 0], out[:, 1]

    return score


if __name__ == "__main__":
    run(NAME, SETTINGS, make_scorer)
