"""REVIEW ONLY. Sensitivity variant 3: the baseline with all-mpnet-base-v2 instead of all-MiniLM-L6-v2.

Only the embedding model changes. Everything else is exactly as in steps 4-5: header kept,
chunks of at most 150 words (a final chunk shorter than 50 words joins the previous one),
lexicon vectors, cosine similarity, length correction, z-scores.

    uv run python notebooks/review/sensitivity/v3_mpnet_only.py --smoke   # quick test + time estimate
    uv run python notebooks/review/sensitivity/v3_mpnet_only.py           # full corpus
"""

from common import run
from europarl_emi import config as cfg

NAME = "v3_mpnet_only"
SETTINGS = {
    "strip_header": False,
    "chunk_max": cfg.CHUNK_MAX_WORDS,  # 150, as in the baseline
    "chunk_min": cfg.CHUNK_MIN_WORDS,  # 50, as in the baseline
    "model": "sentence-transformers/all-mpnet-base-v2",
    "batch_size": 64,  # longer chunks than variant 1: smaller batches to stay within GPU memory
}


def make_scorer():
    from sentence_transformers import SentenceTransformer, util

    from europarl_emi.step04_score_chunks import lexicon_vectors, pick_device

    model = SentenceTransformer(SETTINGS["model"], device=pick_device())
    print(f"Model {SETTINGS['model']} on {pick_device()} (max {model.max_seq_length} tokens)")
    reference = lexicon_vectors(model)  # mean embedding of the lexicon terms, exactly as in step 4

    def score(texts):
        emb = model.encode(texts, batch_size=SETTINGS["batch_size"], convert_to_tensor=True)
        sims = util.cos_sim(emb, reference).cpu().numpy()
        return sims[:, 0], sims[:, 1]

    return score


if __name__ == "__main__":
    run(NAME, SETTINGS, make_scorer)
