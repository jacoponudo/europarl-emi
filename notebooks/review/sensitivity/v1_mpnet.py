"""REVIEW ONLY. Sensitivity variant 1: header removed, 50-word windows, all-mpnet-base-v2.

Changes with respect to the baseline (header kept, 150-word chunks, all-MiniLM-L6-v2):
  - the speech header ("Name (GROUP). - (XX)") is removed before scoring;
  - chunks of at most 50 words (a final chunk shorter than 17 words joins the previous one);
  - sentence embeddings from all-mpnet-base-v2 instead of all-MiniLM-L6-v2.
Everything else is as in steps 4-5 (lexicon vectors, cosine similarity, length correction, z-scores).

    uv run python notebooks/review/sensitivity/v1_mpnet.py --smoke   # quick test + time estimate
    uv run python notebooks/review/sensitivity/v1_mpnet.py           # full corpus
"""

from common import run

NAME = "v1_mpnet"
SETTINGS = {
    "strip_header": True,
    "chunk_max": 50,
    "chunk_min": 17,
    "model": "sentence-transformers/all-mpnet-base-v2",
    "batch_size": 128,
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
