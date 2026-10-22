"""Percorsi e parametri della pipeline, in un solo posto."""

from pathlib import Path

# Cartelle
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_CORPUS = DATA / "raw" / "corpus"
RAW_MEPS = DATA / "raw" / "meps"
LEXICONS = DATA / "raw" / "lexicons"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "results" / "figures"

# Riproducibilita
SEED = 42

# Pulizia del corpus (filterenglish.ipynb, celle 7-8)
MIN_WORDS = 3
MIN_CHARS = 16

# Suddivisione in pezzi e punteggio EMI (cella 30)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCORE_MIN_WORDS = 10
CHUNK_MAX_WORDS = 150
CHUNK_MIN_WORDS = 50

# Abbinamento parlamentari (mepdatabase.ipynb, cella 4)
FUZZY_THRESHOLD = 90

# Legislature
TERMS = {
    4: (1994, 1999),
    5: (1999, 2004),
    6: (2004, 2009),
    7: (2009, 2014),
    8: (2014, 2019),
    9: (2019, 2024),
    10: (2024, 2029),
}
