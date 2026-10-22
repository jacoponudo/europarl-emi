"""Passo 3: togliamo i testi procedurali ("the debate is closed", "before the vote"...).

Entra:  data/interim/02_speeches_clean.parquet
Esce:   data/interim/03_speeches_substantive.parquet  (solo testi non procedurali)
        models/procedural_nblr.joblib                 (il classificatore)
        results/procedural_check_sample.csv           (200 esempi da controllare a mano)

Metodo (Card et al. 2022, ripreso da Aroyehun et al. 2025 e dal vecchio notebook):
- esempi "procedurali": testi brevi (<200 caratteri) ripetuti identici molte volte;
- esempi "non procedurali": pezzi di interventi lunghi;
- classificatore NB-LR su parole e sequenze di 2-3 parole;
- il classificatore si applica solo ai testi brevi (<400 caratteri):
  gli interventi piu lunghi sono considerati sempre non procedurali.
"""

import re
import string
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import diags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from europarl_emi import config as cfg

INPUT = cfg.INTERIM / "02_speeches_clean.parquet"
OUTPUT = cfg.INTERIM / "03_speeches_substantive.parquet"
MODEL = cfg.MODELS / "procedural_nblr.joblib"
SAMPLE = cfg.ROOT / "results" / "procedural_check_sample.csv"
OLD = cfg.INTERIM / "filtered_nov2.csv"

MAX_CHARS_TO_CLASSIFY = 400
PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def simplify(text):
    """Minuscolo, senza punteggiatura: solo per il classificatore, il testo resta intatto."""
    return re.sub(r"\s+", " ", PUNCT.sub("", text)).strip().lower()


def build_examples(texts, rng):
    """Esempi di addestramento e validazione con le stesse regole del vecchio notebook."""
    counter = Counter(t for t in texts if len(t) < 200)
    train, val = [], []

    # Non procedurali: pezzi di interventi lunghi, tenuti con probabilita 1/50
    for t in texts:
        if len(t) <= 1000:
            continue
        tokens = t.split()
        if len(tokens) <= 20:
            continue
        n_pieces = len(t) // 200 + 1
        per = len(tokens) // n_pieces
        for n in range(n_pieces):
            sub = tokens[n * per:(n + 1) * per] if n < n_pieces - 1 else tokens[n * per:]
            piece = " ".join(sub)
            if len(piece) < 1000 and rng.random() < 1 / 50:
                (val if rng.random() < 1 / 20 else train).append((piece, 0))

    # Procedurali: testi brevi ripetuti (>=20 volte in train, 10-19 volte in validazione)
    for t, count in counter.items():
        if 10 <= count < 20:
            val.append((t, 1))
        elif count >= 20:
            train.extend([(t, 1)] * (count // 10))
    return train, val


def fit(train):
    texts, y = zip(*train)
    y = np.array(y)
    vec = CountVectorizer(ngram_range=(1, 3), binary=True, token_pattern=r"\S+")
    X = vec.fit_transform(texts)
    p = 1 + np.asarray(X[y == 1].sum(axis=0)).ravel()
    q = 1 + np.asarray(X[y == 0].sum(axis=0)).ravel()
    r = np.log((p / p.sum()) / (q / q.sum()))
    clf = LogisticRegression(max_iter=1000, random_state=cfg.SEED)
    clf.fit(X @ diags(r), y)
    return {"vectorizer": vec, "r": r, "classifier": clf}


def predict(model, texts):
    X = model["vectorizer"].transform(texts) @ diags(model["r"])
    return model["classifier"].predict_proba(X)[:, 1]


def compare_with_old(df):
    if not OLD.exists():
        return
    old_ids = set(pd.read_csv(OLD, usecols=["interventionID"], dtype=str)["interventionID"])
    old_keep = df["interventionID"].astype(str).isin(old_ids).rename("vecchio: tenuto")
    new_keep = (~df["procedural"]).rename("nuovo: tenuto")
    print("\nConfronto con il vecchio filtro (numero di interventi):")
    print(pd.crosstab(new_keep, old_keep).to_string())


def main():
    rng = np.random.default_rng(cfg.SEED)
    df = pd.read_parquet(INPUT)
    print(f"Interventi in ingresso: {len(df):,}")
    simple = df["text"].map(simplify)

    train, val = build_examples(simple.tolist(), rng)
    print(f"Esempi di addestramento: {len(train):,}  |  di validazione: {len(val):,}")

    print("Addestramento del classificatore...")
    model = fit(train)
    joblib.dump(model, MODEL)

    val_texts, val_y = zip(*val)
    val_pred = predict(model, list(val_texts)) >= 0.5
    cm = confusion_matrix(val_y, val_pred, labels=[0, 1])
    print("\nValidazione (etichette costruite con le regole, non a mano):")
    print(pd.DataFrame(cm, index=["vero: no", "vero: si"], columns=["previsto: no", "previsto: si"]).to_string())

    short = (simple.str.len() < MAX_CHARS_TO_CLASSIFY).to_numpy()
    proba = np.zeros(len(df))
    proba[short] = predict(model, simple[short].tolist())
    df["procedural_proba"] = proba
    df["procedural"] = proba >= 0.5
    print(f"\nTesti brevi esaminati: {short.sum():,}")
    print(f"Classificati come procedurali: {df['procedural'].sum():,}")

    examined = df[short]
    parts = [examined[examined["procedural"] == flag] for flag in (True, False)]
    sample = pd.concat(g.sample(min(100, len(g)), random_state=cfg.SEED) for g in parts)
    sample = sample[["interventionID", "text", "procedural", "procedural_proba"]].assign(controllo_manuale="")
    sample.to_csv(SAMPLE, index=False)

    kept = df[~df["procedural"]]
    kept.to_parquet(OUTPUT, index=False)
    print(f"Interventi tenuti: {len(kept):,}")
    print(f"\nSalvato in: {OUTPUT.relative_to(cfg.ROOT)}")
    print(f"Campione da controllare: {SAMPLE.relative_to(cfg.ROOT)}")

    compare_with_old(df)


if __name__ == "__main__":
    main()
