"""The unified dataset with the English text of every speech: delivered in English or translated with eTranslation.

    from europarl_emi.unified import load
    u = load()                      # MEP oral speeches, metadata and group
    u = load(with_text=True)        # also column `en`, the English text

Groups (column `group`):
  original      english_text == "text (original)"; English text = column `text`
  etranslation  detected as non-English and all eTranslation segments done (from 2013); English text =
                the segments joined in order (header included, as in `text`)
  human         english_text == "text (human translation)": the Parliament's translation (to 2012),
                not used for the EMI
  other         no English text
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from europarl_emi import config as cfg

MEP_ORAL = "MEP, oral"


def translations(db: Path = cfg.ETRANSLATION_DB, with_text: bool = True) -> pd.DataFrame:
    """uid (and English text) of the non-English texts whose segments are all translated."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tx = pd.read_sql("SELECT text_id FROM texts WHERE lang IS NOT NULL AND lang != 'EN'", con)
    cols = "text_id, idx, status, translation" if with_text else "text_id, status"
    sg = pd.read_sql(f"SELECT {cols} FROM segments", con)
    con.close()
    full = sg.groupby("text_id").status.agg(lambda s: s.eq("done").all())
    done = set(full[full].index) & set(tx.text_id)
    print(f"{db.name}: {len(tx):,} non-English texts, {len(done):,} fully translated, "
          f"segments {sg.status.value_counts().to_dict()}")
    if not with_text:
        return pd.DataFrame({"uid": sorted(done)})
    sg = sg[sg.text_id.isin(done)].sort_values(["text_id", "idx"])
    out = sg.groupby("text_id").translation.agg(lambda s: " ".join(s.fillna(""))).rename("text_en").reset_index()
    return out.rename(columns={"text_id": "uid"})


def load(population: str = "mep-oral", with_text: bool = False,
         unified: Path = cfg.UNIFIED, db: Path = cfg.ETRANSLATION_DB) -> pd.DataFrame:
    cols = ["uid", "year", "date", "category", "english_text"] + (["text"] if with_text else [])
    u = pd.read_parquet(unified, columns=cols)
    print(f"{unified.name}: {len(u):,} rows, {u.date.min().date()} to {u.date.max().date()}")
    if population == "mep-oral":
        u = u[u.category.astype("string").str.startswith(MEP_ORAL, na=False)].copy()
        print(f"MEP oral speeches: {len(u):,}")
    tr = translations(db, with_text)
    et = u.english_text.astype("string").fillna("none")
    u["group"] = "other"
    u.loc[et.eq("text (original)"), "group"] = "original"
    u.loc[et.eq("text (human translation)"), "group"] = "human"
    u.loc[u.uid.isin(tr.uid) & ~et.eq("text (original)"), "group"] = "etranslation"
    if with_text:
        u = u.merge(tr, on="uid", how="left")
        u["en"] = u.text.where(u.group.eq("original"), u.text_en.where(u.group.eq("etranslation")))
        u = u.drop(columns=["text", "text_en"])
    u["year"] = u.year.astype(int)
    print("groups:", u.group.value_counts().to_dict())
    return u.reset_index(drop=True)
