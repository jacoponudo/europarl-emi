"""Speech headers in the Parliament's verbatim reports: who speaks, in what role, and whether the text
is a translation. Used by the API scripts (overlap_corpus_api.py, build_unified.py).

Every speech starts with a header, for example:
    "De Esteban Martin (PPE). - (ES) Mr President, ..."      -> translated from Spanish
    "Syed Kamall (PPE-DE). - Mr President, ..."              -> delivered in English
    "Pavel Telicka (ALDE), in writing. - Many Europeans ..."  -> written statement
    "Janez Lenarcic, President-in-Office of the Council. - (SL) ..." -> Council
    "President. - Thank you ..."                              -> chair

parse_header(text) returns (header, orig_lang, written, role, group_raw):
    orig_lang  original language if the text is translated (e.g. "ES"), None if delivered in English
    written    True for written statements ("in writing")
    role       mep / chair / commission / council / unknown
    group_raw  political group as written in the text (e.g. "PPE-DE")
"""

import re

HEADER = re.compile(r"^(?P<header>[^\n]{1,250}?)\.\s*[–−‒—-]\s*(?:\((?P<lang>[A-Z]{2})\)\s*)?")
GROUP_PAREN = re.compile(r"\(([^()]{1,25})\)")
GROUP_BEHALF = re.compile(r"on behalf of the (.{1,25}?) group", re.IGNORECASE)


def parse_header(text):
    m = HEADER.match(text or "")
    if not m:
        return None, None, False, "unknown", None
    header = m.group("header").strip()
    low = header.lower()
    written = "in writing" in low

    group = None
    g = GROUP_PAREN.search(header) or GROUP_BEHALF.search(header)
    if g:
        group = g.group(1).strip()

    if "president-in-office" in low or "of the council" in low or "minister" in low:
        role = "council"
    elif "commission" in low:
        role = "commission"
    elif low in ("president", "the president") or low.startswith("president "):
        role = "chair"
    elif group or "rapporteur" in low or "draftsman" in low or "on behalf of" in low:
        role = "mep"
    else:
        role = "unknown"
    return header, m.group("lang"), written, role, group
