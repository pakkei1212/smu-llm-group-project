"""
Shared tokenizer for BM25 query (must match index-time tokenization).
Uses spaCy; falls back to en_core_web_sm if en_core_sci_sm is unavailable.
"""
from typing import List

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_sci_sm")
        except OSError:
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def spacy_tokenize_texts(texts: List[str]) -> List[List[str]]:
    nlp = _get_nlp()
    out: List[List[str]] = []
    for doc in nlp.pipe(texts):
        toks = [t.text.lower() for t in doc if not t.is_punct and not t.is_space and t.text.strip()]
        out.append(toks)
    return out


def tokenize_query(query: str) -> List[str]:
    return spacy_tokenize_texts([query])[0]