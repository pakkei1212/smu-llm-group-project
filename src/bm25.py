"""
BM25 retrieval: load pickle index, tokenize query, return top-k chunks.
"""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tokenizer import spacy_tokenize_texts, tokenize_query


def load_bm25_index(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


class BM25Retriever:
    """
    Query BM25 index and map scores to chunk DataFrame rows.
    chunks_df row order must match the order used when building the BM25 index.
    """

    def __init__(self, bm25_index: Any, chunks_df: pd.DataFrame, chunk_text_column: str = "chunk_text"):
        self.bm25 = bm25_index
        self.chunks_df = chunks_df.reset_index(drop=True)
        self.chunk_text_column = chunk_text_column
        if chunk_text_column not in self.chunks_df.columns:
            raise ValueError(f"Chunks DataFrame must have column '{chunk_text_column}'")

    def query(self, question: str, top_k: int = 30):
        """
        Return list of (chunk_id, chunk_text, metadata).
        chunk_id is the integer row index for RRF alignment with dense.
        """
        tokenized = tokenize_query(question)
        scores = self.bm25.get_scores(tokenized)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        out = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            row = self.chunks_df.iloc[idx]
            chunk_id = int(idx)
            text = row[self.chunk_text_column]
            meta = {k: row[k] for k in self.chunks_df.columns if k != self.chunk_text_column}
            out.append((chunk_id, str(text), meta))
        return out