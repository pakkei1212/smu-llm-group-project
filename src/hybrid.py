"""
Hybrid retrieval: merge dense and BM25 rankings with Reciprocal Rank Fusion (RRF).
"""
from typing import Any, Callable

RRF_K = 30
DENSE_WEIGHT = 1.0
BM25_WEIGHT  = 1.0   # 🔥 increase this

def rrf_merge(
    dense_results: list[tuple[Any, str, dict]],
    bm25_results: list[tuple[Any, str, dict]],
    k: int = RRF_K,
    n_final: int = 10,
):
    scores: dict[Any, float] = {}
    chunk_data: dict[Any, tuple[str, dict]] = {}

    for rank, (cid, text, meta) in enumerate(dense_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + DENSE_WEIGHT * (1.0 / (k + rank))
        chunk_data[cid] = (text, meta)

    for rank, (cid, text, meta) in enumerate(bm25_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + BM25_WEIGHT * (1.0 / (k + rank))
        if cid not in chunk_data:
            chunk_data[cid] = (text, meta)

    sorted_ids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:n_final]
    return [(cid, chunk_data[cid][0], chunk_data[cid][1]) for cid in sorted_ids]


def hybrid_retrieve(
    question: str,
    dense_query: Callable[[str, int], list[tuple[Any, str, dict]]],
    bm25_query: Callable[[str, int], list[tuple[Any, str, dict]]],
    n_dense: int = 40,
    n_bm25: int = 40,
    n_final: int = 5,
    rrf_k: int = RRF_K,
):
    dense_results = dense_query(question, n_dense)
    bm25_results = bm25_query(question, n_bm25)
    return rrf_merge(dense_results, bm25_results, k=rrf_k, n_final=n_final)