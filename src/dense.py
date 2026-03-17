"""
Dense retrieval via Chroma (PubMedBERT embeddings).
Returns (chunk_id, chunk_text, metadata) for RRF merge.
"""
from pathlib import Path
from typing import Any, Callable

import chromadb

DEFAULT_COLLECTION_NAME = "medical_rag"
DEFAULT_EMBEDDING_MODEL = "pubmedbert"


def _default_embedding_function():
    """Lazy-load a biomedical embedding function for query encoding."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

        def embed(texts: list[str]):
            return model.encode(texts, convert_to_numpy=True).tolist()

        return embed
    except Exception as e:
        raise RuntimeError(
            "Install sentence_transformers and ensure a biomedical model is available, "
            "or pass an embedding_function to ChromaManager."
        ) from e


class ChromaManager:
    def __init__(
        self,
        base_directory: str | Path,
        chunk_strategy: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        self.base_directory = Path(base_directory)
        self.chunk_strategy = chunk_strategy
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._path = self.base_directory / chunk_strategy / embedding_model
        if not self._path.exists():
            raise FileNotFoundError(f"Chroma path not found: {self._path}")
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._embed_fn = embedding_function
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def query(self, question: str, n_results: int = 30):
        """
        Return list of (chunk_id, chunk_text, metadata).
        chunk_id is taken from metadata['chunk_idx'] if present, else Chroma id.
        """
        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=n_results,
                include=["documents", "metadatas", "ids"],
            )
        except Exception:
            if self._embed_fn is None:
                self._embed_fn = _default_embedding_function()
            emb = self._embed_fn([question])
            results = self._collection.query(
                query_embeddings=emb,
                n_results=n_results,
                include=["documents", "metadatas", "ids"],
            )

        out = []
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []
        for doc, meta, id_ in zip(docs, metadatas or [], ids or []):
            meta = meta or {}
            chunk_id = meta.get("chunk_idx", id_)
            if isinstance(chunk_id, (int, float)) or (isinstance(chunk_id, str) and chunk_id.isdigit()):
                chunk_id = int(chunk_id)
            out.append((chunk_id, doc or "", meta))
        return out